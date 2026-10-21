#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF loader and forward for `general.architecture = glm5-next`.
//!
//! KDA layers call [`kda_rule_recurrence`]. DSA layers attend over a cached
//! compressed KV latent in the absorbed MLA form. Hyper-connections use the mix
//! from the same sublayer. Routed experts stay quantized in [`MoEExperts`]. This
//! is not GLM-4.
//!
//! The DSA lightning indexer is not implemented. Below its `top_k` the indexer
//! selects every key, so dense attention is exact there; longer sequences are
//! refused rather than attended differently from the model.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    GgufArchive, GgufBindingMap, GgufMatMul, GgufTensorBinding, GgufWeightSource, QuantMethod,
    QuantMethodConfig,
};

use crate::gdn::kda_rule_recurrence;
use crate::gguf::deepseek41::{
    hyper_mix_projection, rmsnorm, swiglu_clamped, unlagged_block, HyperWeights,
};
use crate::kv_cache::{EitherCache, NormalCache};
use crate::layers::Activation;
use crate::moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig};
use crate::pipeline::extract_logits;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

const ARCH: &str = "glm5-next";

pub struct ModelWeights {
    embed: Arc<dyn QuantMethod>,
    layers: Vec<Layer>,
    norm: Tensor,
    output: Arc<dyn QuantMethod>,
    hc: usize,
    sinkhorn_iters: usize,
    rms_eps: f32,
    hc_eps: f32,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    state: Mutex<RunState>,
}

enum LayerKind {
    Kda(Kda),
    Dsa(Dsa),
}

struct Layer {
    kind: LayerKind,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc_attn: HyperWeights,
    hc_ffn: HyperWeights,
    ffn: Ffn,
}

struct Kda {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    /// The q, k and v depthwise convolutions joined as one `[channels, kernel]` f32 matrix.
    conv: Tensor,
    f_a: Arc<dyn QuantMethod>,
    f_b: Arc<dyn QuantMethod>,
    dt_bias: Tensor,
    a: Tensor,
    beta: Arc<dyn QuantMethod>,
    g_a: Arc<dyn QuantMethod>,
    g_b: Arc<dyn QuantMethod>,
    norm: Tensor,
    output: Arc<dyn QuantMethod>,
    heads: usize,
    head_dim: usize,
    kernel: usize,
    lower_bound: f64,
    eps: f64,
}

struct Dsa {
    q_a: Arc<dyn QuantMethod>,
    q_norm: Tensor,
    q_b: Arc<dyn QuantMethod>,
    kv: Arc<dyn QuantMethod>,
    kv_norm: Tensor,
    /// `attn_k_b` as `[heads, qk_head_dim, kv_lora_rank]`, so `q @ k_b_t` lands in latent space.
    k_b_t: Tensor,
    /// `attn_v_b` as `[heads, kv_lora_rank, v_head_dim]`.
    v_b_t: Tensor,
    output: Arc<dyn QuantMethod>,
    heads: usize,
    eps: f64,
    /// Longest sequence for which dense attention matches the indexer's selection.
    max_dense: usize,
}

enum Ffn {
    Dense {
        gate: Arc<dyn QuantMethod>,
        up: Arc<dyn QuantMethod>,
        down: Arc<dyn QuantMethod>,
        limit: f64,
    },
    Moe {
        experts: MoEExperts,
        router: Arc<dyn QuantMethod>,
        bias: Tensor,
        shared_gate: Arc<dyn QuantMethod>,
        shared_up: Arc<dyn QuantMethod>,
        shared_down: Arc<dyn QuantMethod>,
        top_k: usize,
        scale: f64,
        normalize: bool,
        limit: f64,
    },
}

struct RunState {
    kda: Vec<Option<Tensor>>,
    conv: Vec<Option<Tensor>>,
    /// Normalized KV latent of every token so far, `[batch, tokens, kv_lora_rank]`, per DSA layer.
    dsa: Vec<Option<Tensor>>,
}

impl ModelWeights {
    pub fn load(
        paths: &[std::path::PathBuf],
        device: &Device,
        _dtype: DType,
        comm: Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let archive = Arc::new(GgufArchive::open(paths)?);
        let arch = meta_string(&archive, "general.architecture")?;
        if !arch.eq_ignore_ascii_case(ARCH) {
            candle_core::bail!("Expected `{ARCH}` GGUF architecture, got `{arch}`.");
        }
        let blocks = meta_usize(&archive, &key("block_count"))?;
        let context = meta_usize(&archive, &key("context_length")).unwrap_or(4096);
        let hidden = meta_usize(&archive, &key("embedding_length"))?;
        let heads = meta_usize(&archive, &key("attention.head_count"))?;
        let kv_heads = meta_usize_array(&archive, &key("attention.head_count_kv"))?;
        if kv_heads.len() < blocks {
            candle_core::bail!("glm5-next head_count_kv is shorter than block_count");
        }
        let hc = meta_usize(&archive, &key("hyper_connection.count"))?;
        let sinkhorn_iters = meta_usize(&archive, &key("hyper_connection.sinkhorn_iterations"))?;
        let rms_eps = meta_f32(&archive, &key("attention.layer_norm_rms_epsilon"))?;
        let hc_eps = meta_f32(&archive, &key("hyper_connection.epsilon"))?;
        let head_dim = meta_usize(&archive, &key("kda.head_dim"))?;
        let kernel = meta_usize(&archive, &key("ssm.conv_kernel"))?;
        let lower = f64::from(meta_f32(&archive, &key("kda.gate_lower_bound"))?);
        let leading = meta_usize(&archive, &key("leading_dense_block_count")).unwrap_or(0);
        let experts_n = meta_usize(&archive, &key("expert_count")).unwrap_or(0);
        let expert_k = meta_usize(&archive, &key("expert_used_count")).unwrap_or(1);
        let intermediate =
            meta_usize(&archive, &key("expert_feed_forward_length")).unwrap_or(hidden);
        let dense_ffn = meta_usize(&archive, &key("feed_forward_length")).unwrap_or(intermediate);
        let route_scale = meta_f32(&archive, &key("expert_weights_scale")).unwrap_or(1.0);
        // The released GGUF stores this as a bool. Like llama.cpp, a file without it is refused.
        let route_normalize = meta_flag(&archive, &key("expert_weights_norm"))?;
        // 2 is sigmoid, the only gating function the router below computes.
        let gating = meta_usize(&archive, &key("expert_gating_func"))?;
        if gating != 2 {
            candle_core::bail!(
                "glm5-next expert gating function {gating} is not supported; only sigmoid (2) is"
            );
        }
        let index_top_k = meta_usize(&archive, &key("attention.indexer.top_k")).unwrap_or(context);
        let q_lora = meta_usize(&archive, &key("attention.q_lora_rank")).unwrap_or(0);
        let kv_lora = meta_usize(&archive, &key("attention.kv_lora_rank")).unwrap_or(0);
        let nope = meta_usize(&archive, &key("attention.key_length_mla")).unwrap_or(0);
        let rope = meta_usize(&archive, &key("rope.dimension_count")).unwrap_or(0);
        if rope != 0 {
            candle_core::bail!(
                "glm5-next DSA layers are NoPE; a rope dimension of {rope} is not supported"
            );
        }
        let clamps =
            meta_f32_array(&archive, &key("swiglu_clamp_exp")).unwrap_or_else(|_| vec![10.0]);

        let mut bindings = GgufBindingMap::new();
        for layer in leading..blocks {
            if archive.contains_tensor(&blk(layer, "ffn_gate_exps.weight")) {
                for (projection, source) in [
                    ("gate", "ffn_gate_exps"),
                    ("up", "ffn_up_exps"),
                    ("down", "ffn_down_exps"),
                ] {
                    bindings.insert(
                        format!("blk.{layer}.mlp.experts.{projection}_proj.weight"),
                        GgufTensorBinding::tensor(format!("blk.{layer}.{source}.weight")),
                    );
                }
            }
        }
        let source = Arc::new(GgufWeightSource::new(
            archive.clone(),
            &bindings,
            DType::F32,
        )?);
        let vb = source.sharded_var_builder(device.clone());

        let mut layers = Vec::with_capacity(blocks);
        for (layer, &kv_head_count) in kv_heads.iter().enumerate().take(blocks) {
            if !archive.contains_tensor(&blk(layer, "hc_attn_fn.weight")) {
                // The last block is the MTP head. It has no hyper-connection mix.
                break;
            }
            let limit = f64::from(clamps.get(layer).copied().unwrap_or(10.0));
            let ffn = if layer < leading
                || !archive.contains_tensor(&blk(layer, "ffn_gate_exps.weight"))
            {
                require(&archive, &blk(layer, "ffn_gate.weight"))?;
                Ffn::Dense {
                    gate: linear(&archive, &blk(layer, "ffn_gate.weight"), device)?,
                    up: linear(&archive, &blk(layer, "ffn_up.weight"), device)?,
                    down: linear(&archive, &blk(layer, "ffn_down.weight"), device)?,
                    limit,
                }
            } else {
                require(&archive, &blk(layer, "ffn_up_exps.weight"))?;
                require(&archive, &blk(layer, "ffn_down_exps.weight"))?;
                Ffn::Moe {
                    experts: MoEExperts::new(
                        &MoEExpertsConfig {
                            num_experts: experts_n,
                            num_experts_per_tok: expert_k,
                            hidden_size: hidden,
                            moe_intermediate_size: intermediate,
                            expert_proj_names: ExpertProjNames::DEFAULT,
                        },
                        vb.pp(format!("blk.{layer}.mlp")),
                        device.clone(),
                        &comm,
                        false,
                        &None,
                        Activation::Silu,
                    )?,
                    router: linear(&archive, &blk(layer, "ffn_gate_inp.weight"), device)?,
                    bias: dense(&archive, &blk(layer, "exp_probs_b.bias"), device)?,
                    shared_gate: linear(&archive, &blk(layer, "ffn_gate_shexp.weight"), device)?,
                    shared_up: linear(&archive, &blk(layer, "ffn_up_shexp.weight"), device)?,
                    shared_down: linear(&archive, &blk(layer, "ffn_down_shexp.weight"), device)?,
                    top_k: expert_k,
                    scale: f64::from(route_scale),
                    normalize: route_normalize,
                    limit,
                }
            };
            let kind = if kv_head_count == 0 {
                require(&archive, &blk(layer, "attn_q.weight"))?;
                LayerKind::Kda(Kda {
                    q: linear(&archive, &blk(layer, "attn_q.weight"), device)?,
                    k: linear(&archive, &blk(layer, "attn_k.weight"), device)?,
                    v: linear(&archive, &blk(layer, "attn_v.weight"), device)?,
                    conv: conv_weights(
                        &[
                            dense(&archive, &blk(layer, "ssm_conv1d_q.weight"), device)?,
                            dense(&archive, &blk(layer, "ssm_conv1d_k.weight"), device)?,
                            dense(&archive, &blk(layer, "ssm_conv1d_v.weight"), device)?,
                        ],
                        kernel,
                    )?,
                    f_a: linear(&archive, &blk(layer, "ssm_f_a.weight"), device)?,
                    f_b: linear(&archive, &blk(layer, "ssm_f_b.weight"), device)?,
                    dt_bias: dense(&archive, &blk(layer, "ssm_dt.bias"), device)?,
                    a: dense(&archive, &blk(layer, "ssm_a"), device)?,
                    beta: linear(&archive, &blk(layer, "ssm_beta.weight"), device)?,
                    g_a: linear(&archive, &blk(layer, "ssm_g_a.weight"), device)?,
                    g_b: linear(&archive, &blk(layer, "ssm_g_b.weight"), device)?,
                    norm: dense(&archive, &blk(layer, "ssm_norm.weight"), device)?,
                    output: linear(&archive, &blk(layer, "attn_output.weight"), device)?,
                    heads,
                    head_dim,
                    kernel,
                    lower_bound: lower,
                    eps: f64::from(rms_eps),
                })
            } else {
                require(&archive, &blk(layer, "attn_q_a.weight"))?;
                let _ = (q_lora, kv_lora, nope, dense_ffn);
                let k_b = dense(&archive, &blk(layer, "attn_k_b.weight"), device)?;
                let v_b = dense(&archive, &blk(layer, "attn_v_b.weight"), device)?;
                LayerKind::Dsa(Dsa {
                    q_a: linear(&archive, &blk(layer, "attn_q_a.weight"), device)?,
                    q_norm: dense(&archive, &blk(layer, "attn_q_a_norm.weight"), device)?,
                    q_b: linear(&archive, &blk(layer, "attn_q_b.weight"), device)?,
                    kv: linear(&archive, &blk(layer, "attn_kv_a_mqa.weight"), device)?,
                    kv_norm: dense(&archive, &blk(layer, "attn_kv_a_norm.weight"), device)?,
                    k_b_t: k_b.transpose(1, 2)?.contiguous()?,
                    v_b_t: v_b.transpose(1, 2)?.contiguous()?,
                    output: linear(&archive, &blk(layer, "attn_output.weight"), device)?,
                    heads,
                    eps: f64::from(rms_eps),
                    max_dense: index_top_k,
                })
            };
            layers.push(Layer {
                kind,
                attn_norm: dense(&archive, &blk(layer, "attn_norm.weight"), device)?,
                ffn_norm: dense(&archive, &blk(layer, "ffn_norm.weight"), device)?,
                hc_attn: hyper(&archive, layer, "attn", hc * hidden, device)?,
                hc_ffn: hyper(&archive, layer, "ffn", hc * hidden, device)?,
                ffn,
            });
        }
        if layers.is_empty() {
            candle_core::bail!("glm5-next GGUF has no hyper-connection blocks");
        }
        if hc < 2 {
            candle_core::bail!("glm5-next hyper-connection count {hc} is unusable");
        }
        let n = layers.len();
        Ok(Self {
            embed: linear(&archive, "token_embd.weight", device)?,
            norm: dense(&archive, "output_norm.weight", device)?,
            output: linear(&archive, "output.weight", device)?,
            layers,
            hc,
            sinkhorn_iters,
            rms_eps,
            hc_eps,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(n, context)),
            max_seq_len: context,
            state: Mutex::new(RunState {
                kda: (0..n).map(|_| None).collect(),
                conv: (0..n).map(|_| None).collect(),
                dsa: (0..n).map(|_| None).collect(),
            }),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _paged: Option<PagedAttentionInputMetadata>,
    ) -> Result<Tensor> {
        let (batch, seq) = input_ids.dims2()?;
        if batch != 1 {
            candle_core::bail!("glm5-next serves one sequence at a time");
        }
        let start = *seqlen_offsets.first().unwrap_or(&0);
        if start > 0 && seq != 1 {
            candle_core::bail!("glm5-next continuation must be a single token");
        }
        let ids = input_ids.reshape((batch, seq))?.to_dtype(DType::U32)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("glm5-next state lock was poisoned"))?;
        if start == 0 {
            for slot in &mut state.kda {
                *slot = None;
            }
            for slot in &mut state.conv {
                *slot = None;
            }
            for slot in &mut state.dsa {
                *slot = None;
            }
        }
        let token = self.embed.embedding_forward(&ids, DType::F32)?;
        let token = token.unsqueeze(2)?;
        let mut parts = Vec::with_capacity(self.hc);
        for _ in 0..self.hc {
            parts.push(token.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let mut hidden = Tensor::cat(&refs, 2)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let attn_mix = weights(&layer.hc_attn);
            let ffn_mix = weights(&layer.hc_ffn);
            let (next, _pre) = {
                // unlagged_block returns only the streams. The pre is internal.
                let mut out = None;
                let mut attn = |x: &Tensor| {
                    let y = layer.attend(x, index, &mut state)?;
                    out = Some(y.clone());
                    Ok(y)
                };
                let mut ffn = |x: &Tensor| layer.feed(x);
                let mixed = unlagged_block(
                    &hidden,
                    &attn_mix,
                    &ffn_mix,
                    &layer.attn_norm,
                    &layer.ffn_norm,
                    self.hc,
                    self.sinkhorn_iters,
                    self.rms_eps,
                    self.hc_eps,
                    &mut attn,
                    &mut ffn,
                )?;
                let _ = out;
                (mixed, ())
            };
            hidden = next;
        }
        drop(state);
        let flat = hidden.mean_keepdim(2)?.squeeze(2)?;
        let normed = rmsnorm(&flat, &self.norm, f64::from(self.rms_eps))?;
        let logits = self.output.forward(&normed)?;
        extract_logits(&logits, context_lens)
    }
}

fn weights(src: &HyperWeights) -> HyperWeights {
    HyperWeights {
        fn_weight: src.fn_weight.clone(),
        scale: src.scale.clone(),
        base: src.base.clone(),
    }
}

impl Layer {
    fn attend(&self, x: &Tensor, index: usize, state: &mut RunState) -> Result<Tensor> {
        match &self.kind {
            LayerKind::Kda(kda) => kda.forward(x, &mut state.kda[index], &mut state.conv[index]),
            LayerKind::Dsa(dsa) => dsa.forward(x, &mut state.dsa[index]),
        }
    }

    fn feed(&self, x: &Tensor) -> Result<Tensor> {
        match &self.ffn {
            Ffn::Dense {
                gate,
                up,
                down,
                limit,
            } => {
                let mixed = swiglu_clamped(&gate.forward(x)?, &up.forward(x)?, *limit)?;
                down.forward(&mixed)
            }
            Ffn::Moe {
                experts,
                router,
                bias,
                shared_gate,
                shared_up,
                shared_down,
                top_k,
                scale,
                normalize,
                limit,
            } => {
                let (batch, seq, hidden) = x.dims3()?;
                let logits = router.forward(&x.reshape((batch * seq, hidden))?)?;
                let scores = candle_nn::ops::sigmoid(&logits)?;
                let choice = scores.broadcast_add(bias)?;
                let picked = crate::ops::TopKLastDimOp::topk(&choice, *top_k)?;
                let weights = scores.gather(&picked.indices, D::Minus1)?;
                let weights = if *normalize {
                    weights.broadcast_div(&(weights.sum_keepdim(D::Minus1)? + 1e-20)?)?
                } else {
                    weights
                };
                let weights = (weights * *scale)?;
                let routed = experts.forward(x, weights, &picked.indices)?;
                let shared =
                    swiglu_clamped(&shared_gate.forward(x)?, &shared_up.forward(x)?, *limit)?;
                routed + shared_down.forward(&shared)?
            }
        }
    }
}

impl Kda {
    fn forward(
        &self,
        x: &Tensor,
        recurrent: &mut Option<Tensor>,
        conv_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;
        let (q, k, v) = self.convolve(&q, &k, &v, conv_state)?;
        let q = q.reshape((batch, seq, self.heads, self.head_dim))?;
        let k = k.reshape((batch, seq, self.heads, self.head_dim))?;
        let v = v.reshape((batch, seq, self.heads, self.head_dim))?;
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;
        let g = self.f_b.forward(&self.f_a.forward(x)?)?;
        let g = g.broadcast_add(&self.dt_bias)?;
        let g = g.reshape((batch, seq, self.heads, self.head_dim))?;
        let a = self.a.reshape((1, 1, self.heads, 1))?;
        let g = g.broadcast_mul(&a)?;
        let g = (candle_nn::ops::sigmoid(&(g * -1.0)?)? * self.lower_bound)?;
        let beta = candle_nn::ops::sigmoid(&self.beta.forward(x)?)?;
        let beta = beta.reshape((batch, seq, self.heads))?;
        let mut state = match recurrent {
            Some(state) => state.clone(),
            None => Tensor::zeros(
                (batch, self.heads, self.head_dim, self.head_dim),
                DType::F32,
                x.device(),
            )?,
        };
        let y = kda_rule_recurrence(&q, &k, &v, &g, &beta, &mut state)?;
        *recurrent = Some(state);
        // The output norm weight is `[head_dim]`: each head is normalized on its own.
        let y = y.reshape((batch, seq, self.heads, self.head_dim))?;
        let y = rmsnorm(&y, &self.norm, self.eps)?;
        let gate = candle_nn::ops::sigmoid(&self.g_b.forward(&self.g_a.forward(x)?)?)?;
        let gate = gate.reshape((batch, seq, self.heads, self.head_dim))?;
        let y = (y * gate)?.reshape((batch, seq, self.heads * self.head_dim))?;
        self.output.forward(&y)
    }

    fn convolve(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        state: &mut Option<Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let joined = Tensor::cat(&[q, k, v], D::Minus1)?;
        let (_, seq, channels) = joined.dims3()?;
        let width = channels / 3;
        let history = match state {
            Some(prev) => Tensor::cat(&[prev, &joined], 1)?,
            None => joined.clone(),
        };
        let keep = self.kernel.saturating_sub(1).min(history.dim(1)?);
        if keep > 0 {
            let start = history.dim(1)? - keep;
            *state = Some(history.narrow(1, start, keep)?);
        }
        let mixed = causal_depthwise_conv(&history, &self.conv, seq)?;
        let mixed = candle_nn::ops::silu(&mixed)?;
        let q = mixed.narrow(D::Minus1, 0, width)?;
        let k = mixed.narrow(D::Minus1, width, width)?;
        let v = mixed.narrow(D::Minus1, width * 2, width)?;
        Ok((q, k, v))
    }
}

/// Joins the q, k and v depthwise convolutions into one `[channels, kernel]` f32 matrix, in
/// the order the joined channels run, once at load.
fn conv_weights(parts: &[Tensor], kernel: usize) -> Result<Tensor> {
    let joined = Tensor::cat(parts, 0)?.to_dtype(DType::F32)?.flatten_all()?;
    let values = joined.elem_count();
    if kernel == 0 || !values.is_multiple_of(kernel) {
        candle_core::bail!(
            "glm5-next convolution weights hold {values} values, which is not a whole number of {kernel}-tap kernels"
        );
    }
    joined.reshape((values / kernel, kernel))
}

/// Causal depthwise convolution of the last `seq` positions of `history`
/// (`[batch, hist_len, channels]`) with `weights` (`[channels, kernel]`). Output `t` sums
/// `history[abs + 1 + lag - kernel] * weights[c, lag]` over the lags that land inside the
/// history, where `abs = hist_len - seq + t`. It runs as `kernel` tensor ops on the history's
/// device: reading the history and weights back to the host would stall the GPU every layer.
fn causal_depthwise_conv(history: &Tensor, weights: &Tensor, seq: usize) -> Result<Tensor> {
    let (batch, hist_len, channels) = history.dims3()?;
    let kernel = weights.dim(1)?;
    let history = history.to_dtype(DType::F32)?;
    let padded = if kernel > 1 {
        let zeros = Tensor::zeros((batch, kernel - 1, channels), DType::F32, history.device())?;
        Tensor::cat(&[&zeros, &history], 1)?
    } else {
        history
    };
    let mut acc: Option<Tensor> = None;
    for lag in 0..kernel {
        let rows = padded.narrow(1, hist_len - seq + lag, seq)?;
        let term = rows.broadcast_mul(&weights.narrow(1, lag, 1)?.squeeze(1)?)?;
        acc = Some(match acc {
            Some(sum) => (sum + term)?,
            None => term,
        });
    }
    acc.ok_or_else(|| candle_core::Error::msg("glm5-next convolution has no taps"))
}

impl Dsa {
    /// Absorbed MLA over the cached latent: scores are `(q @ k_b) . latent` and the
    /// output is `(probs @ latent) @ v_b`. The latent is the only thing cached.
    fn forward(&self, x: &Tensor, cache: &mut Option<Tensor>) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let past = match cache.as_ref() {
            Some(prev) => prev.dim(1)?,
            None => 0,
        };
        let total = past + seq;
        if total > self.max_dense {
            candle_core::bail!(
                "glm5-next attends densely only up to {} tokens and this sequence has {total}; the sparse attention indexer that longer sequences need is not implemented",
                self.max_dense
            );
        }
        let q = rmsnorm(&self.q_a.forward(x)?, &self.q_norm, self.eps)?;
        let q = self
            .q_b
            .forward(&q)?
            .reshape((batch, seq, self.heads, ()))?;
        let qk_dim = q.dim(D::Minus1)?;
        let latent = rmsnorm(&self.kv.forward(x)?, &self.kv_norm, self.eps)?;
        let latent = match cache.take() {
            Some(prev) => Tensor::cat(&[&prev, &latent], 1)?,
            None => latent,
        };
        *cache = Some(latent.clone());

        let q = q.transpose(1, 2)?.contiguous()?;
        let q_latent = q.broadcast_matmul(&self.k_b_t.unsqueeze(0)?)?;
        // Every head scores against the same latent, so the heads fold into the query rows
        // of one matmul; broadcasting the latent across them copies it once per head.
        let lora = q_latent.dim(D::Minus1)?;
        let keys = latent.transpose(1, 2)?.contiguous()?;
        let scores = q_latent
            .reshape((batch, self.heads * seq, lora))?
            .matmul(&keys)?
            .reshape((batch, self.heads, seq, total))?;
        let scores = (scores * (1.0 / (qk_dim as f64).sqrt()))?;
        let scores = if seq > 1 {
            let key_pos = Tensor::arange(0u32, total as u32, x.device())?.unsqueeze(0)?;
            let query_pos = Tensor::arange(past as u32, total as u32, x.device())?.unsqueeze(1)?;
            let visible = key_pos.broadcast_le(&query_pos)?;
            let blocked = (scores.zeros_like()? - f64::INFINITY)?;
            visible
                .broadcast_as(scores.shape())?
                .where_cond(&scores, &blocked)?
        } else {
            scores
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let mixed = probs
            .reshape((batch, self.heads * seq, total))?
            .matmul(&latent.contiguous()?)?
            .reshape((batch, self.heads, seq, lora))?;
        let y = mixed.broadcast_matmul(&self.v_b_t.unsqueeze(0)?)?;
        let y = y.transpose(1, 2)?.reshape((batch, seq, ()))?;
        self.output.forward(&y)
    }
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let denom = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&denom)
}

fn hyper(
    archive: &GgufArchive,
    layer: usize,
    which: &str,
    flat_in: usize,
    device: &Device,
) -> Result<HyperWeights> {
    Ok(HyperWeights {
        fn_weight: hyper_mix_projection(
            dense(
                archive,
                &blk(layer, &format!("hc_{which}_fn.weight")),
                device,
            )?,
            flat_in,
        )?,
        scale: dense(
            archive,
            &blk(layer, &format!("hc_{which}_scale.weight")),
            device,
        )?,
        base: dense(
            archive,
            &blk(layer, &format!("hc_{which}_base.weight")),
            device,
        )?,
    })
}

fn linear(archive: &GgufArchive, name: &str, device: &Device) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(archive.load_qtensor(name, device)?),
        b: None,
    })?))
}

fn dense(archive: &GgufArchive, name: &str, device: &Device) -> Result<Tensor> {
    archive.load_qtensor(name, device)?.dequantize(device)
}

fn require(archive: &GgufArchive, name: &str) -> Result<()> {
    if archive.contains_tensor(name) {
        Ok(())
    } else {
        candle_core::bail!("GGUF glm5-next is missing `{name}`")
    }
}

fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

fn blk(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

fn meta_value<'a>(
    archive: &'a GgufArchive,
    name: &str,
) -> Result<&'a candle_core::quantized::gguf_file::Value> {
    archive.metadata_value(name).ok_or_else(|| {
        candle_core::Error::msg(format!("GGUF glm5-next is missing metadata `{name}`"))
    })
}

fn meta_string(archive: &GgufArchive, name: &str) -> Result<String> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::String(value) => Ok(value.clone()),
        _ => candle_core::bail!("GGUF metadata `{name}` must be a string"),
    }
}

fn value_usize(value: &candle_core::quantized::gguf_file::Value, name: &str) -> Result<usize> {
    use candle_core::quantized::gguf_file::Value;
    let value = match value {
        Value::U8(value) => u64::from(*value),
        Value::U16(value) => u64::from(*value),
        Value::U32(value) => u64::from(*value),
        Value::U64(value) => *value,
        Value::I32(value) if *value >= 0 => *value as u64,
        Value::I64(value) if *value >= 0 => *value as u64,
        _ => candle_core::bail!("GGUF metadata `{name}` must be a nonnegative integer"),
    };
    usize::try_from(value).map_err(candle_core::Error::wrap)
}

fn meta_usize(archive: &GgufArchive, name: &str) -> Result<usize> {
    value_usize(meta_value(archive, name)?, name)
}

fn meta_flag(archive: &GgufArchive, name: &str) -> Result<bool> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::Bool(value) => Ok(*value),
        other => Ok(value_usize(other, name)? != 0),
    }
}

fn meta_f32(archive: &GgufArchive, name: &str) -> Result<f32> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::F32(value) => Ok(*value),
        candle_core::quantized::gguf_file::Value::F64(value) => Ok(*value as f32),
        other => Ok(value_usize(other, name)? as f32),
    }
}

fn meta_usize_array(archive: &GgufArchive, name: &str) -> Result<Vec<usize>> {
    let candle_core::quantized::gguf_file::Value::Array(values) = meta_value(archive, name)? else {
        candle_core::bail!("GGUF metadata `{name}` must be an array");
    };
    values
        .iter()
        .map(|value| value_usize(value, name))
        .collect()
}

fn meta_f32_array(archive: &GgufArchive, name: &str) -> Result<Vec<f32>> {
    let candle_core::quantized::gguf_file::Value::Array(values) = meta_value(archive, name)? else {
        candle_core::bail!("GGUF metadata `{name}` must be an array");
    };
    values
        .iter()
        .map(|value| match value {
            candle_core::quantized::gguf_file::Value::F32(inner) => Ok(*inner),
            candle_core::quantized::gguf_file::Value::F64(inner) => Ok(*inner as f32),
            other => Ok(value_usize(other, name)? as f32),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device, Tensor};
    use mistralrs_quant::{Comm, Id};
    use tempfile::NamedTempFile;

    use super::ModelWeights;

    fn pattern(shape: &[u64], salt: f32) -> Vec<f32> {
        let count: u64 = shape.iter().product();
        (0..count)
            .map(|i| salt + ((i % 11) as f32 - 5.0) * 0.04)
            .collect()
    }

    /// Seeded values in `[-scale, scale]`, so each token and channel differs.
    fn noise(shape: &[u64], seed: u64, scale: f32) -> Vec<f32> {
        let count: u64 = shape.iter().product();
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn noisy(name: &str, shape: &[u64], seed: u64, scale: f32) -> Weight {
        Weight {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: f32_bytes(&noise(shape, seed, scale)),
            dtype: 0,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[derive(Clone)]
    struct Weight {
        name: String,
        shape: Vec<u64>,
        data: Vec<u8>,
        dtype: u32,
    }

    fn f32w(name: &str, shape: &[u64], salt: f32) -> Weight {
        Weight {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: f32_bytes(&pattern(shape, salt)),
            dtype: 0,
        }
    }

    fn push_meta(meta: &mut Vec<(String, Vec<u8>)>, key: &str, bytes: Vec<u8>) {
        meta.push((key.to_string(), bytes));
    }

    fn write(tensors: &[Weight], expert_dtype: u32, expert: Vec<u8>) -> NamedTempFile {
        let _ = expert_dtype;
        write_layers(tensors, &[0], Some(expert))
    }

    /// One layer per entry of `kv_heads`: 0 is KDA, 1 is DSA. With `expert`, layer 0
    /// has routed experts; every other layer uses the dense FFN.
    fn write_layers(
        tensors: &[Weight],
        kv_heads: &[u32],
        expert: Option<Vec<u8>>,
    ) -> NamedTempFile {
        write_model(tensors, kv_heads, expert, true)
    }

    fn write_model(
        tensors: &[Weight],
        kv_heads: &[u32],
        expert: Option<Vec<u8>>,
        normalize_experts: bool,
    ) -> NamedTempFile {
        let mut meta: Vec<(String, Vec<u8>)> = Vec::new();
        let push_str = |meta: &mut Vec<(String, Vec<u8>)>, key: &str, value: &str| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&8u32.to_le_bytes());
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
            push_meta(meta, key, bytes);
        };
        let push_u32 = |meta: &mut Vec<(String, Vec<u8>)>, key: &str, value: u32| {
            let mut bytes = 4u32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&value.to_le_bytes());
            push_meta(meta, key, bytes);
        };
        let push_f32 = |meta: &mut Vec<(String, Vec<u8>)>, key: &str, value: f32| {
            let mut bytes = 6u32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&value.to_le_bytes());
            push_meta(meta, key, bytes);
        };
        let push_bool = |meta: &mut Vec<(String, Vec<u8>)>, key: &str, value: bool| {
            let mut bytes = 7u32.to_le_bytes().to_vec();
            bytes.push(u8::from(value));
            push_meta(meta, key, bytes);
        };
        let push_u32s = |meta: &mut Vec<(String, Vec<u8>)>, key: &str, values: &[u32]| {
            let mut bytes = 9u32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            push_meta(meta, key, bytes);
        };
        push_str(&mut meta, "general.architecture", "glm5-next");
        push_u32(&mut meta, "general.alignment", 32);
        push_u32(&mut meta, "glm5-next.block_count", kv_heads.len() as u32);
        push_u32(&mut meta, "glm5-next.context_length", 32);
        push_u32(&mut meta, "glm5-next.embedding_length", 32);
        push_u32(&mut meta, "glm5-next.feed_forward_length", 32);
        push_u32(&mut meta, "glm5-next.attention.head_count", 4);
        push_f32(
            &mut meta,
            "glm5-next.attention.layer_norm_rms_epsilon",
            1e-5,
        );
        push_u32(&mut meta, "glm5-next.expert_count", 2);
        push_u32(&mut meta, "glm5-next.expert_used_count", 1);
        push_u32(&mut meta, "glm5-next.expert_feed_forward_length", 32);
        push_u32s(&mut meta, "glm5-next.attention.head_count_kv", kv_heads);
        push_u32(&mut meta, "glm5-next.attention.q_lora_rank", 16);
        push_u32(&mut meta, "glm5-next.attention.kv_lora_rank", 16);
        push_u32(&mut meta, "glm5-next.attention.key_length_mla", 8);
        push_u32(&mut meta, "glm5-next.rope.dimension_count", 0);
        push_u32(&mut meta, "glm5-next.ssm.conv_kernel", 4);
        push_u32(&mut meta, "glm5-next.kda.head_dim", 8);
        push_f32(&mut meta, "glm5-next.kda.gate_lower_bound", -5.0);
        push_u32(&mut meta, "glm5-next.hyper_connection.count", 2);
        push_u32(
            &mut meta,
            "glm5-next.hyper_connection.sinkhorn_iterations",
            1,
        );
        push_f32(&mut meta, "glm5-next.hyper_connection.epsilon", 1e-6);
        push_u32(&mut meta, "glm5-next.leading_dense_block_count", 0);
        push_f32(&mut meta, "glm5-next.expert_weights_scale", 2.5);
        // The released GGUF stores this flag as a GGUF bool, not an integer.
        push_bool(
            &mut meta,
            "glm5-next.expert_weights_norm",
            normalize_experts,
        );
        push_u32(&mut meta, "glm5-next.expert_gating_func", 2);
        push_u32(&mut meta, "glm5-next.attention.indexer.top_k", 8);
        push_u32s(
            &mut meta,
            "glm5-next.swiglu_clamp_exp",
            &vec![10; kv_heads.len()],
        );
        let mut all = tensors.to_vec();
        if let Some(expert) = expert {
            for (name, data) in [
                ("blk.0.ffn_gate_exps.weight", expert),
                ("blk.0.ffn_up_exps.weight", q8(0.4)),
                ("blk.0.ffn_down_exps.weight", q8(0.7)),
            ] {
                all.push(Weight {
                    name: name.to_string(),
                    shape: vec![2, 32, 32],
                    data,
                    dtype: 8,
                });
            }
        }
        // The rest of the writer mirrors the deepseek test file.
        let alignment = 32usize;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(all.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        for (key, value) in &meta {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(value);
        }
        let mut cursor = 0usize;
        let mut offsets = Vec::new();
        for tensor in &all {
            offsets.push(cursor);
            cursor += tensor.data.len();
            cursor = cursor.div_ceil(alignment) * alignment;
        }
        for (tensor, offset) in all.iter().zip(offsets.iter()) {
            bytes.extend_from_slice(&(tensor.name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(tensor.name.as_bytes());
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in tensor.shape.iter().rev() {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.dtype.to_le_bytes());
            bytes.extend_from_slice(&(*offset as u64).to_le_bytes());
        }
        let data_at = bytes.len().div_ceil(alignment) * alignment;
        bytes.resize(data_at, 0);
        let mut at = 0usize;
        for tensor in &all {
            bytes.resize(data_at + at, 0);
            bytes.extend_from_slice(&tensor.data);
            at += tensor.data.len();
            at = at.div_ceil(alignment) * alignment;
            bytes.resize(data_at + at, 0);
        }
        let mut file = NamedTempFile::new().expect("gguf");
        std::io::Write::write_all(&mut file, &bytes).expect("write");
        file
    }

    fn kda_weights(q_salt: f32) -> Vec<Weight> {
        vec![
            f32w("token_embd.weight", &[8, 32], 0.1),
            f32w("output.weight", &[8, 32], 0.2),
            f32w("output_norm.weight", &[32], 1.0),
            f32w("blk.0.attn_norm.weight", &[32], 1.0),
            f32w("blk.0.ffn_norm.weight", &[32], 1.0),
            f32w("blk.0.attn_q.weight", &[32, 32], q_salt),
            f32w("blk.0.attn_k.weight", &[32, 32], 0.3),
            f32w("blk.0.attn_v.weight", &[32, 32], 0.4),
            f32w("blk.0.attn_output.weight", &[32, 32], 0.5),
            f32w("blk.0.ssm_conv1d_q.weight", &[32, 4], 0.2),
            f32w("blk.0.ssm_conv1d_k.weight", &[32, 4], 0.2),
            f32w("blk.0.ssm_conv1d_v.weight", &[32, 4], 0.2),
            f32w("blk.0.ssm_f_a.weight", &[8, 32], 0.05),
            f32w("blk.0.ssm_f_b.weight", &[32, 8], 0.05),
            f32w("blk.0.ssm_dt.bias", &[32], 0.0),
            f32w("blk.0.ssm_a", &[4], 0.1),
            f32w("blk.0.ssm_beta.weight", &[4, 32], 0.1),
            f32w("blk.0.ssm_g_a.weight", &[8, 32], 0.05),
            f32w("blk.0.ssm_g_b.weight", &[32, 8], 0.05),
            // Per-head output norm, as in the released GGUF: `[kda.head_dim]`.
            f32w("blk.0.ssm_norm.weight", &[8], 1.0),
            f32w("blk.0.hc_attn_fn.weight", &[64, 8], 0.01),
            f32w("blk.0.hc_attn_scale.weight", &[3], 1.0),
            f32w("blk.0.hc_attn_base.weight", &[8], 0.0),
            f32w("blk.0.hc_ffn_fn.weight", &[64, 8], 0.01),
            f32w("blk.0.hc_ffn_scale.weight", &[3], 1.0),
            f32w("blk.0.hc_ffn_base.weight", &[8], 0.0),
            f32w("blk.0.ffn_gate_inp.weight", &[2, 32], 0.2),
            f32w("blk.0.exp_probs_b.bias", &[2], 0.0),
            f32w("blk.0.ffn_gate_shexp.weight", &[32, 32], 0.1),
            f32w("blk.0.ffn_up_shexp.weight", &[32, 32], 0.1),
            f32w("blk.0.ffn_down_shexp.weight", &[32, 32], 0.1),
        ]
    }

    /// A DSA layer with a dense FFN. Shapes follow the released GGUF: `attn_k_b`
    /// is `[heads, kv_lora, qk]` and `attn_v_b` is `[heads, v, kv_lora]`.
    fn dsa_layer_weights(layer: usize) -> Vec<Weight> {
        let w = |suffix: &str, shape: &[u64], salt: f32| {
            f32w(&format!("blk.{layer}.{suffix}"), shape, salt)
        };
        let n = |suffix: &str, shape: &[u64], seed: u64| {
            noisy(&format!("blk.{layer}.{suffix}"), shape, seed, 0.5)
        };
        vec![
            w("attn_norm.weight", &[32], 1.0),
            w("ffn_norm.weight", &[32], 1.0),
            n("attn_q_a.weight", &[16, 32], 11),
            w("attn_q_a_norm.weight", &[16], 1.0),
            n("attn_q_b.weight", &[32, 16], 12),
            n("attn_kv_a_mqa.weight", &[16, 32], 13),
            w("attn_kv_a_norm.weight", &[16], 1.0),
            n("attn_k_b.weight", &[4, 16, 8], 14),
            n("attn_v_b.weight", &[4, 8, 16], 15),
            n("attn_output.weight", &[32, 32], 16),
            w("hc_attn_fn.weight", &[64, 8], 0.02),
            w("hc_attn_scale.weight", &[3], 1.0),
            w("hc_attn_base.weight", &[8], 0.0),
            w("hc_ffn_fn.weight", &[64, 8], 0.02),
            w("hc_ffn_scale.weight", &[3], 1.0),
            w("hc_ffn_base.weight", &[8], 0.0),
            w("ffn_gate.weight", &[32, 32], 0.1),
            w("ffn_up.weight", &[32, 32], 0.2),
            w("ffn_down.weight", &[32, 32], 0.3),
        ]
    }

    fn embedding_weights() -> Vec<Weight> {
        vec![
            noisy("token_embd.weight", &[8, 32], 1, 1.0),
            noisy("output.weight", &[8, 32], 2, 0.5),
            f32w("output_norm.weight", &[32], 1.0),
        ]
    }

    fn q8(salt: f32) -> Vec<u8> {
        let values = pattern(&[2, 32, 32], salt);
        let tensor = Tensor::from_vec(values, (2, 32, 32), &Device::Cpu).expect("pattern");
        candle_core::quantized::QTensor::quantize(&tensor, GgmlDType::Q8_0)
            .expect("q8")
            .data()
            .expect("bytes")
            .to_vec()
    }

    fn load(file: &NamedTempFile) -> ModelWeights {
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).expect("comm"));
        ModelWeights::load(
            &[PathBuf::from(file.path())],
            &Device::Cpu,
            DType::F32,
            comm,
        )
        .expect("load glm5-next")
    }

    fn logits(model: &ModelWeights, tokens: &[u32]) -> Vec<f32> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).expect("ids");
        model
            .forward(&ids, &[0], vec![(tokens.len() - 1, 1)], None)
            .expect("forward")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("f32")
    }

    fn step(model: &ModelWeights, tokens: &[u32], start: usize) -> Vec<f32> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).expect("ids");
        model
            .forward(&ids, &[start], vec![(tokens.len() - 1, 1)], None)
            .expect("forward")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("f32")
    }

    /// Decoding a token after a prefill must give the logits of recomputing the
    /// whole sequence: the KDA state and the DSA latent cache carry everything the
    /// earlier tokens contribute.
    /// The host loop `causal_depthwise_conv` replaced, as the reference it is checked against.
    fn conv_on_the_host(
        history: &[f32],
        batch: usize,
        hist_len: usize,
        channels: usize,
        weights: &[f32],
        kernel: usize,
        seq: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; batch * seq * channels];
        for b in 0..batch {
            for t in 0..seq {
                let abs = hist_len - seq + t;
                for c in 0..channels {
                    let mut acc = 0f32;
                    for lag in 0..kernel {
                        if abs + 1 + lag < kernel {
                            continue;
                        }
                        let src = abs + 1 + lag - kernel;
                        acc += history[(b * hist_len + src) * channels + c]
                            * weights[c * kernel + lag];
                    }
                    out[(b * seq + t) * channels + c] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn the_device_convolution_matches_the_host_loop() {
        let device = Device::Cpu;
        let kernel = 4;
        for (batch, hist_len, seq, channels) in [
            (1, 1, 1, 6),
            (1, 2, 1, 6),
            (1, 6, 1, 9),
            (2, 5, 3, 6),
            (1, 3, 3, 3),
        ] {
            let history: Vec<f32> = (0..batch * hist_len * channels)
                .map(|i| ((i * 29 % 23) as f32 - 11.0) * 0.05)
                .collect();
            let weights: Vec<f32> = (0..channels * kernel)
                .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.03)
                .collect();
            let got = super::causal_depthwise_conv(
                &Tensor::from_vec(history.clone(), (batch, hist_len, channels), &device).unwrap(),
                &Tensor::from_vec(weights.clone(), (channels, kernel), &device).unwrap(),
                seq,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let expect =
                conv_on_the_host(&history, batch, hist_len, channels, &weights, kernel, seq);
            assert_eq!(got.len(), expect.len());
            for (index, (g, e)) in got.iter().zip(&expect).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6,
                    "batch={batch} hist_len={hist_len} seq={seq} index={index}: device {g}, host {e}"
                );
            }
        }
    }

    #[test]
    fn decode_after_prefill_matches_a_full_recompute() {
        let mut weights = kda_weights(0.2);
        weights.extend(dsa_layer_weights(1));
        let file = write_layers(&weights, &[0, 1], Some(q8(0.1)));
        let model = load(&file);

        let full = step(&model, &[1, 2, 3, 4], 0);
        let _ = step(&model, &[1, 2, 3], 0);
        let decoded = step(&model, &[4], 3);
        assert_eq!(full.len(), decoded.len());
        let worst = full
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-4,
            "decode diverged from recompute by {worst}: {full:?} vs {decoded:?}"
        );
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// With DSA as the only layer, attention is the only path between tokens, so a
    /// decode step that loses the cached latent answers as if the prefix were absent.
    #[test]
    fn dsa_decode_attends_to_the_cached_prefix() {
        let mut weights = embedding_weights();
        weights.extend(dsa_layer_weights(0));
        let file = write_layers(&weights, &[1], None);
        let model = load(&file);

        let full = step(&model, &[1, 2, 3, 4], 0);
        let _ = step(&model, &[1, 2, 3], 0);
        let decoded = step(&model, &[4], 3);
        let alone = step(&model, &[4], 0);

        let drift = max_abs_diff(&full, &decoded);
        assert!(
            drift < 1e-5,
            "decode diverged from recompute by {drift}: {full:?} vs {decoded:?}"
        );
        let prefix_effect = max_abs_diff(&decoded, &alone);
        assert!(
            prefix_effect > 1e-3,
            "the prefix moved the decoded logits by only {prefix_effect}; the test cannot see a lost cache"
        );
    }

    #[test]
    fn dsa_refuses_sequences_past_the_indexer_top_k() {
        let mut weights = embedding_weights();
        weights.extend(dsa_layer_weights(0));
        let file = write_layers(&weights, &[1], None);
        let model = load(&file);
        let ids = Tensor::from_vec((0u32..9).map(|t| t % 8).collect(), (1, 9), &Device::Cpu)
            .expect("ids");
        let err = model
            .forward(&ids, &[0], vec![(8, 1)], None)
            .expect_err("9 tokens exceed the indexer top_k of 8");
        assert!(
            err.to_string()
                .contains("attends densely only up to 8 tokens and this sequence has 9"),
            "{err}"
        );
    }

    /// Routed weights are renormalized only when `expert_weights_norm` says so, and the
    /// flag is a GGUF bool. Reading it as an integer failed and silently skipped the
    /// renormalization, which scaled GLM-5.3-Flash's routed experts by up to 6x.
    #[test]
    fn expert_weights_norm_is_read_from_a_gguf_bool() {
        let weights = kda_weights(0.2);
        let normalized = write_model(&weights, &[0], Some(q8(0.1)), true);
        let raw = write_model(&weights, &[0], Some(q8(0.1)), false);
        let with_norm = logits(&load(&normalized), &[1, 2]);
        let without_norm = logits(&load(&raw), &[1, 2]);
        assert_ne!(
            with_norm, without_norm,
            "`expert_weights_norm` must change how routed experts are weighted"
        );
    }

    #[test]
    fn kda_and_quantized_experts_change_the_logits() {
        let file = write(&kda_weights(0.2), 8, q8(0.1));
        let model = load(&file);
        let first = logits(&model, &[1, 2]);
        let second = logits(&model, &[3, 4]);
        assert_ne!(
            first, second,
            "different tokens must change glm5-next logits"
        );
        let other = write(&kda_weights(1.4), 8, q8(0.1));
        let changed = logits(&load(&other), &[1, 2]);
        assert_ne!(first, changed, "KDA query weights must change logits");
        let experts = write(&kda_weights(0.2), 8, q8(1.3));
        let changed = logits(&load(&experts), &[1, 2]);
        assert_ne!(first, changed, "quantized experts must change logits");

        let mut dropped = kda_weights(0.2);
        dropped.retain(|weight| weight.name != "blk.0.attn_q.weight");
        let broken = write(&dropped, 8, q8(0.1));
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).expect("comm"));
        let err = match ModelWeights::load(
            &[PathBuf::from(broken.path())],
            &Device::Cpu,
            DType::F32,
            comm,
        ) {
            Ok(_) => panic!("missing KDA query must fail closed"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("attn_q.weight"), "{}", err);
    }
}

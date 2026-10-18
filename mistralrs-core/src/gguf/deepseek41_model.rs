#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF loader and forward for `general.architecture = deepseek41`.
//!
//! Linears stay in [`GgufMatMul`]. Routed experts stay in [`MoEExperts`] under
//! the quantized stacked tensors (`ffn_*_exps`), including Q2_K. The Engram
//! table is gathered by row. This graph is not DeepSeek-V3.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    GgufArchive, GgufBindingMap, GgufMatMul, GgufTensorBinding, GgufWeightSource, QuantMethod,
    QuantMethodConfig, QuantizedWeightSource, Shard,
};

use crate::gguf::deepseek41::{
    compress_groups, engram_apply, engram_row_ids, lagged_block, moe_route, rmsnorm,
    swiglu_clamped, HyperWeights,
};
use crate::kv_cache::{EitherCache, NormalCache};
use crate::layers::{
    Activation, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling, DeepSeekV2RotaryEmbedding,
    ScaledRopeType,
};
use crate::moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig};
use crate::ops::TopKLastDimOp;
use crate::pipeline::extract_logits;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

const ARCH: &str = "deepseek41";
/// Rope tables cover a chat prompt. A longer position is an error, not a wrap.
const ROPE_TABLE: usize = 8192;

pub struct ModelWeights {
    embed: Arc<dyn QuantMethod>,
    layers: Vec<Layer>,
    norm: Tensor,
    output: Arc<dyn QuantMethod>,
    hc: usize,
    sinkhorn_iters: usize,
    rms_eps: f32,
    hc_eps: f32,
    engram: Option<EngramHash>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    state: Mutex<RunState>,
}

struct EngramHash {
    token_map: Vec<u32>,
    pad_token: u32,
    n_layers: usize,
    n_heads: usize,
    max_ngram: usize,
    key_length: usize,
    multipliers: Vec<u64>,
    primes: Vec<u64>,
    offsets: Vec<u64>,
}

struct Layer {
    index: usize,
    ratio: usize,
    window: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    groups: usize,
    index_heads: usize,
    index_dim: usize,
    index_topk: usize,
    expert_k: usize,
    route_scale: f64,
    swiglu_limit: f64,
    kv_owner: Option<usize>,
    q_a: Arc<dyn QuantMethod>,
    q_a_norm: Tensor,
    q_b: Arc<dyn QuantMethod>,
    kv: Arc<dyn QuantMethod>,
    kv_norm: Tensor,
    sinks: Tensor,
    out_a: Vec<Arc<dyn QuantMethod>>,
    out_b: Arc<dyn QuantMethod>,
    rope: Arc<DeepSeekV2RotaryEmbedding>,
    compress_rope: Option<Arc<DeepSeekV2RotaryEmbedding>>,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc_attn: HyperWeights,
    hc_ffn: HyperWeights,
    experts: MoEExperts,
    router: Arc<dyn QuantMethod>,
    router_bias: Tensor,
    shared_gate: Arc<dyn QuantMethod>,
    shared_up: Arc<dyn QuantMethod>,
    shared_down: Arc<dyn QuantMethod>,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    engram: Option<EngramLayer>,
}

struct Compressor {
    kv: Arc<dyn QuantMethod>,
    gate: Option<Arc<dyn QuantMethod>>,
    norm: Tensor,
}

struct Indexer {
    q_b: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    k: Option<Arc<dyn QuantMethod>>,
    k_norm: Option<Tensor>,
}

/// The Engram table stays in the GGUF map. A token reads a few rows; copying the
/// whole table (tens of GiB, once per rank) does not fit next to the expert shard.
struct EngramTable {
    archive: Arc<GgufArchive>,
    name: String,
}

impl EngramTable {
    fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        let flat = ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let rows = self
            .archive
            .dequantize_rows(&self.name, &flat, ids.device())?;
        let mut shape = ids.dims().to_vec();
        shape.push(rows.dim(D::Minus1)?);
        rows.reshape(shape)
    }
}

struct EngramLayer {
    table: EngramTable,
    wkv: Tensor,
    q: Tensor,
    k: Tensor,
    hash_index: usize,
}

struct LayerState {
    window: Option<Tensor>,
    compressed: Option<Tensor>,
    index_k: Option<Tensor>,
    partial_kv: Option<Tensor>,
    partial_score: Option<Tensor>,
    partial: usize,
}

struct RunState {
    tokens: Vec<u32>,
    last_pre: Option<Tensor>,
    layers: Vec<LayerState>,
    /// Compressed-row indices from the most recent indexer, into that owner's rows.
    index_ids: Option<(usize, Tensor)>,
}

impl ModelWeights {
    /// Load `deepseek41` from GGUF shards. Experts stay quantized.
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
        let hidden = meta_usize(&archive, &key("embedding_length"))?;
        let heads = meta_usize(&archive, &key("attention.head_count"))?;
        let head_dim = meta_usize(&archive, &key("attention.key_length"))?;
        let rope_dim = meta_usize(&archive, &key("rope.dimension_count"))?;
        let q_lora = meta_usize(&archive, &key("attention.q_lora_rank"))?;
        let window = meta_usize(&archive, &key("attention.sliding_window"))?;
        let experts = meta_usize(&archive, &key("expert_count"))?;
        let expert_k = meta_usize(&archive, &key("expert_used_count"))?;
        let intermediate = meta_usize(&archive, &key("expert_feed_forward_length"))?;
        let groups = meta_usize(&archive, &key("attention.output_group_count"))?;
        let rank = meta_usize(&archive, &key("attention.output_lora_rank"))?;
        let hc = meta_usize(&archive, &key("hyper_connection.count"))?;
        let sinkhorn_iters = meta_usize(&archive, &key("hyper_connection.sinkhorn_iterations"))?;
        let rms_eps = meta_f32(&archive, &key("attention.layer_norm_rms_epsilon"))?;
        let hc_eps = meta_f32(&archive, &key("hyper_connection.epsilon"))?;
        let route_scale = f64::from(meta_f32(&archive, &key("expert_weights_scale"))?);
        let weights_norm = meta_flag(&archive, &key("expert_weights_norm"))?;
        if !weights_norm {
            candle_core::bail!("deepseek41 expert weights must be renormalized");
        }
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) || rope_dim > head_dim {
            candle_core::bail!("deepseek41 rope dimension {rope_dim} does not fit head dim {head_dim}");
        }
        if !heads.is_multiple_of(groups) || groups == 0 || hc < 2 {
            candle_core::bail!("deepseek41 head, group, or hyper-connection counts are unusable");
        }
        let ratios = meta_usize_array(&archive, &key("attention.compress_ratios"))?;
        if ratios.len() < blocks {
            candle_core::bail!(
                "deepseek41 compress_ratios has {} entries for {blocks} blocks",
                ratios.len()
            );
        }
        let clamps = meta_f32_array(&archive, &key("swiglu_clamp_exp"))?;
        let freq_base = meta_f32(&archive, &key("rope.freq_base"))?;
        let compress_base = meta_f32(&archive, &key("attention.compress_rope_freq_base"))?;
        let yarn_factor = meta_f32(&archive, &key("rope.scaling.factor"))?;
        let yarn_original = meta_usize(&archive, &key("rope.scaling.original_context_length"))?;
        let beta_fast = meta_f32(&archive, &key("rope.scaling.yarn_beta_fast"))?;
        let beta_slow = meta_f32(&archive, &key("rope.scaling.yarn_beta_slow"))?;
        let context = meta_usize(&archive, &key("context_length")).unwrap_or(ROPE_TABLE);
        let rope_len = context.min(ROPE_TABLE).max(8);
        let rope = Arc::new(build_rope(freq_base, rope_dim, rope_len, None, device)?);
        let compress_rope = Arc::new(build_rope(
            compress_base,
            rope_dim,
            rope_len,
            Some(Yarn {
                factor: yarn_factor,
                original: yarn_original,
                beta_fast,
                beta_slow,
            }),
            device,
        )?);
        let index_heads = meta_usize(&archive, &key("attention.indexer.head_count"))?;
        let index_dim = meta_usize(&archive, &key("attention.indexer.key_length"))?;
        let index_topk = meta_usize(&archive, &key("attention.indexer.top_k"))?;
        let engram_layers = meta_usize_array(&archive, &key("engram.layer_ids")).unwrap_or_default();
        let engram = load_engram_hash(&archive, &engram_layers)?;

        let mut bindings = GgufBindingMap::new();
        for layer in 0..blocks {
            bind_experts(&mut bindings, layer);
            let group_rows = rank;
            for group in 0..groups {
                bindings.insert(
                    format!("blk.{layer}.attn_output_a.{group}.weight"),
                    GgufTensorBinding::tensor(format!("blk.{layer}.attn_output_a.weight")).slice(
                        0,
                        group * group_rows,
                        group_rows,
                    ),
                );
            }
        }
        let source = Arc::new(GgufWeightSource::new(archive.clone(), &bindings, DType::F32)?);
        let vb = source.sharded_var_builder(device.clone());

        let embed = linear(&archive, "token_embd.weight", device)?;
        let norm = dense(&archive, "output_norm.weight", device)?;
        let output = if archive.contains_tensor("output.weight") {
            linear(&archive, "output.weight", device)?
        } else {
            linear(&archive, "token_embd.weight", device)?
        };

        let mut owners: Vec<Option<usize>> = vec![None; blocks];
        let mut last_for_ratio: Vec<(usize, usize)> = Vec::new();
        for layer in 0..blocks {
            let ratio = ratios[layer];
            if ratio == 0 {
                continue;
            }
            if archive.contains_tensor(&blk(layer, "attn_compressor_kv.weight")) {
                last_for_ratio.retain(|(prev_ratio, _)| *prev_ratio != ratio);
                last_for_ratio.push((ratio, layer));
            }
            owners[layer] = last_for_ratio
                .iter()
                .rev()
                .find(|(prev_ratio, _)| *prev_ratio == ratio)
                .map(|(_, owner)| *owner);
            if owners[layer].is_none() {
                candle_core::bail!(
                    "deepseek41 layer {layer} has compress ratio {ratio} and no compressor owner"
                );
            }
        }

        let mut layers = Vec::with_capacity(blocks);
        for layer in 0..blocks {
            let ratio = ratios[layer];
            require(&archive, &blk(layer, "attn_q_a.weight"))?;
            require(&archive, &blk(layer, "ffn_gate_exps.weight"))?;
            require(&archive, &blk(layer, "ffn_up_exps.weight"))?;
            require(&archive, &blk(layer, "ffn_down_exps.weight"))?;
            let experts_layer = MoEExperts::new(
                &MoEExpertsConfig {
                    num_experts: experts,
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
            )?;
            let compressor = if archive.contains_tensor(&blk(layer, "attn_compressor_kv.weight")) {
                Some(Compressor {
                    kv: linear(&archive, &blk(layer, "attn_compressor_kv.weight"), device)?,
                    gate: if ratio > 1 {
                        require(&archive, &blk(layer, "attn_compressor_gate.weight"))?;
                        Some(linear(
                            &archive,
                            &blk(layer, "attn_compressor_gate.weight"),
                            device,
                        )?)
                    } else {
                        None
                    },
                    norm: dense(&archive, &blk(layer, "attn_compressor_norm.weight"), device)?,
                })
            } else {
                None
            };
            let indexer = if archive.contains_tensor(&blk(layer, "indexer.attn_q_b.weight")) {
                require(&archive, &blk(layer, "indexer.proj.weight"))?;
                let owns_k = archive.contains_tensor(&blk(layer, "indexer.attn_k.weight"));
                if owns_k {
                    require(&archive, &blk(layer, "indexer.k_norm.weight"))?;
                }
                Some(Indexer {
                    q_b: linear(&archive, &blk(layer, "indexer.attn_q_b.weight"), device)?,
                    proj: linear(&archive, &blk(layer, "indexer.proj.weight"), device)?,
                    k_norm: if owns_k {
                        Some(dense(&archive, &blk(layer, "indexer.k_norm.weight"), device)?)
                    } else {
                        None
                    },
                    k: if owns_k {
                        Some(linear(&archive, &blk(layer, "indexer.attn_k.weight"), device)?)
                    } else {
                        None
                    },
                })
            } else {
                None
            };
            if ratio > 0 && indexer.is_none() && owners[layer].is_none() {
                candle_core::bail!("deepseek41 layer {layer} has no indexer");
            }
            let engram_layer = engram_layers.iter().position(|id| *id == layer);
            let engram_weights = if let Some(hash_index) = engram_layer {
                require(&archive, &blk(layer, "engram_embd.weight"))?;
                require(&archive, &blk(layer, "engram_wkv.weight"))?;
                require(&archive, &blk(layer, "engram_q.weight"))?;
                require(&archive, &blk(layer, "engram_k.weight"))?;
                Some(EngramLayer {
                    table: EngramTable {
                        archive: archive.clone(),
                        name: blk(layer, "engram_embd.weight"),
                    },
                    wkv: dense(&archive, &blk(layer, "engram_wkv.weight"), device)?,
                    q: dense(&archive, &blk(layer, "engram_q.weight"), device)?,
                    k: dense(&archive, &blk(layer, "engram_k.weight"), device)?,
                    hash_index,
                })
            } else {
                None
            };
            let mut out_a = Vec::with_capacity(groups);
            for group in 0..groups {
                let loaded = source
                    .load_linear(
                        &format!("blk.{layer}.attn_output_a.{group}"),
                        device,
                        Shard::default(),
                    )?
                    .ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "deepseek41 layer {layer} grouped output {group} did not load"
                        ))
                    })?;
                out_a.push(loaded);
            }
            let clamp = clamps.get(layer).copied().unwrap_or(10.0);
            layers.push(Layer {
                index: layer,
                ratio,
                window,
                heads,
                head_dim,
                rope_dim,
                groups,
                index_heads,
                index_dim,
                index_topk,
                expert_k,
                route_scale,
                swiglu_limit: f64::from(clamp),
                kv_owner: owners[layer],
                q_a: linear(&archive, &blk(layer, "attn_q_a.weight"), device)?,
                q_a_norm: dense(&archive, &blk(layer, "attn_q_a_norm.weight"), device)?,
                q_b: linear(&archive, &blk(layer, "attn_q_b.weight"), device)?,
                kv: linear(&archive, &blk(layer, "attn_kv.weight"), device)?,
                kv_norm: dense(&archive, &blk(layer, "attn_kv_a_norm.weight"), device)?,
                sinks: dense(&archive, &blk(layer, "attn_sinks.weight"), device)?,
                out_a,
                out_b: linear(&archive, &blk(layer, "attn_output_b.weight"), device)?,
                rope: rope.clone(),
                compress_rope: (ratio > 0).then(|| compress_rope.clone()),
                attn_norm: dense(&archive, &blk(layer, "attn_norm.weight"), device)?,
                ffn_norm: dense(&archive, &blk(layer, "ffn_norm.weight"), device)?,
                hc_attn: hyper(&archive, layer, "attn", device)?,
                hc_ffn: hyper(&archive, layer, "ffn", device)?,
                experts: experts_layer,
                router: linear(&archive, &blk(layer, "ffn_gate_inp.weight"), device)?,
                router_bias: dense(&archive, &blk(layer, "exp_probs_b.bias"), device)?,
                shared_gate: linear(&archive, &blk(layer, "ffn_gate_shexp.weight"), device)?,
                shared_up: linear(&archive, &blk(layer, "ffn_up_shexp.weight"), device)?,
                shared_down: linear(&archive, &blk(layer, "ffn_down_shexp.weight"), device)?,
                compressor,
                indexer,
                engram: engram_weights,
            });
            let q_shape = archive
                .tensor_info(&blk(layer, "attn_q_a.weight"))?
                .shape();
            if q_shape != [q_lora, hidden] && q_shape != [hidden, q_lora] {
                candle_core::bail!(
                    "deepseek41 `{}.attn_q_a.weight` shape {q_shape:?} does not match q_lora {q_lora} and hidden {hidden}",
                    format!("blk.{layer}")
                );
            }
        }

        let layer_states = (0..blocks)
            .map(|_| LayerState {
                window: None,
                compressed: None,
                index_k: None,
                partial_kv: None,
                partial_score: None,
                partial: 0,
            })
            .collect();
        Ok(Self {
            embed,
            layers,
            norm,
            output,
            hc,
            sinkhorn_iters,
            rms_eps,
            hc_eps,
            engram,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(blocks, context)),
            max_seq_len: context,
            state: Mutex::new(RunState {
                tokens: Vec::new(),
                last_pre: None,
                layers: layer_states,
                index_ids: None,
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
            candle_core::bail!("deepseek41 serves one sequence at a time");
        }
        let start = *seqlen_offsets.first().unwrap_or(&0);
        if seqlen_offsets.iter().any(|offset| *offset != start) {
            candle_core::bail!("deepseek41 sequence offsets disagree");
        }
        if start > 0 && seq != 1 {
            candle_core::bail!(
                "deepseek41 continuation of {seq} tokens is not supported; prefill the prompt or decode one token"
            );
        }
        let ids = input_ids.reshape((batch, seq))?.to_dtype(DType::U32)?;
        let new_tokens = ids.flatten_all()?.to_vec1::<u32>()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("deepseek41 state lock was poisoned"))?;
        if start == 0 {
            state.tokens.clear();
            state.last_pre = None;
            state.index_ids = None;
            for layer in &mut state.layers {
                *layer = LayerState {
                    window: None,
                    compressed: None,
                    index_k: None,
                    partial_kv: None,
                    partial_score: None,
                    partial: 0,
                };
            }
        }
        state.tokens.extend(new_tokens);
        let token = self.embed.embedding_forward(&ids, DType::F32)?;
        let token = token.unsqueeze(2)?;
        let mut parts = Vec::with_capacity(self.hc);
        for _ in 0..self.hc {
            parts.push(token.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let mut hidden = Tensor::cat(&refs, 2)?;
        let pre = if let Some(prev) = &state.last_pre {
            prev.clone()
        } else {
            let ones = Tensor::ones((batch, seq, 1), DType::F32, &self.device)?;
            let zeros = Tensor::zeros((batch, seq, self.hc - 1), DType::F32, &self.device)?;
            Tensor::cat(&[&ones, &zeros], 2)?
        };
        let positions = crate::pipeline::text_positions_tensor(&[start], seq, &self.device)?;
        if start + seq > self.layers.first().map(|layer| layer.rope_len()).unwrap_or(ROPE_TABLE) {
            candle_core::bail!(
                "deepseek41 position {} is past the rope table",
                start + seq
            );
        }
        let mut incoming = pre;
        for layer in &self.layers {
            if let Some(engram) = &layer.engram {
                hidden = apply_engram(
                    &hidden,
                    engram,
                    self.engram.as_ref(),
                    &state.tokens,
                    self.rms_eps,
                    &self.device,
                )?;
            }
            let attn_mix = layer.hc_attn_weights();
            let ffn_mix = layer.hc_ffn_weights();
            let step = Step {
                start,
                positions: &positions,
                state: &mut state,
            };
            // `lagged_block` takes Fn. The step's state is exclusively borrowed for this layer.
            let (next, next_pre) = {
                let layer_attn = |x: &Tensor| layer.attend(x, step.start, step.positions, step.state);
                let layer_ffn = |x: &Tensor| layer.feed_forward(x);
                lagged_block(
                    &hidden,
                    &incoming,
                    &attn_mix,
                    &ffn_mix,
                    &layer.attn_norm,
                    &layer.ffn_norm,
                    self.hc,
                    self.sinkhorn_iters,
                    self.rms_eps,
                    self.hc_eps,
                    layer_attn,
                    layer_ffn,
                )?
            };
            hidden = next;
            incoming = next_pre;
        }
        let mixed = crate::gguf::dsv4_hc::mix_streams(&hidden, &incoming)?;
        state.last_pre = Some(incoming.narrow(1, seq - 1, 1)?);
        drop(state);
        let normed = rmsnorm(&mixed, &self.norm, f64::from(self.rms_eps))?;
        let logits = self.output.forward(&normed)?;
        extract_logits(&logits, context_lens)
    }
}

struct Step<'a> {
    start: usize,
    positions: &'a Tensor,
    state: &'a mut RunState,
}

struct Yarn {
    factor: f32,
    original: usize,
    beta_fast: f32,
    beta_slow: f32,
}

fn build_rope(
    theta: f32,
    dim: usize,
    len: usize,
    yarn: Option<Yarn>,
    device: &Device,
) -> Result<DeepSeekV2RotaryEmbedding> {
    let scaling = yarn.map(|yarn| DeepSeekV2RopeScaling::Yarn {
        original_max_position_embeddings: yarn.original,
        beta_fast: yarn.beta_fast,
        beta_slow: yarn.beta_slow,
        mscale: 1.0,
        mscale_all_dim: 1.0,
        factor: yarn.factor,
        scaling_type: ScaledRopeType::Yarn,
    });
    DeepSeekV2RotaryEmbedding::new(
        &DeepSeekV2RopeConfig {
            rope_scaling: scaling,
            max_position_embeddings: len,
            rope_theta: theta,
            qk_rope_head_dim: dim,
        },
        DType::F32,
        device,
    )
}

impl Layer {
    fn rope_len(&self) -> usize {
        ROPE_TABLE
    }

    fn hc_attn_weights(&self) -> HyperWeights {
        HyperWeights {
            fn_weight: self.hc_attn.fn_weight.clone(),
            scale: self.hc_attn.scale.clone(),
            base: self.hc_attn.base.clone(),
        }
    }

    fn hc_ffn_weights(&self) -> HyperWeights {
        HyperWeights {
            fn_weight: self.hc_ffn.fn_weight.clone(),
            scale: self.hc_ffn.scale.clone(),
            base: self.hc_ffn.base.clone(),
        }
    }

    fn attend(
        &self,
        x: &Tensor,
        start: usize,
        positions: &Tensor,
        state: &mut RunState,
    ) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let q_low = rmsnorm(
            &self.q_a.forward(x)?,
            &self.q_a_norm,
            f64::from(self.rms_eps()),
        )?;
        let q = self.q_b.forward(&q_low)?;
        let q = q.reshape((batch, seq, self.heads, self.head_dim))?;
        let q = q.transpose(1, 2)?;
        let q = rotate_tail(&self.rope, &q, positions, self.rope_dim)?;
        let kv = rmsnorm(&self.kv.forward(x)?, &self.kv_norm, f64::from(self.rms_eps()))?;
        let kv = kv.unsqueeze(1)?;
        let kv = rotate_tail(&self.rope, &kv, positions, self.rope_dim)?;
        let kv = kv.squeeze(1)?;
        let window = self.push_window(&kv, start, &mut state.layers[self.index])?;
        let window_len = window.dim(1)?;
        let mut keys = window;
        let mut compressed_len = 0;
        if self.ratio > 0 {
            let compressed = self.compress(x, &q_low, start, state)?;
            compressed_len = compressed.dim(1)?;
            if compressed_len > 0 {
                keys = Tensor::cat(&[&keys, &compressed], 1)?;
            }
        }
        let visible = attention_visible(
            seq,
            window_len,
            compressed_len,
            start,
            self.window,
            self.ratio,
            x.device(),
        )?;
        let mixed = sparse_mqa(&q.transpose(1, 2)?, &keys, &self.sinks, &visible)?;
        let mixed = mixed.transpose(1, 2)?;
        let mixed = inverse_rotate_tail(&self.rope, &mixed, positions, self.rope_dim)?;
        let mixed = mixed.transpose(1, 2)?.contiguous()?;
        let group_in = self.heads / self.groups * self.head_dim;
        let grouped = mixed.reshape((batch, seq, self.groups, group_in))?;
        let mut low = Vec::with_capacity(self.groups);
        for (group, weight) in self.out_a.iter().enumerate() {
            let slice = grouped.narrow(2, group, 1)?.squeeze(2)?;
            low.push(weight.forward(&slice)?);
        }
        let refs: Vec<&Tensor> = low.iter().collect();
        let low = Tensor::cat(&refs, D::Minus1)?;
        self.out_b.forward(&low)
    }

    fn rms_eps(&self) -> f32 {
        // The block already normalized with the model epsilon. Attention's
        // own norms use the same value; it is stored on the tensors' consumer.
        1e-6
    }

    fn push_window(&self, kv: &Tensor, start: usize, state: &mut LayerState) -> Result<Tensor> {
        let history = if start == 0 {
            kv.clone()
        } else if let Some(prev) = &state.window {
            Tensor::cat(&[prev, kv], 1)?
        } else {
            candle_core::bail!("deepseek41 layer {} is missing its window cache", self.index)
        };
        let keep = history.dim(1)?.min(self.window);
        let start_keep = history.dim(1)? - keep;
        let kept = history.narrow(1, start_keep, keep)?;
        state.window = Some(kept.clone());
        Ok(history)
    }

    fn compress(
        &self,
        x: &Tensor,
        q_low: &Tensor,
        start: usize,
        state: &mut RunState,
    ) -> Result<Tensor> {
        let owner = self.kv_owner.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "deepseek41 layer {} has no compressed-kv owner",
                self.index
            ))
        })?;
        if self.compressor.is_some() {
            let fresh = self.pool(x, start, &mut state.layers[self.index])?;
            if fresh.dim(1)? > 0 {
                let roped = self.rope_compressed(&fresh, start)?;
                let prev = state.layers[owner].compressed.clone();
                state.layers[owner].compressed = Some(match prev {
                    Some(prev) => Tensor::cat(&[&prev, &roped], 1)?,
                    None => roped,
                });
                if let Some(indexer) = &self.indexer {
                    if let Some(k_proj) = &indexer.k {
                        let k = rmsnorm(
                            &k_proj.forward(&fresh)?,
                            indexer.k_norm.as_ref().ok_or_else(|| {
                                candle_core::Error::msg("deepseek41 indexer is missing k norm")
                            })?,
                            f64::from(self.rms_eps()),
                        )?;
                        let k = self.rope_compressed(&k, start)?;
                        let prev = state.layers[owner].index_k.clone();
                        state.layers[owner].index_k = Some(match prev {
                            Some(prev) => Tensor::cat(&[&prev, &k], 1)?,
                            None => k,
                        });
                    }
                }
            }
        }
        let compressed = match state.layers[owner].compressed.clone() {
            Some(rows) => rows,
            None => Tensor::zeros((1, 0, self.head_dim), DType::F32, x.device())?,
        };
        if let Some(indexer) = &self.indexer {
            let ids = self.index_rows(indexer, q_low, x, owner, start, state)?;
            state.index_ids = Some((owner, ids));
        }
        Ok(compressed)
    }

    fn pool(&self, x: &Tensor, start: usize, state: &mut LayerState) -> Result<Tensor> {
        let compressor = self.compressor.as_ref().ok_or_else(|| {
            candle_core::Error::msg(format!("deepseek41 layer {} has no compressor", self.index))
        })?;
        let kv = compressor.kv.forward(x)?;
        if self.ratio == 1 {
            return rmsnorm(&kv, &compressor.norm, f64::from(self.rms_eps()));
        }
        let gate = compressor
            .gate
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("deepseek41 ratio > 1 compressor has no gate"))?
            .forward(x)?;
        let (kv, score) = if start == 0 {
            state.partial = 0;
            state.partial_kv = None;
            state.partial_score = None;
            (kv, gate)
        } else {
            match (&state.partial_kv, &state.partial_score) {
                (Some(prev_kv), Some(prev_score)) if state.partial > 0 => (
                    Tensor::cat(&[prev_kv, &kv], 1)?,
                    Tensor::cat(&[prev_score, &gate], 1)?,
                ),
                _ => (kv, gate),
            }
        };
        let pooled = compress_groups(&kv, &score, self.ratio)?;
        let used = pooled.dim(1)? * self.ratio;
        let tail = kv.dim(1)? - used;
        if tail > 0 {
            state.partial_kv = Some(kv.narrow(1, used, tail)?);
            state.partial_score = Some(score.narrow(1, used, tail)?);
            state.partial = tail;
        } else {
            state.partial_kv = None;
            state.partial_score = None;
            state.partial = 0;
        }
        rmsnorm(&pooled, &compressor.norm, f64::from(self.rms_eps()))
    }

    fn rope_compressed(&self, rows: &Tensor, start: usize) -> Result<Tensor> {
        let groups = rows.dim(1)?;
        if groups == 0 {
            return Ok(rows.clone());
        }
        let first = start - (start % self.ratio);
        let mut positions = Vec::with_capacity(groups);
        for group in 0..groups {
            positions.push(u32::try_from(first + group * self.ratio).map_err(candle_core::Error::wrap)?);
        }
        let positions = Tensor::from_vec(positions, (groups,), rows.device())?;
        let rows = rows.unsqueeze(1)?;
        let roped = rotate_tail(
            self.compress_rope.as_ref().ok_or_else(|| {
                candle_core::Error::msg("deepseek41 compressed rope is missing")
            })?,
            &rows,
            &positions,
            self.rope_dim,
        )?;
        roped.squeeze(1)
    }

    fn index_rows(
        &self,
        indexer: &Indexer,
        q_low: &Tensor,
        x: &Tensor,
        owner: usize,
        start: usize,
        state: &RunState,
    ) -> Result<Tensor> {
        let Some(keys) = state.layers[owner].index_k.clone() else {
            let (batch, seq, _) = x.dims3()?;
            return Tensor::zeros((batch, seq, 0), DType::U32, x.device());
        };
        let (batch, seq, _) = x.dims3()?;
        let q = indexer.q_b.forward(q_low)?;
        let q = q.reshape((batch, seq, self.index_heads, self.index_dim))?;
        let weight = indexer.proj.forward(x)?;
        let weight = weight.reshape((batch, seq, self.index_heads))?;
        let scale = (self.index_dim as f64).sqrt().recip() / (self.index_heads as f64).sqrt();
        let weight = (weight * scale)?;
        let t = keys.dim(1)?;
        let mut score = Tensor::zeros((batch, seq, t), DType::F32, x.device())?;
        let keys_t = keys.transpose(1, 2)?;
        for head in 0..self.index_heads {
            let qh = q.narrow(2, head, 1)?.squeeze(2)?;
            let mut dots = qh.matmul(&keys_t)?;
            dots = dots.relu()?;
            let wh = weight.narrow(2, head, 1)?;
            dots = dots.broadcast_mul(&wh)?;
            score = (score + dots)?;
        }
        let mut allowed = Vec::with_capacity(seq * t);
        for query in 0..seq {
            let visible = (start + query + 1) / self.ratio;
            for key in 0..t {
                allowed.push(u8::from(key < visible));
            }
        }
        let allowed = Tensor::from_vec(allowed, (1, seq, t), x.device())?;
        let blocked = (score.zeros_like()? - f64::INFINITY)?;
        let score = allowed
            .broadcast_as(score.shape())?
            .where_cond(&score, &blocked)?;
        let k = self.index_topk.min(t).max(1);
        let picked = score.topk(k)?;
        Ok(picked.indices)
    }

    fn feed_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = x.dims3()?;
        let flat = x.reshape((batch * seq, hidden))?;
        let logits = self.router.forward(&flat)?;
        let (ids, weights) = moe_route(&logits, &self.router_bias, self.expert_k, self.route_scale)?;
        let routed = self.experts.forward(x, weights, &ids)?;
        let gate = self.shared_gate.forward(x)?;
        let up = self.shared_up.forward(x)?;
        let mixed = swiglu_clamped(&gate, &up, self.swiglu_limit)?;
        let shared = self.shared_down.forward(&mixed)?;
        routed + shared
    }
}

fn apply_engram(
    h: &Tensor,
    layer: &EngramLayer,
    hash: Option<&EngramHash>,
    tokens: &[u32],
    eps: f32,
    device: &Device,
) -> Result<Tensor> {
    let hash = hash.ok_or_else(|| candle_core::Error::msg("deepseek41 Engram hash tables are missing"))?;
    let ids = engram_row_ids(
        tokens,
        &hash.token_map,
        hash.pad_token,
        hash.n_layers,
        hash.n_heads,
        hash.max_ngram,
        &hash.multipliers,
        &hash.primes,
        &hash.offsets,
    )?;
    let cols = (hash.max_ngram - 1) * hash.n_heads;
    let seq = h.dim(1)?;
    let start = tokens.len().saturating_sub(seq);
    let mut flat = Vec::with_capacity(seq * cols);
    for pos in start..tokens.len() {
        for col in 0..cols {
            let slot = (pos * hash.n_layers + layer.hash_index) * cols + col;
            flat.push(u32::try_from(ids[slot]).map_err(candle_core::Error::wrap)?);
        }
    }
    let flat = Tensor::from_vec(flat, (h.dim(0)?, seq, cols), device)?;
    let rows = layer.table.gather(&flat)?;
    let rows = rows.reshape((h.dim(0)?, seq, cols * hash.key_length))?;
    engram_apply(h, &rows, &layer.wkv, &layer.q, &layer.k, f64::from(eps))
}

fn rotate_tail(
    rope: &DeepSeekV2RotaryEmbedding,
    x: &Tensor,
    positions: &Tensor,
    rope_dim: usize,
) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let nope = x.narrow(D::Minus1, 0, dim - rope_dim)?;
    let tail = x.narrow(D::Minus1, dim - rope_dim, rope_dim)?;
    let (tail, _) = rope.forward(&tail.contiguous()?, &tail.contiguous()?, positions)?;
    Tensor::cat(&[&nope, &tail], D::Minus1)
}

fn inverse_rotate_tail(
    rope: &DeepSeekV2RotaryEmbedding,
    x: &Tensor,
    positions: &Tensor,
    rope_dim: usize,
) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let nope = x.narrow(D::Minus1, 0, dim - rope_dim)?;
    let tail = x.narrow(D::Minus1, dim - rope_dim, rope_dim)?;
    let tail = rope.inverse(&tail.contiguous()?, positions)?;
    Tensor::cat(&[&nope, &tail], D::Minus1)
}

fn attention_visible(
    seq: usize,
    window_len: usize,
    compressed_len: usize,
    start: usize,
    window: usize,
    ratio: usize,
    device: &Device,
) -> Result<Tensor> {
    let keys = window_len + compressed_len;
    let mut visible = Vec::with_capacity(seq * keys);
    for query in 0..seq {
        let abs = start + query;
        let oldest = abs.saturating_sub(window.saturating_sub(1));
        let groups = if ratio == 0 { 0 } else { (abs + 1) / ratio };
        for key in 0..keys {
            let allowed = if key < window_len {
                let key_abs = if start == 0 {
                    key
                } else {
                    start.saturating_sub(window.saturating_sub(1)) + key
                };
                key_abs <= abs && key_abs >= oldest
            } else {
                key - window_len < groups
            };
            visible.push(u8::from(allowed));
        }
    }
    Tensor::from_vec(visible, (seq, keys), device)
}

fn sparse_mqa(q: &Tensor, k: &Tensor, sinks: &Tensor, visible: &Tensor) -> Result<Tensor> {
    let (batch, _seq, heads, dim) = q.dims4()?;
    let scale = 1.0 / (dim as f64).sqrt();
    let q = q.transpose(1, 2)?;
    let k = k.unsqueeze(1)?;
    let scores = q.broadcast_matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
    let scores = (scores * scale)?;
    let blocked = (scores.zeros_like()? - f64::INFINITY)?;
    let scores = visible
        .broadcast_as(scores.shape())?
        .where_cond(&scores, &blocked)?;
    let row_max = scores.max_keepdim(D::Minus1)?;
    let exp_scores = scores.broadcast_sub(&row_max)?.exp()?;
    let sink = sinks
        .reshape((1, heads, 1, 1))?
        .broadcast_sub(&row_max)?
        .exp()?;
    let denom = (exp_scores.sum_keepdim(D::Minus1)? + sink)?;
    let probs = exp_scores.broadcast_div(&denom)?;
    let out = probs.broadcast_matmul(&k.reshape((batch, 1, k.dim(2)?, dim))?)?;
    out.transpose(1, 2)?.contiguous()
}

fn load_engram_hash(archive: &GgufArchive, layers: &[usize]) -> Result<Option<EngramHash>> {
    if layers.is_empty() {
        return Ok(None);
    }
    let n_heads = meta_usize(archive, &key("engram.head_count"))?;
    let key_length = meta_usize(archive, &key("engram.key_length"))?;
    let max_ngram = meta_usize(archive, &key("engram.max_ngram_size"))?;
    let pad = meta_usize(archive, &key("engram.pad_id"))?;
    Ok(Some(EngramHash {
        token_map: meta_u32_array(archive, &key("engram.token_map"))?,
        pad_token: u32::try_from(pad).map_err(candle_core::Error::wrap)?,
        n_layers: layers.len(),
        n_heads,
        max_ngram,
        key_length,
        multipliers: meta_u64_array(archive, &key("engram.multipliers"))?,
        primes: meta_u64_array(archive, &key("engram.primes"))?,
        offsets: meta_u64_array(archive, &key("engram.offsets"))?,
    }))
}

fn hyper(archive: &GgufArchive, layer: usize, which: &str, device: &Device) -> Result<HyperWeights> {
    Ok(HyperWeights {
        fn_weight: dense(archive, &blk(layer, &format!("hc_{which}_fn.weight")), device)?,
        scale: dense(archive, &blk(layer, &format!("hc_{which}_scale.weight")), device)?,
        base: dense(archive, &blk(layer, &format!("hc_{which}_base.weight")), device)?,
    })
}

fn bind_experts(bindings: &mut GgufBindingMap, layer: usize) {
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

fn linear(archive: &GgufArchive, name: &str, device: &Device) -> Result<Arc<dyn QuantMethod>> {
    let weight = archive.load_qtensor(name, device)?;
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(weight),
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
        candle_core::bail!("GGUF deepseek41 is missing `{name}`")
    }
}

fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

fn blk(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

fn meta_value<'a>(archive: &'a GgufArchive, name: &str) -> Result<&'a candle_core::quantized::gguf_file::Value> {
    archive.metadata_value(name).ok_or_else(|| {
        candle_core::Error::msg(format!("GGUF deepseek41 is missing metadata `{name}`"))
    })
}

fn meta_string(archive: &GgufArchive, name: &str) -> Result<String> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::String(value) => Ok(value.clone()),
        _ => candle_core::bail!("GGUF metadata `{name}` must be a string"),
    }
}

fn meta_usize(archive: &GgufArchive, name: &str) -> Result<usize> {
    value_usize(meta_value(archive, name)?, name)
}

fn meta_f32(archive: &GgufArchive, name: &str) -> Result<f32> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::F32(value) => Ok(*value),
        candle_core::quantized::gguf_file::Value::F64(value) => Ok(*value as f32),
        other => value_usize(other, name).map(|value| value as f32),
    }
}

fn meta_flag(archive: &GgufArchive, name: &str) -> Result<bool> {
    match meta_value(archive, name)? {
        candle_core::quantized::gguf_file::Value::Bool(value) => Ok(*value),
        other => Ok(value_usize(other, name)? != 0),
    }
}

fn meta_usize_array(archive: &GgufArchive, name: &str) -> Result<Vec<usize>> {
    let candle_core::quantized::gguf_file::Value::Array(values) = meta_value(archive, name)? else {
        candle_core::bail!("GGUF metadata `{name}` must be an array");
    };
    values.iter().map(|value| value_usize(value, name)).collect()
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
            other => value_usize(other, name).map(|inner| inner as f32),
        })
        .collect()
}

fn meta_u32_array(archive: &GgufArchive, name: &str) -> Result<Vec<u32>> {
    meta_usize_array(archive, name)?
        .into_iter()
        .map(|value| u32::try_from(value).map_err(candle_core::Error::wrap))
        .collect()
}

fn meta_u64_array(archive: &GgufArchive, name: &str) -> Result<Vec<u64>> {
    meta_usize_array(archive, name)?
        .into_iter()
        .map(|value| u64::try_from(value).map_err(candle_core::Error::wrap))
        .collect()
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_core::quantized::GgmlDType;
    use mistralrs_quant::{Comm, Id};
    use tempfile::NamedTempFile;

    use super::ModelWeights;

    struct Meta {
        key: &'static str,
        value: MetaValue,
    }

    enum MetaValue {
        Str(&'static str),
        U32(u32),
        F32(f32),
        U32s(Vec<u32>),
        F32s(Vec<f32>),
    }

    struct Weight {
        name: String,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
    }

    fn pattern(shape: &[u64], salt: f32) -> Vec<f32> {
        let count: u64 = shape.iter().product();
        (0..count)
            .map(|index| salt + ((index % 13) as f32 - 6.0) * 0.05)
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn q2k(shape: &[u64], salt: f32) -> Vec<u8> {
        let values = pattern(shape, salt);
        let dims: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
        let tensor = Tensor::from_vec(values, dims.as_slice(), &Device::Cpu).expect("pattern");
        let quantized = candle_core::quantized::QTensor::quantize(&tensor, GgmlDType::Q2K)
            .expect("q2k");
        quantized.data().expect("q2k bytes").to_vec()
    }

    fn write_gguf(metadata: &[Meta], tensors: &[Weight]) -> NamedTempFile {
        let alignment = 32usize;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for item in metadata {
            write_string(&mut bytes, item.key);
            match &item.value {
                MetaValue::Str(value) => {
                    bytes.extend_from_slice(&8u32.to_le_bytes());
                    write_string(&mut bytes, value);
                }
                MetaValue::U32(value) => {
                    bytes.extend_from_slice(&4u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetaValue::F32(value) => {
                    bytes.extend_from_slice(&6u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetaValue::U32s(values) => {
                    bytes.extend_from_slice(&9u32.to_le_bytes());
                    bytes.extend_from_slice(&4u32.to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                MetaValue::F32s(values) => {
                    bytes.extend_from_slice(&9u32.to_le_bytes());
                    bytes.extend_from_slice(&6u32.to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        let mut offset = 0usize;
        let mut offsets = Vec::new();
        for tensor in tensors {
            offsets.push(offset as u64);
            offset += tensor.data.len();
            offset = offset.div_ceil(alignment) * alignment;
        }
        for (tensor, tensor_offset) in tensors.iter().zip(offsets.iter()) {
            write_string(&mut bytes, &tensor.name);
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dimension in tensor.shape.iter().rev() {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.dtype.to_le_bytes());
            bytes.extend_from_slice(&tensor_offset.to_le_bytes());
        }
        let data_offset = bytes.len().div_ceil(alignment) * alignment;
        bytes.resize(data_offset, 0);
        let mut cursor = 0usize;
        for tensor in tensors {
            let start = data_offset + cursor;
            bytes.resize(start, 0);
            bytes.extend_from_slice(&tensor.data);
            cursor += tensor.data.len();
            cursor = cursor.div_ceil(alignment) * alignment;
            bytes.resize(data_offset + cursor, 0);
        }
        let mut file = NamedTempFile::new().expect("temp gguf");
        std::io::Write::write_all(&mut file, &bytes).expect("write gguf");
        file
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn base_meta(ratio: u32) -> Vec<Meta> {
        vec![
            Meta { key: "general.architecture", value: MetaValue::Str("deepseek41") },
            Meta { key: "general.alignment", value: MetaValue::U32(32) },
            Meta { key: "deepseek41.block_count", value: MetaValue::U32(1) },
            Meta { key: "deepseek41.context_length", value: MetaValue::U32(32) },
            Meta { key: "deepseek41.embedding_length", value: MetaValue::U32(256) },
            Meta { key: "deepseek41.attention.head_count", value: MetaValue::U32(2) },
            Meta { key: "deepseek41.attention.head_count_kv", value: MetaValue::U32(1) },
            Meta { key: "deepseek41.attention.key_length", value: MetaValue::U32(64) },
            Meta { key: "deepseek41.attention.value_length", value: MetaValue::U32(64) },
            Meta { key: "deepseek41.rope.dimension_count", value: MetaValue::U32(4) },
            Meta { key: "deepseek41.rope.freq_base", value: MetaValue::F32(10_000.0) },
            Meta { key: "deepseek41.attention.compress_rope_freq_base", value: MetaValue::F32(160_000.0) },
            Meta { key: "deepseek41.rope.scaling.factor", value: MetaValue::F32(16.0) },
            Meta { key: "deepseek41.rope.scaling.original_context_length", value: MetaValue::U32(64) },
            Meta { key: "deepseek41.rope.scaling.yarn_beta_fast", value: MetaValue::F32(32.0) },
            Meta { key: "deepseek41.rope.scaling.yarn_beta_slow", value: MetaValue::F32(1.0) },
            Meta { key: "deepseek41.attention.layer_norm_rms_epsilon", value: MetaValue::F32(1e-5) },
            Meta { key: "deepseek41.attention.q_lora_rank", value: MetaValue::U32(32) },
            Meta { key: "deepseek41.attention.sliding_window", value: MetaValue::U32(8) },
            Meta { key: "deepseek41.expert_count", value: MetaValue::U32(2) },
            Meta { key: "deepseek41.expert_used_count", value: MetaValue::U32(1) },
            Meta { key: "deepseek41.expert_feed_forward_length", value: MetaValue::U32(256) },
            Meta { key: "deepseek41.expert_gating_func", value: MetaValue::U32(4) },
            Meta { key: "deepseek41.expert_weights_scale", value: MetaValue::F32(1.5) },
            Meta { key: "deepseek41.expert_weights_norm", value: MetaValue::U32(1) },
            Meta { key: "deepseek41.attention.output_group_count", value: MetaValue::U32(2) },
            Meta { key: "deepseek41.attention.output_lora_rank", value: MetaValue::U32(8) },
            Meta { key: "deepseek41.hyper_connection.count", value: MetaValue::U32(2) },
            Meta { key: "deepseek41.hyper_connection.sinkhorn_iterations", value: MetaValue::U32(1) },
            Meta { key: "deepseek41.hyper_connection.epsilon", value: MetaValue::F32(1e-6) },
            Meta { key: "deepseek41.attention.indexer.head_count", value: MetaValue::U32(2) },
            Meta { key: "deepseek41.attention.indexer.key_length", value: MetaValue::U32(8) },
            Meta { key: "deepseek41.attention.indexer.top_k", value: MetaValue::U32(4) },
            Meta { key: "deepseek41.attention.compress_ratios", value: MetaValue::U32s(vec![ratio]) },
            Meta { key: "deepseek41.swiglu_clamp_exp", value: MetaValue::F32s(vec![10.0]) },
            Meta { key: "deepseek41.engram.layer_ids", value: MetaValue::U32s(vec![]) },
        ]
    }

    fn f32_weight(name: &str, shape: &[u64], salt: f32) -> Weight {
        let values = pattern(shape, salt);
        Weight {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: 0,
            data: f32_bytes(&values),
        }
    }

    fn tiny_weights(expert_salt: f32, compressor: bool) -> Vec<Weight> {
        let mut weights = vec![
            f32_weight("token_embd.weight", &[16, 256], 0.1),
            f32_weight("output.weight", &[16, 256], 0.2),
            f32_weight("output_norm.weight", &[256], 1.0),
            f32_weight("blk.0.attn_norm.weight", &[256], 1.0),
            f32_weight("blk.0.ffn_norm.weight", &[256], 1.0),
            f32_weight("blk.0.attn_q_a.weight", &[32, 256], 0.3),
            f32_weight("blk.0.attn_q_a_norm.weight", &[32], 1.0),
            f32_weight("blk.0.attn_q_b.weight", &[128, 32], 0.4),
            f32_weight("blk.0.attn_kv.weight", &[64, 256], 0.5),
            f32_weight("blk.0.attn_kv_a_norm.weight", &[64], 1.0),
            f32_weight("blk.0.attn_sinks.weight", &[2], 0.0),
            f32_weight("blk.0.attn_output_a.weight", &[16, 64], 0.6),
            f32_weight("blk.0.attn_output_b.weight", &[256, 16], 0.7),
            f32_weight("blk.0.hc_attn_fn.weight", &[512, 8], 0.01),
            f32_weight("blk.0.hc_attn_scale.weight", &[3], 1.0),
            f32_weight("blk.0.hc_attn_base.weight", &[8], 0.0),
            f32_weight("blk.0.hc_ffn_fn.weight", &[512, 8], 0.02),
            f32_weight("blk.0.hc_ffn_scale.weight", &[3], 1.0),
            f32_weight("blk.0.hc_ffn_base.weight", &[8], 0.0),
            f32_weight("blk.0.ffn_gate_inp.weight", &[2, 256], 0.8),
            f32_weight("blk.0.exp_probs_b.bias", &[2], 0.0),
            f32_weight("blk.0.ffn_gate_shexp.weight", &[256, 256], 0.15),
            f32_weight("blk.0.ffn_up_shexp.weight", &[256, 256], 0.16),
            f32_weight("blk.0.ffn_down_shexp.weight", &[256, 256], 0.17),
            Weight {
                name: "blk.0.ffn_gate_exps.weight".to_string(),
                shape: vec![2, 256, 256],
                dtype: 10,
                data: q2k(&[2, 256, 256], expert_salt),
            },
            Weight {
                name: "blk.0.ffn_up_exps.weight".to_string(),
                shape: vec![2, 256, 256],
                dtype: 10,
                data: q2k(&[2, 256, 256], expert_salt + 0.2),
            },
            Weight {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                shape: vec![2, 256, 256],
                dtype: 10,
                data: q2k(&[2, 256, 256], expert_salt + 0.4),
            },
        ];
        if compressor {
            weights.extend([
                f32_weight("blk.0.attn_compressor_kv.weight", &[64, 256], 0.9),
                f32_weight("blk.0.attn_compressor_gate.weight", &[64, 256], 0.4),
                f32_weight("blk.0.attn_compressor_norm.weight", &[64], 1.0),
                f32_weight("blk.0.indexer.attn_q_b.weight", &[16, 32], 0.25),
                f32_weight("blk.0.indexer.proj.weight", &[2, 256], 0.3),
                f32_weight("blk.0.indexer.attn_k.weight", &[8, 64], 0.35),
                f32_weight("blk.0.indexer.k_norm.weight", &[8], 1.0),
            ]);
        }
        weights
    }

    fn load(file: &NamedTempFile) -> ModelWeights {
        let path = PathBuf::from(file.path());
        let comm = Arc::new(
            Comm::from_device(Id::new(), &Device::Cpu, 0, 1).expect("comm"),
        );
        ModelWeights::load(&[path], &Device::Cpu, DType::F32, comm).expect("load deepseek41")
    }

    fn logits(model: &ModelWeights, tokens: &[u32]) -> Vec<f32> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).expect("ids");
        model
            .forward(&ids, &[0], vec![(tokens.len() - 1, 1)], None)
            .expect("forward")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values")
    }

    #[test]
    fn device_map_sizing_returns_the_stored_bytes() {
        let file = write_gguf(&base_meta(0), &tiny_weights(0.1, false));
        let mut opened = vec![std::fs::File::open(file.path()).expect("open tiny gguf")];
        let mut readers: Vec<&mut std::fs::File> = opened.iter_mut().collect();
        let model = crate::gguf::Content::from_readers(&mut readers).expect("read tiny gguf");
        let loader = crate::utils::gguf_metadata::GgufDeviceMapLoaderInner {
            model: &model,
            arch: model.arch(),
        };
        let layers = crate::pipeline::DeviceMappedModelLoader::layer_sizes_in_bytes(
            &loader,
            "cfg",
            DType::F32,
            1,
            None,
        )
        .expect("deepseek41 layer sizes");
        assert_eq!(layers.len(), 1);
        let non_mapped = crate::pipeline::DeviceMappedModelLoader::non_mapped_size_in_bytes(
            &loader,
            "cfg",
            DType::F32,
            1,
            None,
            None,
        )
        .expect("deepseek41 non-mapped size");
        let layered = model.quantized_storage_bytes(|name| name.starts_with("blk."));
        let global = model.quantized_storage_bytes(|name| !name.starts_with("blk."));
        assert_eq!(layers[0], layered);
        assert_eq!(non_mapped, global);
        assert!(layers[0] > 0);
        assert!(non_mapped > 0);
    }

    #[test]
    fn quantized_experts_change_the_logits_and_a_missing_stack_fails() {
        let file = write_gguf(&base_meta(0), &tiny_weights(0.1, false));
        let model = load(&file);
        let first = logits(&model, &[1, 2]);
        let second = logits(&model, &[4, 5]);
        assert_ne!(first, second, "different tokens must change deepseek41 logits");
        assert!(first.iter().any(|value| value.is_finite()));

        let other = write_gguf(&base_meta(0), &tiny_weights(1.7, false));
        let other = load(&other);
        let changed = logits(&other, &[1, 2]);
        assert_ne!(
            first, changed,
            "Q2_K expert bytes must be consumed by the forward"
        );

        let mut dropped = tiny_weights(0.1, false);
        dropped.retain(|weight| weight.name != "blk.0.ffn_gate_exps.weight");
        let broken = write_gguf(&base_meta(0), &dropped);
        let path = PathBuf::from(broken.path());
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).expect("comm"));
        let err = match ModelWeights::load(&[path], &Device::Cpu, DType::F32, comm) {
            Ok(_) => panic!("missing experts must fail closed"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("ffn_gate_exps"),
            "the missing expert tensor has to be named: {msg}"
        );
    }

    #[test]
    fn compressed_kv_is_part_of_the_forward() {
        let file = write_gguf(&base_meta(2), &tiny_weights(0.1, true));
        let model = load(&file);
        let first = logits(&model, &[1, 2, 3, 4]);
        let mut weights = tiny_weights(0.1, true);
        for weight in &mut weights {
            if weight.name == "blk.0.attn_compressor_kv.weight" {
                weight.data = f32_bytes(&pattern(&[64, 256], 2.5));
            }
        }
        let other = write_gguf(&base_meta(2), &weights);
        let other = load(&other);
        let changed = logits(&other, &[1, 2, 3, 4]);
        assert_ne!(first, changed, "compressor weights must change logits");

        let mut dropped = tiny_weights(0.1, true);
        dropped.retain(|weight| weight.name != "blk.0.attn_compressor_kv.weight");
        let broken = write_gguf(&base_meta(2), &dropped);
        let path = PathBuf::from(broken.path());
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).expect("comm"));
        let err = match ModelWeights::load(&[path], &Device::Cpu, DType::F32, comm) {
            Ok(_) => panic!("a ratio-2 layer without a compressor must fail"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("compressor"),
            "the failure has to name the compressor: {msg}"
        );
    }
}

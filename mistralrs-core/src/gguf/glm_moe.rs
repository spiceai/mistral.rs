#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF loader + dense forward for GLM-4.x / GLM-5.2 MoE models whose
//! `general.architecture == "glm-dsa"`.
//!
//! `glm-dsa` is structurally DeepSeek-V3: Multi-head Latent Attention (MLA) with a
//! LoRA-compressed Q/KV path, plus a sigmoid-gated MoE FFN with a shared expert and
//! a routing-bias correction term. It additionally carries a DeepSeek-Sparse-Attention
//! (DSA) "lightning indexer" and an MTP (next-n) head.
//!
//! This implementation runs attention **DENSE** (full softmax SDPA over the
//! reconstructed K/V): the DSA sparse-attention indexer is not implemented anywhere in
//! this codebase, and llama.cpp also evaluates this model with dense attention. The
//! per-layer `indexer.*` tensors and the `nextn.*` (MTP) tensors are therefore loaded
//! by name but intentionally skipped — they are not consumed by the dense forward.
//!
//! Structure mirrors `models::quantized_qwen3_moe` (GGUF reading idioms, `GgufMatMul`,
//! the block loop, KV cache) and `models::deepseek3` (the MLA attention forward and the
//! sigmoid/bias/group MoE gate). Routed experts are built as [`GgufMatMul`] and run
//! through `QuantMethod::gather_forward`, so i-quant expert dtypes (IQ1_S / IQ1_M /
//! IQ2_XXS / IQ3_XXS / IQ4_XS) flow through the i-quant gather fallback in
//! `mistralrs-quant/src/gguf/cuda.rs` (dequantize-per-expert), rather than candle's
//! fused `QMatMul::indexed_moe_forward` (which has no i-quant kernel).

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::ops::TopKLastDimOp;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

// Default fallback for models that don't specify context_length.
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

fn gguf_linear(qt: candle_core::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(qt),
        b: None,
    })?))
}

/// Standard dense SwiGLU MLP (leading dense layers + the MoE shared expert).
struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

/// Sigmoid-gated, bias-corrected top-k router (DeepSeek-V3 / GLM-MoE "noaux_tc").
///
/// `expert_group_count == 1` for this model, so the group-limited routing reduces to a
/// plain top-k over all experts; the group machinery is therefore a no-op and omitted.
struct MoeGate {
    weight: Tensor,                  // [n_experts, hidden] f32
    e_score_correction_bias: Tensor, // [n_experts] f32
    top_k: usize,
    weights_norm: bool,
    weights_scale: f64,
}

impl MoeGate {
    /// Returns `(topk_idx [n, top_k], topk_weight [n, top_k])`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_bs, _seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?)?;
        let scores = candle_nn::ops::sigmoid(&logits)?;

        // Selection uses scores + correction bias; the returned weights use raw scores.
        let scores_for_choice = scores.broadcast_add(&self.e_score_correction_bias.unsqueeze(0)?)?;
        let topk_idx = scores_for_choice.topk(self.top_k)?.indices;
        let mut topk_weight = scores.gather(&topk_idx, 1)?;

        if self.weights_norm {
            let denom = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denom)?;
        }
        topk_weight = (topk_weight * self.weights_scale)?;
        Ok((topk_idx, topk_weight))
    }
}

/// Routed MoE: top-k experts (i-quant-capable gather) + a single shared expert.
struct Moe {
    gate: MoeGate,
    gate_experts: Arc<dyn QuantMethod>,
    up_experts: Arc<dyn QuantMethod>,
    down_experts: Arc<dyn QuantMethod>,
    shared: Mlp,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden) = xs.dims3()?;
        let identity = xs.clone();

        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        let num_tokens = bs * seq_len;
        let xs_flat = xs.reshape((num_tokens, 1, hidden))?;

        // gather_forward -> qmatmul_indexed_moe_forward (i-quant fallback lives there).
        let gate = self.gate_experts.gather_forward(&xs_flat, &topk_idx)?;
        let up = self.up_experts.gather_forward(&xs_flat, &topk_idx)?;
        let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        let ys = self.down_experts.gather_forward(&activated, &topk_idx)?;

        // Weight by routing scores and sum over the selected experts.
        let topk_weight = topk_weight.to_dtype(ys.dtype())?;
        let routed = ys
            .broadcast_mul(&topk_weight.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((bs, seq_len, hidden))?;

        routed + self.shared.forward(&identity)?
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

/// MLA attention weights for one layer, evaluated dense.
struct LayerWeights {
    // Q path: down-proj -> RMSNorm -> up-proj.
    q_a_proj: Arc<dyn QuantMethod>,
    q_a_norm: QRmsNorm,
    q_b_proj: Arc<dyn QuantMethod>,
    // KV path: down-proj (to kv_lora + rope) -> RMSNorm on the kv_lora part.
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_norm: QRmsNorm,
    // Latent K/V up-projections, per head, dequantized to f32 at load.
    // k_b: [n_head, kv_lora, qk_nope]   (k_nope = ckv @ k_b[h])
    // v_b_t: [n_head, kv_lora, v_head]  (v      = ckv @ v_b_t[h])
    k_b: Tensor,
    v_b_t: Tensor,
    o_proj: Arc<dyn QuantMethod>,

    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: MoeOrMlp,

    rotary: Arc<RotaryEmbedding>,
    n_head: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    #[allow(dead_code)]
    v_head_dim: usize,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;

        // --- Query: x -> q_a -> norm -> q_b -> [b, n_head, s, q_head_dim] ---
        let q = self.q_b_proj.forward(&self.q_a_norm.forward(&self.q_a_proj.forward(x)?)?)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head, q_head_dim))?
            .transpose(1, 2)?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // --- Compressed KV: x -> kv_a_mqa -> split(kv_lora, rope) ---
        let compressed = self.kv_a_proj_with_mqa.forward(x)?;
        let ckv = compressed.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe = compressed.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;
        let k_pe = k_pe
            .reshape((b_sz, seq_len, 1, self.qk_rope_head_dim))?
            .transpose(1, 2)?; // [b, 1, s, rope]
        let ckv = self.kv_a_norm.forward(&ckv)?; // [b, s, kv_lora]

        // --- RoPE (NeoX / rotate-half) on the rope slices only ---
        let (q_pe, k_pe) = self.rotary.forward(&q_pe.contiguous()?, &k_pe, start_offsets)?;

        // --- Reconstruct per-head K_nope and V from the latent (dense MLA) ---
        // ckv: [b, s, kv_lora] -> [b, 1, s, kv_lora] to broadcast against [1, n_head, kv_lora, *]
        let ckv_b = ckv.reshape((b_sz, 1, seq_len, self.kv_lora_rank))?;
        let k_b = self.k_b.to_dtype(ckv_b.dtype())?.unsqueeze(0)?; // [1, n_head, kv_lora, qk_nope]
        let v_b_t = self.v_b_t.to_dtype(ckv_b.dtype())?.unsqueeze(0)?; // [1, n_head, kv_lora, v_head]
        let k_nope = ckv_b.broadcast_matmul(&k_b)?; // [b, n_head, s, qk_nope]
        let v = ckv_b.broadcast_matmul(&v_b_t)?; // [b, n_head, s, v_head]

        // --- Assemble full Q, K (broadcast k_pe across heads) ---
        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?; // [b, n_head, s, q_head_dim]
        let k_pe_full = k_pe.broadcast_as((b_sz, self.n_head, seq_len, self.qk_rope_head_dim))?;
        let k = Tensor::cat(&[&k_nope, &k_pe_full], D::Minus1)?.contiguous()?; // [b, n_head, s, q_head_dim]

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        // Dense KV cache + full SDPA. v_head_dim (256) == q_head_dim (256) here, so the
        // standard NormalCache layout is fine.
        let (k, v) = kv_cache.append(&k, &v)?;
        let y = Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?;

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.o_proj.forward(&y.to_dtype(x.dtype())?)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

pub(crate) struct PropsGGUF {
    head_count: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    rope_dim: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    // MoE
    leading_dense_block_count: usize,
    expert_used_count: usize,
    expert_weights_norm: bool,
    expert_weights_scale: f64,
}

fn verify_glm_dsa_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "glm-dsa" {
        candle_core::bail!("Expected `glm-dsa` architecture, got `{arch}`.");
    }
    Ok(())
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let required = [
            "attention.head_count",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "attention.q_lora_rank",
            "attention.kv_lora_rank",
            "rope.dimension_count",
            "expert_count",
            "expert_used_count",
        ];
        c.has_required_keys(&required)?;

        // q_head_dim = key_length (576) is kv_lora + rope for the *compressed* path; the
        // *decompressed* per-head q/k dim is qk_nope + qk_rope. From the GGUF:
        //   key_length_mla = 256 = qk_nope + qk_rope (per-head decompressed)
        //   rope.dimension_count = qk_rope_head_dim
        //   value_length_mla = 256 = v_head_dim
        let rope_dim = c.get_value::<u32>("rope.dimension_count")? as usize;
        let per_head_qk = c
            .get_value::<u32>("attention.key_length_mla")
            .map(|x| x as usize)
            .unwrap_or(256);
        let v_head_dim = c
            .get_value::<u32>("attention.value_length_mla")
            .map(|x| x as usize)
            .unwrap_or(256);

        Ok(Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            rope_dim,
            q_lora_rank: c.get_value::<u32>("attention.q_lora_rank")? as usize,
            kv_lora_rank: c.get_value::<u32>("attention.kv_lora_rank")? as usize,
            qk_rope_head_dim: rope_dim,
            qk_nope_head_dim: per_head_qk - rope_dim,
            v_head_dim,
            leading_dense_block_count: c
                .get_value::<u32>("leading_dense_block_count")
                .map(|x| x as usize)
                .unwrap_or(0),
            expert_used_count: c.get_value::<u32>("expert_used_count")? as usize,
            expert_weights_norm: c.get_value::<bool>("expert_weights_norm").unwrap_or(true),
            expert_weights_scale: c
                .get_value::<f32>("expert_weights_scale")
                .map(|x| x as f64)
                .unwrap_or(1.0),
        })
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        verify_glm_dsa_arch(meta)?;

        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            candle_core::bail!(
                "glm-dsa GGUF currently supports only dense (Eager) attention; PagedAttention/MLA is not implemented."
            );
        }

        let metadata = ContentMetadata {
            path_prefix: "glm-dsa",
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            rope_dim,
            q_lora_rank: _q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            leading_dense_block_count,
            expert_used_count,
            expert_weights_norm,
            expert_weights_scale,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        // RoPE: NeoX (rotate-half) — llama.cpp evaluates deepseek2/glm with
        // LLAMA_ROPE_TYPE_NEOX and the GGUF weights are not pre-permuted.
        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    rope_dim,
                    max_seq_len,
                    device,
                    /* is_gpt_neox = */ true,
                    DType::F32,
                )?),
            );
        }

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            // --- Attention (MLA) ---
            let q_a_proj = gguf_linear(ct.tensor(&format!("{prefix}.attn_q_a.weight"), device)?)?;
            let q_a_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_a_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let q_b_proj = gguf_linear(ct.tensor(&format!("{prefix}.attn_q_b.weight"), device)?)?;
            let kv_a_proj_with_mqa =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_kv_a_mqa.weight"), device)?)?;
            let kv_a_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_kv_a_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            // Latent up-projections. Dequantize to f32 (Q8_0, small) and normalize layout to
            // [n_head, kv_lora, *] so K_nope/V = ckv @ W per head.
            //   k_b GGUF candle dims = [n_head, kv_lora, qk_nope] (already [h, kv_lora, qk_nope]).
            //   v_b GGUF candle dims = [n_head, v_head, kv_lora]  -> transpose last two to [h, kv_lora, v_head].
            let k_b = ct
                .tensor(&format!("{prefix}.attn_k_b.weight"), device)?
                .dequantize(device)?;
            let v_b = ct
                .tensor(&format!("{prefix}.attn_v_b.weight"), device)?
                .dequantize(device)?;
            // Validate the assumed layout against the metadata-derived dims.
            {
                let kd = k_b.dims();
                if kd != [head_count, kv_lora_rank, qk_nope_head_dim] {
                    candle_core::bail!(
                        "attn_k_b dims {kd:?} != expected [{head_count}, {kv_lora_rank}, {qk_nope_head_dim}]"
                    );
                }
                let vd = v_b.dims();
                if vd != [head_count, v_head_dim, kv_lora_rank] {
                    candle_core::bail!(
                        "attn_v_b dims {vd:?} != expected [{head_count}, {v_head_dim}, {kv_lora_rank}]"
                    );
                }
            }
            let v_b_t = v_b.transpose(1, 2)?.contiguous()?; // [n_head, kv_lora, v_head]

            let o_proj =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;

            // --- FFN: leading dense layers vs MoE layers ---
            let mlp = if layer_idx < leading_dense_block_count {
                MoeOrMlp::Mlp(Mlp {
                    gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                    up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
                    down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
                })
            } else {
                let gate_inp = ct
                    .tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let e_score_correction_bias = ct
                    .tensor(&format!("{prefix}.exp_probs_b.bias"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let gate = MoeGate {
                    weight: gate_inp,
                    e_score_correction_bias,
                    top_k: expert_used_count,
                    weights_norm: expert_weights_norm,
                    weights_scale: expert_weights_scale,
                };
                let gate_experts =
                    gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?)?;
                let up_experts =
                    gguf_linear(ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?)?;
                let down_experts =
                    gguf_linear(ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?)?;
                let shared = Mlp {
                    gate: gguf_linear(
                        ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?,
                    )?,
                    up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?)?,
                    down: gguf_linear(
                        ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?,
                    )?,
                };
                MoeOrMlp::Moe(Box::new(Moe {
                    gate,
                    gate_experts,
                    up_experts,
                    down_experts,
                    shared,
                }))
            };

            let attn_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let ffn_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            // NOTE: `blk.N.indexer.*` (DSA lightning indexer) and `blk.{last}.nextn.*`
            // (MTP head) tensors are intentionally NOT read — dense attention does not use
            // them. See module docs / the coverage report.

            layers.push(LayerWeights {
                q_a_proj,
                q_a_norm,
                q_b_proj,
                kv_a_proj_with_mqa,
                kv_a_norm,
                k_b,
                v_b_t,
                o_proj,
                attn_norm,
                ffn_norm,
                mlp,
                rotary: rotary.clone(),
                n_head: head_count,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                sdpa_params: SdpaParams {
                    n_kv_groups: 1,
                    softcap: None,
                    softmax_scale: 1.0 / (q_head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: gguf_linear(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let x = layer_in;
            let residual = &x;
            let x_normed = layer.attn_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x_normed,
                &mask.get(x_normed.device()),
                start_offsets,
                &mut cache[i],
            )?;
            let x = (attn + residual)?;

            let residual = &x;
            let x_normed = layer.ffn_norm.forward(&x)?;
            let x = (layer.mlp.forward(&x_normed)? + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}

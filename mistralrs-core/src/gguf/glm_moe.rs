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
//! Tensor parallelism: under a multi-rank communicator this loads *sharded*, since the whole
//! model does not fit one device. Three axes are split, all of them dim-0 slices — contiguous
//! byte ranges in GGUF, so they need no dequantization or quant-block re-alignment (see
//! [`Content::tensor_dim0_shard`]), and each rank reads only its own slice off disk:
//!   - routed experts, by expert index;
//!   - attention, by head (`attn_q_b`/`attn_k_b`/`attn_v_b`), which also narrows the KV cache
//!     to this rank's heads;
//!   - `o_proj`, by output feature.
//!
//! The shared/dense MLPs stay replicated. Because attention and `o_proj` are split along
//! *different* axes, `forward_attn` reduces the per-head outputs together before applying
//! `o_proj` and then reduces its output shards — see [`all_gather_by_sum`], which uses the
//! sum-only ring collective as an all-gather. The routed MoE output is reduced the same way.
//!
//! Structure mirrors `models::quantized_qwen3_moe` (GGUF reading idioms, `GgufMatMul`,
//! the block loop, KV cache) and `models::deepseek3` (the MLA attention forward and the
//! sigmoid/bias/group MoE gate). Routed experts are stored in small groups (see
//! [`expert_group`]) and dispatched by bucketing tokens per group on the host, rather than as
//! one stacked tensor per projection. That is deliberate: i-quant expert dtypes (IQ1_S /
//! IQ1_M / IQ2_XXS / IQ3_XXS / IQ4_XS) have no fused indexed-MoE kernel, so
//! `QuantMethod::gather_forward` falls back to dequantizing whichever tensor it is handed —
//! for one stacked shard that is the entire local expert stack (gigabytes of f32 per call).

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::CausalMaskConfig;
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
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, SumAllReduce};

// Default fallback for models that don't specify context_length.
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

/// How many routed experts share one stacked tensor.
///
/// Two costs pull in opposite directions here. i-quant experts have no fused indexed-MoE
/// kernel, so a forward dequantizes whichever stacked tensor it touches — smaller groups
/// waste less dequantization work, since only the experts a token routed to get expanded.
/// But CUDA rounds allocations up to a ~2MB granularity, and one tensor per expert (~3MB
/// each here) wastes 25-40% of *resident* memory across the tens of thousands of tensors a
/// 256-expert model needs, which is tens of gigabytes and does not fit. Grouping keeps each
/// allocation comfortably above that granularity while bounding the transient dequantization
/// to this many experts.
const EXPERT_GROUP_DEFAULT: usize = 8;

/// [`EXPERT_GROUP_DEFAULT`], overridable with `MISTRALRS_EXPERT_GROUP` so the memory/compute
/// balance can be retuned for a given box without a rebuild.
fn expert_group() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("MISTRALRS_EXPERT_GROUP")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|x| *x > 0)
            .unwrap_or(EXPERT_GROUP_DEFAULT)
    })
}

/// Tensor-parallel split of `total` items over `world_size` ranks; returns this rank's
/// `(offset, len)`. Shards are contiguous and balanced within one item, so the split does
/// not require `world_size` to divide `total`. A single rank owns everything.
fn tp_split(total: usize, rank: usize, world_size: usize, what: &str) -> Result<(usize, usize)> {
    if world_size <= 1 {
        return Ok((0, total));
    }
    // Every rank must own at least one unit: a zero-width shard would leave a rank
    // with no heads (or no experts) and empty tensors to slice.
    if total < world_size {
        candle_core::bail!(
            "{what} ({total}) is smaller than the tensor-parallel world size {world_size}; each rank needs at least one"
        );
    }
    // Uneven splits are allowed: the first `total % world_size` ranks take one extra
    // unit. Attention head counts and expert counts are rarely divisible by 3, and
    // rejecting that would cap tensor parallelism at power-of-two node counts.
    let base = total / world_size;
    let remainder = total % world_size;
    let len = base + usize::from(rank < remainder);
    let offset = rank * base + remainder.min(rank);
    Ok((offset, len))
}

/// Rebuild a full `[batch, seq, total]` tensor from per-rank column shards: pad this rank's
/// slice into its place with zeros and sum across the group. Every column is produced by
/// exactly one rank, so the sum is exactly the concatenation of the shards — which lets the
/// sum-only ring collective stand in for an all-gather.
fn all_gather_by_sum(
    all_reduce: &SumAllReduce,
    shard: Tensor,
    offset: usize,
    total: usize,
) -> Result<Tensor> {
    let (b_sz, seq_len, len) = shard.dims3()?;
    if len == total {
        return Ok(shard);
    }
    let (dtype, device) = (shard.dtype(), shard.device().clone());
    let mut parts = Vec::with_capacity(3);
    if offset > 0 {
        parts.push(Tensor::zeros((b_sz, seq_len, offset), dtype, &device)?);
    }
    parts.push(shard);
    let trailing = total - offset - len;
    if trailing > 0 {
        parts.push(Tensor::zeros((b_sz, seq_len, trailing), dtype, &device)?);
    }
    all_reduce.sum_all_reduce(&Tensor::cat(&parts, D::Minus1)?.contiguous()?)
}

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
        let scores_for_choice =
            scores.broadcast_add(&self.e_score_correction_bias.unsqueeze(0)?)?;
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
/// One group of routed experts: `[group_len, out, in]` per projection.
struct ExpertGroup {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

struct Moe {
    gate: MoeGate,
    /// The routed experts this rank owns, in groups of [`expert_group`] rather than one
    /// stacked tensor: the i-quant fallback dequantizes whatever tensor it is handed, and
    /// the whole local stack would be ~6.4GB of f32 per call.
    expert_groups: Vec<ExpertGroup>,
    shared: Mlp,
    // Expert-parallel tensor parallelism: this rank owns routed experts
    // `[experts_offset, experts_offset + n_local_experts)`. When `world_size == 1` it owns
    // them all and `all_reduce` is a no-op.
    all_reduce: SumAllReduce,
    world_size: usize,
    experts_offset: usize,
    n_local_experts: usize,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden) = xs.dims3()?;
        let identity = xs.clone();

        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        let num_tokens = bs * seq_len;
        let xs_flat = xs.reshape((num_tokens, hidden))?;

        // Resolve the routing on the host and bucket the (token, weight) pairs by the local
        // expert that owns them. Experts that live on another rank are simply skipped: each
        // expert is resident on exactly one rank, so the all-reduce below sums this rank's
        // partial contribution with the peers' to recover the full routed output.
        let selected = topk_idx.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let scores = topk_weight.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        // (token, index within the group, routing weight), bucketed per expert group.
        let mut buckets: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); self.expert_groups.len()];
        for (token, (token_experts, token_scores)) in selected.iter().zip(scores.iter()).enumerate()
        {
            for (expert, score) in token_experts.iter().zip(token_scores.iter()) {
                let expert = *expert as usize;
                if expert < self.experts_offset
                    || expert >= self.experts_offset + self.n_local_experts
                {
                    continue;
                }
                let local = expert - self.experts_offset;
                buckets[local / expert_group()].push((
                    token as u32,
                    (local % expert_group()) as u32,
                    *score,
                ));
            }
        }

        // Touch only the groups something actually routed to, each over just its own tokens.
        let mut routed = Tensor::zeros((num_tokens, hidden), xs.dtype(), xs.device())?;
        for (group_idx, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let group = &self.expert_groups[group_idx];
            let rows = Tensor::from_iter(bucket.iter().map(|(token, _, _)| *token), xs.device())?;
            let ids = Tensor::from_iter(bucket.iter().map(|(_, expert, _)| *expert), xs.device())?
                .reshape((bucket.len(), 1))?;
            let weights =
                Tensor::from_iter(bucket.iter().map(|(_, _, score)| *score), xs.device())?
                    .reshape((bucket.len(), 1))?;
            // `gather_forward` takes [n_rows, experts_per_row, cols] activations against
            // [n_rows, experts_per_row] ids; one expert per row here.
            let group_in = xs_flat.index_select(&rows, 0)?.unsqueeze(1)?;
            let gate_out = group.gate.gather_forward(&group_in, &ids)?;
            let up_out = group.up.gather_forward(&group_in, &ids)?;
            let act = crate::ops::mul_and_act(&gate_out, &up_out, crate::layers::Activation::Silu)?;
            let group_out = group.down.gather_forward(&act, &ids)?.squeeze(1)?;
            let group_out = group_out.broadcast_mul(&weights.to_dtype(group_out.dtype())?)?;
            routed = routed.index_add(&rows, &group_out.to_dtype(routed.dtype())?, 0)?;
        }
        let mut routed = routed.reshape((bs, seq_len, hidden))?;

        // Sum the per-rank partial routed contributions across the tensor-parallel group.
        if self.world_size > 1 {
            routed = self.all_reduce.sum_all_reduce(&routed.contiguous()?)?;
        }

        // The shared expert is replicated on every rank (added after the all-reduce so it
        // is counted once, not once per rank).
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
    // Latent K/V up-projections, for this rank's heads, dequantized to the model dtype.
    // k_b: [n_head_local, kv_lora, qk_nope]   (k_nope = ckv @ k_b[h])
    // v_b_t: [n_head_local, kv_lora, v_head]  (v      = ckv @ v_b_t[h])
    k_b: Tensor,
    v_b_t: Tensor,
    // Column-parallel: holds output features [o_out_offset, o_out_offset + o_out_len) of
    // o_out_total, reduced back to the full hidden size in `forward_attn`.
    o_proj: Arc<dyn QuantMethod>,
    o_all_reduce: SumAllReduce,
    o_world_size: usize,
    o_out_offset: usize,
    o_out_total: usize,

    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: MoeOrMlp,

    rotary: Arc<RotaryEmbedding>,
    /// Global attention head count; `n_head_local` of them live on this rank, starting at
    /// `head_offset`. Head-parallel attention halves both the Q/K/V up-projections and the
    /// KV cache, at the cost of one extra reduction to reassemble the per-head outputs
    /// before `o_proj` (which is sharded along a different axis and needs all heads).
    n_head: usize,
    n_head_local: usize,
    head_offset: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
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
        let q = self
            .q_b_proj
            .forward(&self.q_a_norm.forward(&self.q_a_proj.forward(x)?)?)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head_local, q_head_dim))?
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
        // candle 0.11 / mistral v0.9.0: RotaryEmbedding::forward takes positions
        // as a &Tensor; build it from the per-sequence start offsets.
        let positions = crate::pipeline::text_positions_tensor(start_offsets, seq_len, x.device())?;
        let (q_pe, k_pe) = self
            .rotary
            .forward(&q_pe.contiguous()?, &k_pe, &positions)?;

        // --- Reconstruct per-head K_nope and V from the latent (dense MLA) ---
        // ckv: [b, s, kv_lora] -> [b, 1, s, kv_lora] to broadcast against [1, n_head, kv_lora, *]
        let ckv_b = ckv.reshape((b_sz, 1, seq_len, self.kv_lora_rank))?;
        let k_b = self.k_b.to_dtype(ckv_b.dtype())?.unsqueeze(0)?; // [1, n_head, kv_lora, qk_nope]
        let v_b_t = self.v_b_t.to_dtype(ckv_b.dtype())?.unsqueeze(0)?; // [1, n_head, kv_lora, v_head]
        let k_nope = ckv_b.broadcast_matmul(&k_b)?; // [b, n_head, s, qk_nope]
        let v = ckv_b.broadcast_matmul(&v_b_t)?; // [b, n_head, s, v_head]

        // --- Assemble full Q, K (broadcast k_pe across heads) ---
        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?; // [b, n_head, s, q_head_dim]
        let k_pe_full =
            k_pe.broadcast_as((b_sz, self.n_head_local, seq_len, self.qk_rope_head_dim))?;
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

        let y = if !matches!(mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        // `o_proj` is sharded by output feature, so it needs *every* head's slice of the
        // attention output — reassemble those first. (Sharding both by head and by output
        // feature would make each rank hold a partial product of two different axes, which
        // no single sum can reconstruct, hence the separate reduction here.)
        let y = if self.o_world_size > 1 {
            all_gather_by_sum(
                &self.o_all_reduce,
                y,
                self.head_offset * self.v_head_dim,
                self.n_head * self.v_head_dim,
            )?
        } else {
            y
        };

        let y = self.o_proj.forward(&y.to_dtype(x.dtype())?)?;

        // Then rebuild the full hidden vector from the column-parallel o_proj shards.
        if self.o_world_size > 1 {
            return all_gather_by_sum(&self.o_all_reduce, y, self.o_out_offset, self.o_out_total);
        }
        Ok(y)
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

        // Dequantize to the model dtype rather than leaving the f32 that `dequantize`
        // returns: this matrix is vocab x hidden (~0.95B params for GLM-5.2), so f32 costs
        // ~3.5GB of device memory against ~1.8GB at bf16, and the embedding output is cast
        // to `dtype` on the very first use anyway.
        let tok_embeddings = ct
            .tensor("token_embd.weight", device)?
            .dequantize(device)?
            .to_dtype(dtype)?;
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

            // Per-layer tensor-parallel communicator. Under ring/NCCL distributed this
            // reports the real `world_size`; otherwise it is a single-rank no-op.
            let comm = mapper.get_comm_for(layer_idx)?;
            let rank = comm.rank();
            let world_size = comm.world_size();

            // --- Attention (MLA) ---
            let q_a_proj = gguf_linear(ct.tensor(&format!("{prefix}.attn_q_a.weight"), device)?)?;
            let q_a_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_a_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            // Head-parallel attention: this rank owns heads
            // [head_offset, head_offset + n_head_local). `attn_q_b` has candle dims
            // [n_head * q_head_dim, q_lora_rank] with the head axis outermost, so a head range
            // is a contiguous dim-0 slice; `attn_k_b`/`attn_v_b` are indexed by head directly.
            let (head_offset, n_head_local) =
                tp_split(head_count, rank, world_size, "glm-dsa attention head count")?;
            let q_b_proj = gguf_linear(ct.tensor_dim0_shard(
                &format!("{prefix}.attn_q_b.weight"),
                head_offset * q_head_dim,
                n_head_local * q_head_dim,
                device,
            )?)?;
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
            // Kept at the model dtype, not the f32 `dequantize` returns: across 79 layers
            // these two are ~1.16B params, i.e. ~4.6GB at f32 versus ~2.3GB at bf16. The
            // forward previously cast them to the activation dtype on every call, so
            // storing them cast also removes that per-forward conversion.
            let k_b = ct
                .tensor(&format!("{prefix}.attn_k_b.weight"), device)?
                .dequantize(device)?
                .to_dtype(dtype)?;
            let v_b = ct
                .tensor(&format!("{prefix}.attn_v_b.weight"), device)?
                .dequantize(device)?
                .to_dtype(dtype)?;
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
            // Keep only this rank's heads (narrowing after dequantize: the transient is the
            // full tensor, but only the shard stays resident).
            let k_b = k_b.narrow(0, head_offset, n_head_local)?.contiguous()?;
            let v_b = v_b.narrow(0, head_offset, n_head_local)?;
            let v_b_t = v_b.transpose(1, 2)?.contiguous()?; // [n_head_local, kv_lora, v_head]

            // Column-parallel o_proj. Candle dims are [hidden_out, n_head * v_head_dim], so
            // dim 0 is the output-feature axis: each rank owns a contiguous slice of the
            // output hidden features of what is the largest attention tensor (~5.1GB summed
            // over layers). Its input axis is every head's output, which head-parallel
            // attention leaves split across ranks, so `forward_attn` reduces those together
            // first and then zero-pads and sums these output shards.
            let o_proj_name = format!("{prefix}.attn_output.weight");
            let o_out_total = ct.tensor_info(&o_proj_name)?.shape.dims()[0];
            let (o_out_offset, o_out_len) =
                tp_split(o_out_total, rank, world_size, "attn_output output features")?;
            let o_proj = gguf_linear(ct.tensor_dim0_shard(
                &o_proj_name,
                o_out_offset,
                o_out_len,
                device,
            )?)?;

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
                // Expert-parallel: each rank loads only its slice of the routed experts
                // (`n_experts / world_size`), reading just those bytes off disk so the
                // full expert stack is never materialized on any single device.
                let gate_exps_name = format!("{prefix}.ffn_gate_exps.weight");
                let up_exps_name = format!("{prefix}.ffn_up_exps.weight");
                let down_exps_name = format!("{prefix}.ffn_down_exps.weight");
                let n_experts = ct.tensor_info(&gate_exps_name)?.shape.dims()[0];
                let (experts_offset, n_local_experts) =
                    tp_split(n_experts, rank, world_size, "glm-dsa expert count")?;
                let mut expert_groups =
                    Vec::with_capacity(n_local_experts.div_ceil(expert_group()));
                let mut group_start = experts_offset;
                while group_start < experts_offset + n_local_experts {
                    let len = expert_group().min(experts_offset + n_local_experts - group_start);
                    expert_groups.push(ExpertGroup {
                        gate: gguf_linear(ct.tensor_dim0_shard(
                            &gate_exps_name,
                            group_start,
                            len,
                            device,
                        )?)?,
                        up: gguf_linear(ct.tensor_dim0_shard(
                            &up_exps_name,
                            group_start,
                            len,
                            device,
                        )?)?,
                        down: gguf_linear(ct.tensor_dim0_shard(
                            &down_exps_name,
                            group_start,
                            len,
                            device,
                        )?)?,
                    });
                    group_start += len;
                }
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
                    expert_groups,
                    shared,
                    all_reduce: SumAllReduce::new(&comm),
                    world_size,
                    experts_offset,
                    n_local_experts,
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
                o_all_reduce: SumAllReduce::new(&comm),
                o_world_size: world_size,
                o_out_offset,
                o_out_total,
                attn_norm,
                ffn_norm,
                mlp,
                rotary: rotary.clone(),
                n_head: head_count,
                n_head_local,
                head_offset,
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
        // The embedding table is stored at `dtype` (bf16) to halve its footprint, but the
        // residual stream is f32 — every `QRmsNorm` weight is a dequantized f32 tensor, and
        // `rms_norm` is an elementwise op2 that requires both operands to share a dtype. So
        // bring the looked-up rows back to f32 here. This gives up no memory: the saving is
        // in the table itself, whereas this activation is only [batch, seq, hidden].
        let mut layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
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

#[cfg(test)]
mod tests {
    use super::tp_split;

    /// Every rank's shard must be contiguous, non-empty, and the shards together must
    /// tile `total` exactly — including when `total` is not divisible by `world_size`.
    #[test]
    fn tp_split_tiles_total_for_any_world_size() {
        for world_size in 1..=8 {
            for total in world_size..=64 {
                let mut next_offset = 0;
                for rank in 0..world_size {
                    let (offset, len) = tp_split(total, rank, world_size, "test")
                        .expect("a total at least as large as the world size splits");
                    assert_eq!(
                        offset, next_offset,
                        "total {total}, world {world_size}, rank {rank}"
                    );
                    assert!(len > 0, "total {total}, world {world_size}, rank {rank}");
                    next_offset = offset + len;
                }
                assert_eq!(
                    next_offset, total,
                    "shards do not cover total {total} at world size {world_size}"
                );
            }
        }
    }

    /// Shard widths differ by at most one, so no rank does materially more work.
    #[test]
    fn tp_split_is_balanced_within_one() {
        // 64 attention heads / 256 experts over 3 ranks is the case that used to be rejected.
        for total in [64usize, 96, 256, 78] {
            for world_size in 2..=8 {
                let lens: Vec<usize> = (0..world_size)
                    .map(|rank| {
                        tp_split(total, rank, world_size, "test")
                            .expect("total exceeds the world size")
                            .1
                    })
                    .collect();
                let (min, max) = (
                    lens.iter().copied().min().unwrap_or_default(),
                    lens.iter().copied().max().unwrap_or_default(),
                );
                assert!(
                    max - min <= 1,
                    "total {total} over {world_size} ranks: {lens:?}"
                );
                assert_eq!(lens.iter().sum::<usize>(), total);
            }
        }
    }

    /// A world size wider than the thing being split would leave a rank empty, which the
    /// loader cannot represent — that must be an error, not a zero-width tensor slice.
    #[test]
    fn tp_split_rejects_a_world_wider_than_the_split() {
        assert!(tp_split(2, 0, 3, "expert count").is_err());
        // Single-rank always owns everything, whatever the total.
        assert_eq!(
            tp_split(2, 0, 1, "expert count").expect("single rank is always valid"),
            (0, 2)
        );
    }
}

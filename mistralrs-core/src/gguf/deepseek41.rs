#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSeek-V4.1 pieces that are not DeepSeek-V3.
//!
//! The lagged hyper-connection block, Engram, compressed-KV pooling, and the
//! sqrt-softplus router live here. `deepseek41_model` is what loads a GGUF and
//! calls them. This is not the DeepSeek-V3 graph.

#[cfg(test)]
use std::collections::HashSet;

use candle_core::{DType, Result, Tensor, D};

use crate::ops::TopKLastDimOp;

use super::dsv4_hc::{combine_streams, mix_streams, sinkhorn};

/// Tensor suffixes a `deepseek41` GGUF has to contain.
#[cfg(test)]
pub(crate) const REQUIRED_TENSORS: &[&str] = &[
    "attn_compressor_gate.weight",
    "attn_compressor_kv.weight",
    "attn_compressor_norm.weight",
    "attn_kv.weight",
    "attn_kv_a_norm.weight",
    "attn_norm.weight",
    "attn_output_a.weight",
    "attn_output_b.weight",
    "attn_q_a.weight",
    "attn_q_a_norm.weight",
    "attn_q_b.weight",
    "attn_sinks.weight",
    "engram_embd.weight",
    "engram_k.weight",
    "engram_q.weight",
    "engram_wkv.weight",
    "exp_probs_b.bias",
    "exp_probs_b_vl.bias",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_shexp.weight",
    "ffn_norm.weight",
    "ffn_up_exps.weight",
    "ffn_up_shexp.weight",
    "hc_attn_base.weight",
    "hc_attn_fn.weight",
    "hc_attn_scale.weight",
    "hc_ffn_base.weight",
    "hc_ffn_fn.weight",
    "hc_ffn_scale.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
    "indexer.k_norm.weight",
    "indexer.proj.weight",
    "output.weight",
    "output_norm.weight",
    "token_embd.weight",
];

#[cfg(test)]
pub(crate) fn missing_tensors<'a>(present: &HashSet<&str>) -> Vec<&'a str> {
    REQUIRED_TENSORS
        .iter()
        .copied()
        .filter(|name| !present.contains(name))
        .collect()
}

/// Row ids for one sequence. Layout is `[position, layer, column]` with
/// `column = (ngram_order - 1) * n_heads + head`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn engram_row_ids(
    tokens: &[u32],
    token_map: &[u32],
    pad_token: u32,
    n_layers: usize,
    n_heads: usize,
    max_ngram: usize,
    multipliers: &[u64],
    primes: &[u64],
    offsets: &[u64],
) -> Result<Vec<u64>> {
    let n_orders = max_ngram.saturating_sub(1);
    let n_cols = n_orders.saturating_mul(n_heads);
    if n_layers == 0 || n_heads == 0 || max_ngram < 2 {
        candle_core::bail!("Engram hash needs at least one layer, one head, and a 2-gram");
    }
    if multipliers.len() != n_layers * max_ngram
        || primes.len() != n_layers * n_orders * n_heads
        || offsets.len() != n_layers * n_cols
    {
        candle_core::bail!("Engram hash tables do not match the layer, head, and n-gram counts");
    }
    let pad = u64::from(
        *token_map
            .get(pad_token as usize)
            .ok_or_else(|| candle_core::Error::msg("Engram pad token is outside the token map"))?,
    );
    let mut ids = vec![0u64; tokens.len() * n_layers * n_cols];
    for (pos, _) in tokens.iter().enumerate() {
        let mut gram = vec![0u64; max_ngram];
        for (shift, gram_token) in gram.iter_mut().enumerate() {
            *gram_token = if pos < shift {
                pad
            } else {
                let token = tokens[pos - shift] as usize;
                u64::from(*token_map.get(token).ok_or_else(|| {
                    candle_core::Error::msg("Engram token is outside the token map")
                })?)
            };
        }
        for layer in 0..n_layers {
            let mut rolling = gram[0].wrapping_mul(multipliers[layer * max_ngram]);
            for order in 1..max_ngram {
                rolling ^= gram[order].wrapping_mul(multipliers[layer * max_ngram + order]);
                let prime_base = (layer * n_orders + (order - 1)) * n_heads;
                for head in 0..n_heads {
                    let prime = primes[prime_base + head];
                    if prime == 0 {
                        candle_core::bail!("Engram prime is zero");
                    }
                    let bucket = rolling.rem_euclid(prime);
                    let col = (order - 1) * n_heads + head;
                    let slot = (pos * n_layers + layer) * n_cols + col;
                    ids[slot] = bucket.wrapping_add(offsets[layer * n_cols + col]);
                }
            }
        }
    }
    Ok(ids)
}

pub(crate) fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // A warmup step can hand this a zero-length sequence. CUDA's reduction
    // divides by the output size and panics when that size is zero.
    if x.dims().contains(&0) {
        return Ok(x.clone());
    }
    let xf = x.to_dtype(DType::F32)?;
    let variance = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(variance + eps)?.sqrt()?)?;
    normed.broadcast_mul(weight)?.to_dtype(x.dtype())
}

/// Learned mix for one sublayer. `fn_weight` is `[hc * hidden, (2 + hc) * hc]`.
pub(crate) struct HyperWeights {
    pub fn_weight: Tensor,
    pub scale: Tensor,
    pub base: Tensor,
}

struct HyperCoeff {
    pre: Tensor,
    post: Tensor,
    comb: Tensor,
}

/// Orients a GGUF `hc_*_fn` mix as the `[in, out]` matrix [`hyper_coeff`] multiplies the
/// flattened streams by. GGUF stores it `[out, in]`, and the mix is `x @ W.T`.
pub(crate) fn hyper_mix_projection(fn_weight: Tensor, flat_in: usize) -> Result<Tensor> {
    if fn_weight.dim(1)? == flat_in {
        fn_weight.t()?.contiguous()
    } else {
        Ok(fn_weight)
    }
}

fn hyper_coeff(
    h: &Tensor,
    weights: &HyperWeights,
    hc: usize,
    iters: usize,
    rms_eps: f32,
    hc_eps: f32,
) -> Result<HyperCoeff> {
    let dims = h.dims();
    let hidden = *dims.last().unwrap_or(&0);
    let flat_in = hc * hidden;
    let batch = dims[..dims.len() - 2].iter().product::<usize>();
    let flat = h.reshape((batch, flat_in))?.to_dtype(DType::F32)?;
    let variance = flat.sqr()?.mean_keepdim(D::Minus1)?;
    let scale = (variance + f64::from(rms_eps))?.sqrt()?.recip()?;
    // `fn_weight` arrives as `[in, out]`, oriented once at load by `hyper_mix_projection`;
    // transposing it here would copy the whole matrix on every token.
    if weights.fn_weight.dim(0)? != flat_in {
        candle_core::bail!(
            "deepseek41 hyper-connection mix is {:?}, but the streams flatten to {flat_in} values",
            weights.fn_weight.dims()
        );
    }
    let weight = weights.fn_weight.to_dtype(DType::F32)?;
    let projected = flat.matmul(&weight)?;
    let mixes = projected.broadcast_mul(&scale)?;
    let scale_w = weights.scale.to_dtype(DType::F32)?;
    let base = weights.base.to_dtype(DType::F32)?;
    let pre = mixes.narrow(1, 0, hc)?;
    let pre = pre
        .broadcast_mul(&scale_w.get(0)?)?
        .broadcast_add(&base.narrow(0, 0, hc)?)?;
    let pre = (candle_nn::ops::sigmoid(&pre)? + f64::from(hc_eps))?;
    let post = mixes.narrow(1, hc, hc)?;
    let post = post
        .broadcast_mul(&scale_w.get(1)?)?
        .broadcast_add(&base.narrow(0, hc, hc)?)?;
    let post = (candle_nn::ops::sigmoid(&post)? * 2.0)?;
    // Flat index `dst + src * hc`: reshape as `[src, dst]`, then swap.
    let comb = mixes.narrow(1, 2 * hc, hc * hc)?.reshape((batch, hc, hc))?;
    let comb_base = base
        .narrow(0, 2 * hc, hc * hc)?
        .reshape((hc, hc))?
        .transpose(0, 1)?
        .contiguous()?;
    let comb = comb.transpose(1, 2)?.contiguous()?;
    let comb = comb
        .broadcast_mul(&scale_w.get(2)?)?
        .broadcast_add(&comb_base)?;
    let comb = sinkhorn(&comb, iters, hc_eps)?;
    let mut stream_dims = dims[..dims.len() - 2].to_vec();
    stream_dims.push(hc);
    let mut comb_dims = stream_dims.clone();
    comb_dims.push(hc);
    Ok(HyperCoeff {
        pre: pre.reshape(stream_dims.clone())?,
        post: post.reshape(stream_dims)?,
        comb: comb.reshape(comb_dims)?,
    })
}

/// One V4.1 block. Attention reads `incoming_pre` from the previous FFN.
/// The returned pre is this block's FFN pre, for the next block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lagged_block<A, F>(
    h: &Tensor,
    incoming_pre: &Tensor,
    attn_mix: &HyperWeights,
    ffn_mix: &HyperWeights,
    attn_norm: &Tensor,
    ffn_norm: &Tensor,
    hc: usize,
    iters: usize,
    rms_eps: f32,
    hc_eps: f32,
    mut attn: A,
    mut ffn: F,
) -> Result<(Tensor, Tensor)>
where
    A: FnMut(&Tensor) -> Result<Tensor>,
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let attn_coeff = hyper_coeff(h, attn_mix, hc, iters, rms_eps, hc_eps)?;
    let mut x = mix_streams(h, incoming_pre)?;
    x = rmsnorm(&x, attn_norm, f64::from(rms_eps))?;
    x = attn(&x)?;
    let h = combine_streams(&x, h, &attn_coeff.post, &attn_coeff.comb)?;
    let ffn_coeff = hyper_coeff(&h, ffn_mix, hc, iters, rms_eps, hc_eps)?;
    let mut x = mix_streams(&h, &attn_coeff.pre)?;
    x = rmsnorm(&x, ffn_norm, f64::from(rms_eps))?;
    x = ffn(&x)?;
    let h = combine_streams(&x, &h, &ffn_coeff.post, &ffn_coeff.comb)?;
    Ok((h, ffn_coeff.pre))
}

/// GLM-5.3 applies the mix it just computed. DeepSeek-V4.1 does not: its
/// attention reads the previous block's pre.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn unlagged_block<A, F>(
    h: &Tensor,
    attn_mix: &HyperWeights,
    ffn_mix: &HyperWeights,
    attn_norm: &Tensor,
    ffn_norm: &Tensor,
    hc: usize,
    iters: usize,
    rms_eps: f32,
    hc_eps: f32,
    mut attn: A,
    mut ffn: F,
) -> Result<Tensor>
where
    A: FnMut(&Tensor) -> Result<Tensor>,
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let attn_coeff = hyper_coeff(h, attn_mix, hc, iters, rms_eps, hc_eps)?;
    let mut x = mix_streams(h, &attn_coeff.pre)?;
    x = rmsnorm(&x, attn_norm, f64::from(rms_eps))?;
    x = attn(&x)?;
    let h = combine_streams(&x, h, &attn_coeff.post, &attn_coeff.comb)?;
    let ffn_coeff = hyper_coeff(&h, ffn_mix, hc, iters, rms_eps, hc_eps)?;
    let mut x = mix_streams(&h, &ffn_coeff.pre)?;
    x = rmsnorm(&x, ffn_norm, f64::from(rms_eps))?;
    x = ffn(&x)?;
    combine_streams(&x, &h, &ffn_coeff.post, &ffn_coeff.comb)
}

/// Gated write of gathered Engram rows into the residual streams.
/// `wkv` is `[cols * head_dim, hc * hidden + hidden]`. A zero row leaves `h` unchanged.
/// Orients a GGUF `engram_wkv` as the `[in_dim, out]` matrix [`engram_apply`] multiplies
/// the gathered rows by, transposing it when it is stored `[out, in_dim]`.
pub(crate) fn engram_projection(wkv: Tensor, in_dim: usize) -> Result<Tensor> {
    if wkv.dim(1)? == in_dim {
        wkv.t()?.contiguous()
    } else {
        Ok(wkv)
    }
}

pub(crate) fn engram_apply(
    h: &Tensor,
    rows: &Tensor,
    wkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let dims = h.dims().to_vec();
    let hidden = *dims.last().unwrap_or(&0);
    let hc = dims[dims.len() - 2];
    let lead: usize = dims[..dims.len() - 2].iter().product();
    let in_dim = *rows.dims().last().unwrap_or(&0);
    // `wkv` arrives as `[in_dim, out]`, oriented once at load by `engram_projection`;
    // transposing it here would copy the whole matrix on every token.
    if wkv.dim(0)? != in_dim {
        candle_core::bail!(
            "deepseek41 engram projection is {:?}, but the gathered rows have {in_dim} columns",
            wkv.dims()
        );
    }
    let wkv = wkv.to_dtype(DType::F32)?;
    let kv = rows
        .reshape((lead, in_dim))?
        .to_dtype(DType::F32)?
        .matmul(&wkv)?;
    let key = kv.narrow(1, 0, hc * hidden)?.reshape((lead, hc, hidden))?;
    let value = kv.narrow(1, hc * hidden, hidden)?;
    let hf = h.reshape((lead, hc, hidden))?.to_dtype(DType::F32)?;
    // GGUF stores these as `[hc, hidden]`, which broadcasts onto `hf`.
    // `[hidden, hc]` is the other orientation and is transposed into place.
    let mut qk = (q_weight.to_dtype(DType::F32)? * k_weight.to_dtype(DType::F32)?)?;
    if qk.dims() == [hidden, hc] {
        qk = qk.t()?.contiguous()?;
    }
    let h_var = hf.sqr()?.mean_keepdim(D::Minus1)?;
    let k_var = key.sqr()?.mean_keepdim(D::Minus1)?;
    let rstd = (h_var + eps)?.sqrt()?.recip()?;
    let rstd = rstd.broadcast_mul(&(k_var + eps)?.sqrt()?.recip()?)?;
    let dot = hf.broadcast_mul(&qk)?.broadcast_mul(&key)?.sum(D::Minus1)?;
    let dot = (dot * rstd.squeeze(D::Minus1)?)?;
    let dot = (dot / (hidden as f64).sqrt())?;
    let floor = (dot.ones_like()? * 1e-6)?;
    let magnitude = dot.abs()?.maximum(&floor)?.sqrt()?;
    let gate = candle_nn::ops::sigmoid(&(magnitude * dot.sign()?)?)?;
    let value = value.unsqueeze(D::Minus2)?;
    let updated = (hf + gate.unsqueeze(D::Minus1)?.broadcast_mul(&value)?)?;
    updated.reshape(dims)?.to_dtype(h.dtype())
}

/// Pool complete groups. `kv` and `score` are `[batch, seq, hidden]`.
/// A tail shorter than `ratio` is left for the next step and is not returned.
pub(crate) fn compress_groups(kv: &Tensor, score: &Tensor, ratio: usize) -> Result<Tensor> {
    if ratio == 0 {
        candle_core::bail!("compress ratio is zero");
    }
    let (batch, seq, hidden) = kv.dims3()?;
    let groups = seq / ratio;
    if groups == 0 {
        return kv.narrow(1, 0, 0);
    }
    let used = groups * ratio;
    let kv = kv
        .narrow(1, 0, used)?
        .reshape((batch, groups, ratio, hidden))?;
    let score = score
        .narrow(1, 0, used)?
        .reshape((batch, groups, ratio, hidden))?;
    let probs = candle_nn::ops::softmax(&score, 2)?;
    kv.broadcast_mul(&probs)?.sum(2)
}

fn sqrt_softplus(x: &Tensor) -> Result<Tensor> {
    let positive = x.maximum(&x.zeros_like()?)?;
    let stable = (x.abs()? * -1.0)?;
    let log_term = (stable.exp()? + 1.0)?.log()?;
    (positive + log_term)?.sqrt()
}

/// Sqrt-softplus router. `bias` changes which experts win and does not scale
/// the weights. Weights are renormalized and multiplied by `scale`.
pub(crate) fn moe_route(
    logits: &Tensor,
    bias: &Tensor,
    top_k: usize,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let scores = sqrt_softplus(logits)?;
    let choice = scores.broadcast_add(bias)?;
    let picked = choice.topk(top_k)?;
    let weights = scores.gather(&picked.indices, D::Minus1)?;
    let denom = (weights.sum_keepdim(D::Minus1)? + 1e-20)?;
    let weights = (weights.broadcast_div(&denom)? * scale)?;
    Ok((picked.indices, weights))
}

pub(crate) fn swiglu_clamped(gate: &Tensor, up: &Tensor, limit: f64) -> Result<Tensor> {
    let gate = gate.to_dtype(DType::F32)?;
    let up = up.to_dtype(DType::F32)?;
    let (gate, up) = if limit > 0.0 {
        (
            gate.clamp(f64::NEG_INFINITY, limit)?,
            up.clamp(-limit, limit)?,
        )
    } else {
        (gate, up)
    };
    candle_nn::ops::silu(&gate)? * up
}

/// Causal multi-query attention. `k` is the single KV head `[batch, seq, dim]`.
/// `sinks` is one logit per head that takes probability mass and adds nothing.
#[cfg(test)]
pub(crate) fn mqa_with_sink(q: &Tensor, k: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    let (batch, seq, heads, dim) = q.dims4()?;
    let scale = 1.0 / (dim as f64).sqrt();
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.unsqueeze(1)?.contiguous()?;
    let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
    let scores = (scores * scale)?;
    let key_index = Tensor::arange(0u32, seq as u32, q.device())?;
    let query_index = key_index.unsqueeze(D::Minus1)?;
    let causal = query_index.broadcast_ge(&key_index.unsqueeze(0)?)?;
    let blocked = (scores.zeros_like()? - f64::INFINITY)?;
    let scores = causal
        .broadcast_as(scores.shape())?
        .where_cond(&scores, &blocked)?;
    let row_max = scores.max_keepdim(D::Minus1)?;
    let exp_scores = (scores - &row_max)?.exp()?;
    let sink = (sinks.reshape((1, heads, 1, 1))? - &row_max)?.exp()?;
    let denom = (exp_scores.sum_keepdim(D::Minus1)? + sink)?;
    let probs = exp_scores.broadcast_div(&denom)?;
    let out = probs
        .contiguous()?
        .matmul(&k.reshape((batch, 1, seq, dim))?.contiguous()?)?;
    out.transpose(1, 2)?.contiguous()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;

    use candle_core::{DType, Device, Tensor};

    use crate::gguf::dsv4_hc::mix_streams;

    use super::{
        compress_groups, engram_apply, engram_projection, engram_row_ids, hyper_coeff,
        hyper_mix_projection, lagged_block, missing_tensors, moe_route, mqa_with_sink,
        swiglu_clamped, HyperWeights, REQUIRED_TENSORS,
    };

    #[test]
    fn hyper_coeff_takes_the_gguf_out_in_matrix_oriented_at_load() {
        let device = Device::Cpu;
        let hc = 2usize;
        let hidden = 5usize;
        let flat_in = hc * hidden;
        let out = (2 + hc) * hc;
        let values: Vec<f32> = (0..flat_in * out)
            .map(|index| ((index % 7) as f32) * 0.01)
            .collect();
        let inn = Tensor::from_vec(values.clone(), (flat_in, out), &device).expect("in-out");
        let out_in = inn.t().expect("transpose").contiguous().expect("row major");
        assert_eq!(out_in.dims(), &[out, flat_in]);
        let scale = Tensor::from_vec(vec![1f32, 1.0, 1.0], (3,), &device).expect("scale");
        let base = Tensor::zeros((out,), DType::F32, &device).expect("base");
        let h = Tensor::ones((1, 1, hc, hidden), DType::F32, &device).expect("hidden");
        let left = hyper_coeff(
            &h,
            &HyperWeights {
                fn_weight: inn,
                scale: scale.clone(),
                base: base.clone(),
            },
            hc,
            1,
            1e-5,
            1e-6,
        )
        .expect("in-out coeff");
        let right = hyper_coeff(
            &h,
            &HyperWeights {
                fn_weight: hyper_mix_projection(out_in.clone(), flat_in).expect("oriented"),
                scale: scale.clone(),
                base: base.clone(),
            },
            hc,
            1,
            1e-5,
            1e-6,
        )
        .expect("out-in coeff");
        assert_eq!(
            left.pre
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("pre"),
            right
                .pre
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("pre")
        );
        let unoriented = hyper_coeff(
            &h,
            &HyperWeights {
                fn_weight: out_in,
                scale,
                base,
            },
            hc,
            1,
            1e-5,
            1e-6,
        )
        .err()
        .expect("an unoriented mix is refused")
        .to_string();
        assert_eq!(
            unoriented,
            "deepseek41 hyper-connection mix is [8, 10], but the streams flatten to 10 values"
        );
    }

    fn close(lhs: &Tensor, rhs: &Tensor) {
        let lhs = lhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rhs = rhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(diff < 1e-4, "idx={idx} lhs={lhs} rhs={rhs} diff={diff}");
        }
    }

    #[test]
    fn every_listed_family_is_required_and_a_gap_is_named() {
        let present: HashSet<&str> = REQUIRED_TENSORS.iter().copied().collect();
        assert!(missing_tensors(&present).is_empty());
        let mut dropped = present;
        assert!(dropped.remove("engram_embd.weight"));
        assert_eq!(missing_tensors(&dropped), vec!["engram_embd.weight"]);
        assert!(dropped.remove("attn_compressor_kv.weight"));
        let missing = missing_tensors(&dropped);
        assert!(missing.contains(&"engram_embd.weight"));
        assert!(missing.contains(&"attn_compressor_kv.weight"));
    }

    #[test]
    fn engram_hash_reads_earlier_tokens_and_pads_the_start() {
        let token_map: Vec<u32> = (0..8).collect();
        let multipliers = [3u64, 5];
        let primes = [11u64, 13];
        let offsets = [100u64, 200];
        let first = engram_row_ids(
            &[4, 1],
            &token_map,
            0,
            1,
            2,
            2,
            &multipliers,
            &primes,
            &offsets,
        )
        .unwrap();
        let second = engram_row_ids(
            &[5, 1],
            &token_map,
            0,
            1,
            2,
            2,
            &multipliers,
            &primes,
            &offsets,
        )
        .unwrap();
        assert_eq!(first.len(), 4);
        assert_ne!(
            first[2..],
            second[2..],
            "position 1 must read the token before it"
        );
        let same_start = engram_row_ids(
            &[4],
            &token_map,
            0,
            1,
            2,
            2,
            &multipliers,
            &primes,
            &offsets,
        )
        .unwrap();
        assert_eq!(first[..2], same_start[..]);
    }

    #[test]
    fn rmsnorm_of_an_empty_sequence_does_not_reduce() {
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 0, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones((4,), DType::F32, &device).unwrap();
        let out = super::rmsnorm(&x, &weight, 1e-6).unwrap();
        assert_eq!(out.dims(), &[1, 0, 4]);
    }

    #[test]
    fn zero_engram_row_leaves_the_residual() {
        let device = Device::Cpu;
        let h = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device).unwrap();
        let rows = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
        let wkv = Tensor::from_vec(
            vec![
                0.5f32, -0.25, 0.125, 1.0, -1.0, 0.5, 0.25, 0.5, -0.125, 0.75, 0.25, -0.5,
            ],
            (2, 6),
            &device,
        )
        .unwrap();
        let q = Tensor::from_vec(vec![0.2f32, -0.3, 0.4, -0.1], (2, 2), &device).unwrap();
        let k = Tensor::from_vec(vec![0.5f32, 0.25, -0.5, 1.0], (2, 2), &device).unwrap();
        let out = engram_apply(&h, &rows, &wkv, &q, &k, 1e-6).unwrap();
        close(&out, &h);
        let rows = Tensor::from_vec(vec![1f32, -1.0], (1, 1, 2), &device).unwrap();
        let changed = engram_apply(&h, &rows, &wkv, &q, &k, 1e-6).unwrap();
        assert_ne!(
            changed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            h.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn an_engram_projection_stored_out_by_in_is_oriented_once_at_load() {
        let device = Device::Cpu;
        let h = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device).unwrap();
        let rows = Tensor::from_vec(vec![1f32, -1.0], (1, 1, 2), &device).unwrap();
        let in_by_out = Tensor::from_vec(
            vec![
                0.5f32, -0.25, 0.125, 1.0, -1.0, 0.5, 0.25, 0.5, -0.125, 0.75, 0.25, -0.5,
            ],
            (2, 6),
            &device,
        )
        .unwrap();
        let q = Tensor::from_vec(vec![0.2f32, -0.3, 0.4, -0.1], (2, 2), &device).unwrap();
        let k = Tensor::from_vec(vec![0.5f32, 0.25, -0.5, 1.0], (2, 2), &device).unwrap();
        // GGUF stores the projection `[out, in_dim]`.
        let stored = in_by_out.t().unwrap().contiguous().unwrap();

        let oriented = engram_projection(stored.clone(), 2).unwrap();
        assert_eq!(oriented.dims(), &[2, 6]);
        assert_eq!(
            engram_apply(&h, &rows, &oriented, &q, &k, 1e-6)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            engram_apply(&h, &rows, &in_by_out, &q, &k, 1e-6)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let unoriented = engram_apply(&h, &rows, &stored, &q, &k, 1e-6)
            .err()
            .expect("an unoriented projection is refused")
            .to_string();
        assert_eq!(
            unoriented,
            "deepseek41 engram projection is [6, 2], but the gathered rows have 2 columns"
        );
    }

    #[test]
    fn attention_reads_the_incoming_mix_not_its_own() {
        let device = Device::Cpu;
        let hc = 2usize;
        let hidden = 2usize;
        let h = Tensor::from_vec(
            vec![0.5f32, -0.25, 0.125, 0.75],
            (1, 1, hc, hidden),
            &device,
        )
        .unwrap();
        let incoming = Tensor::from_vec(vec![1f32, 0.0], (1, 1, hc), &device).unwrap();
        let mix_len = (2 + hc) * hc;
        let weight = Tensor::from_vec(
            vec![0.1f32; (hc * hidden) * mix_len],
            (hc * hidden, mix_len),
            &device,
        )
        .unwrap();
        let scale = Tensor::from_vec(vec![1f32, 1.0, 1.0], (3,), &device).unwrap();
        let base = Tensor::zeros((mix_len,), DType::F32, &device).unwrap();
        let attn_mix = HyperWeights {
            fn_weight: weight.clone(),
            scale: scale.clone(),
            base: base.clone(),
        };
        let ffn_mix = HyperWeights {
            fn_weight: weight,
            scale,
            base,
        };
        let norm = Tensor::ones((hidden,), DType::F32, &device).unwrap();
        let seen = RefCell::new(None);
        let attn = |x: &Tensor| {
            *seen.borrow_mut() = Some(x.clone());
            Ok(x.clone())
        };
        let ffn = |x: &Tensor| Ok(x.clone());
        let (_h, next_pre) = lagged_block(
            &h, &incoming, &attn_mix, &ffn_mix, &norm, &norm, hc, 1, 1e-6, 1e-6, attn, ffn,
        )
        .unwrap();
        let expected = mix_streams(&h, &incoming).unwrap();
        let expected = super::rmsnorm(&expected, &norm, 1e-6).unwrap();
        close(seen.borrow().as_ref().unwrap(), &expected);
        let own_pre_would_match_incoming =
            next_pre.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                == incoming.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(!own_pre_would_match_incoming);
    }

    #[test]
    fn swapping_a_group_does_not_change_its_pool_and_the_score_does() {
        let device = Device::Cpu;
        let kv = Tensor::from_vec(vec![1f32, 3.0, 10.0, 0.0], (1, 4, 1), &device).unwrap();
        let score = Tensor::from_vec(vec![0f32, 0.0, 2.0, -2.0], (1, 4, 1), &device).unwrap();
        let pooled = compress_groups(&kv, &score, 2).unwrap();
        let kv_swap = Tensor::from_vec(vec![3f32, 1.0, 0.0, 10.0], (1, 4, 1), &device).unwrap();
        let score_swap = Tensor::from_vec(vec![0f32, 0.0, -2.0, 2.0], (1, 4, 1), &device).unwrap();
        close(&pooled, &compress_groups(&kv_swap, &score_swap, 2).unwrap());
        let score_flip = Tensor::from_vec(vec![0f32, 0.0, -2.0, 2.0], (1, 4, 1), &device).unwrap();
        let flipped = compress_groups(&kv, &score_flip, 2).unwrap();
        assert_ne!(
            pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            flipped.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn router_bias_selects_the_expert_and_the_weight_stays_unbiased() {
        let device = Device::Cpu;
        let logits = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let bias = Tensor::from_vec(vec![0f32, 0.0, 5.0, 0.0], (4,), &device).unwrap();
        let (idx, weights) = moe_route(&logits, &bias, 1, 1.5).unwrap();
        assert_eq!(idx.to_vec2::<u32>().unwrap(), vec![vec![2]]);
        let weight = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((weight - 1.5).abs() < 1e-4, "{weight}");
    }

    #[test]
    fn a_large_sink_shrinks_the_attention_output() {
        let device = Device::Cpu;
        let q = Tensor::from_vec(vec![1f32, 0.0], (1, 1, 1, 2), &device).unwrap();
        let k = Tensor::from_vec(vec![1f32, 0.0], (1, 1, 2), &device).unwrap();
        let quiet = mqa_with_sink(
            &q,
            &k,
            &Tensor::from_vec(vec![-20f32], (1,), &device).unwrap(),
        )
        .unwrap();
        let loud = mqa_with_sink(
            &q,
            &k,
            &Tensor::from_vec(vec![20f32], (1,), &device).unwrap(),
        )
        .unwrap();
        let quiet = quiet
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let loud = loud
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(quiet > loud, "quiet {quiet} loud {loud}");
    }

    #[test]
    fn a_closed_swiglu_gate_is_near_zero() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![-40f32], (1,), &device).unwrap();
        let up = Tensor::from_vec(vec![3f32], (1,), &device).unwrap();
        let value = swiglu_clamped(&gate, &up, 10.0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(value.abs() < 1e-4, "{value}");
    }
}

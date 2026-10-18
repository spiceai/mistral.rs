//! Non-fused DeepSeek-V4 hyper-connection mixer.
//!
//! Streams are `[n_tokens, stream, hidden]`. The combine matrix is
//! `[n_tokens, dst, src]`. This is the sinkhorn mixer: a destination softmax,
//! epsilon, a source normalization, then `iters - 1` destination/source
//! normalizations. It is not the Qwen4-Exp low-rank gated residual.

use candle_core::{Result, Tensor, D};

fn normalize_dim(x: &Tensor, dim: D, eps: f32) -> Result<Tensor> {
    let denom = (x.sum_keepdim(dim)? + f64::from(eps))?;
    x.broadcast_div(&denom)
}

/// `logits` is `[n_tokens, dst, src]`. Softmax runs over destinations.
pub(crate) fn sinkhorn(logits: &Tensor, iters: usize, eps: f32) -> Result<Tensor> {
    let mut comb = candle_nn::ops::softmax(logits, D::Minus2)?;
    comb = (comb + f64::from(eps))?;
    comb = normalize_dim(&comb, D::Minus1, eps)?;
    for _ in 1..iters {
        comb = normalize_dim(&comb, D::Minus2, eps)?;
        comb = normalize_dim(&comb, D::Minus1, eps)?;
    }
    Ok(comb)
}

/// Weighted sum of streams. `x` is `[n_tokens, hc, hidden]`, `pre` is `[n_tokens, hc]`.
pub(crate) fn mix_streams(x: &Tensor, pre: &Tensor) -> Result<Tensor> {
    let weight = pre.unsqueeze(D::Minus1)?;
    x.broadcast_mul(&weight)?.sum(D::Minus2)
}

/// `x` is the block output `[n_tokens, hidden]`.
/// `residual` is `[n_tokens, src, hidden]`.
/// `post` is `[n_tokens, dst]`.
/// `comb` is `[n_tokens, dst, src]`.
/// Returns `[n_tokens, dst, hidden]`.
pub(crate) fn combine_streams(
    x: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    comb: &Tensor,
) -> Result<Tensor> {
    let injected = x.unsqueeze(D::Minus2)?.broadcast_mul(&post.unsqueeze(D::Minus1)?)?;
    let mixed = comb.matmul(&residual)?;
    injected.broadcast_add(&mixed)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{combine_streams, mix_streams, sinkhorn};

    fn close(got: &[f32], expect: &[f32]) {
        assert_eq!(got.len(), expect.len());
        for (g, e) in got.iter().zip(expect) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {g} expected {e} in {got:?} vs {expect:?}"
            );
        }
    }

    #[test]
    fn sinkhorn_one_iteration_matches_the_reference_matrix() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1f32, 0.0, 0.0, 3.0], (1, 2, 2), &device)
            .expect("logits");
        let got = sinkhorn(&logits, 1, 1e-6)
            .expect("sinkhorn")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values");
        close(
            &got,
            &[0.939_076_89, 0.060_921_82, 0.220_170_56, 0.779_828_62],
        );
    }

    #[test]
    fn sinkhorn_second_iteration_renormalizes_both_axes() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1f32, 0.0, 0.0, 3.0], (1, 2, 2), &device)
            .expect("logits");
        let got = sinkhorn(&logits, 2, 1e-6)
            .expect("sinkhorn")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values");
        close(
            &got,
            &[0.917_893_26, 0.082_105_61, 0.169_960_97, 0.830_038_13],
        );
    }

    #[test]
    fn mix_and_combine_follow_the_stream_formula() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0], (1, 2, 2), &device).expect("streams");
        let pre = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device).expect("pre");
        let mixed = mix_streams(&x, &pre)
            .expect("mix")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values");
        close(&mixed, &[0.25, 0.75]);

        let block = Tensor::from_vec(vec![10f32, 0.0], (1, 2), &device).expect("block");
        let post = Tensor::from_vec(vec![2f32, 0.0], (1, 2), &device).expect("post");
        let comb = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0], (1, 2, 2), &device).expect("comb");
        let out = combine_streams(&block, &x, &post, &comb)
            .expect("combine")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values");
        close(&out, &[21.0, 0.0, 0.0, 1.0]);
    }
}

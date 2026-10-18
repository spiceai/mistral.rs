//! GLM-5.3-Flash tensor families, checked by the tests below. `glm5next_model`
//! loads them and runs the forward.

#[cfg(test)]
use std::collections::HashSet;

/// Tensor suffixes a `glm5-next` GGUF has to contain. The last block is the
/// draft head and does not carry `hc_*`.
#[cfg(test)]
pub(crate) const REQUIRED_TENSORS: &[&str] = &[
    "attn_k.weight",
    "attn_k_b.weight",
    "attn_kv_a_mqa.weight",
    "attn_kv_a_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_a.weight",
    "attn_q_a_norm.weight",
    "attn_q_b.weight",
    "attn_v.weight",
    "attn_v_b.weight",
    "exp_probs_b.bias",
    "ffn_down.weight",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
    "ffn_gate.weight",
    "ffn_gate_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_shexp.weight",
    "ffn_norm.weight",
    "ffn_up.weight",
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
    "indexer.k_norm.bias",
    "indexer.k_norm.weight",
    "indexer.proj.weight",
    "indexer_compressor_ape.weight",
    "indexer_compressor_gate.weight",
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
    "output.weight",
    "output_norm.weight",
    "ssm_a",
    "ssm_beta.weight",
    "ssm_conv1d_k.weight",
    "ssm_conv1d_q.weight",
    "ssm_conv1d_v.weight",
    "ssm_dt.bias",
    "ssm_f_a.weight",
    "ssm_f_b.weight",
    "ssm_g_a.weight",
    "ssm_g_b.weight",
    "ssm_norm.weight",
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{missing_tensors, REQUIRED_TENSORS};

    #[test]
    fn every_listed_family_is_required_and_a_gap_is_named() {
        let present: HashSet<&str> = REQUIRED_TENSORS.iter().copied().collect();
        assert!(missing_tensors(&present).is_empty());
        let mut dropped = present;
        assert!(dropped.remove("ssm_a"));
        assert_eq!(missing_tensors(&dropped), vec!["ssm_a"]);
    }
}

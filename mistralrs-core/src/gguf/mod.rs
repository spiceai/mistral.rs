pub(crate) mod base_model;
mod deepseek41;
pub(crate) mod deepseek41_model;
mod dsv4_hc;
mod glm5next;
pub(crate) mod glm5next_model;
mod chat_template;
mod content;
pub(crate) mod gemma3_bindings;
pub(crate) mod gemma3_config;
pub(crate) mod gemma3n_bindings;
mod gguf_tokenizer;
pub(crate) mod glm_moe;
pub(crate) mod idefics3_bindings;
mod lfm2_vl_bindings;
pub(crate) mod llama4_bindings;
mod mistral3_bindings;
mod multimodal_binding_utils;
pub(crate) mod multimodal_bindings;
pub(crate) mod multimodal_vision_registry;
pub(crate) mod muse_glimmer_bindings;
pub(crate) mod normal_bindings;
pub(crate) mod normal_config;
pub(crate) mod normal_registry;
pub(crate) mod qwen4exp;
pub(crate) mod qwen_multimodal_bindings;
use strum::{EnumIter, EnumString, IntoEnumIterator};

use anyhow::{Context, Result};
pub(crate) use chat_template::{get_gguf_chat_template, get_gguf_chat_template_from_metadata};
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{
    convert_gguf_metadata_to_hf_tokenizer, validate_external_gguf_tokenizer,
    GgufTokenizerConversion,
};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";

#[derive(Debug, EnumString, EnumIter, Clone, Copy, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
    Starcoder2,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    Mistral3,
    #[strum(serialize = "glm-dsa")]
    GlmDsa,
    /// DeepSeek-V4.1. Not DeepSeek-V3 and not `deepseekv3`.
    DeepSeek41,
    /// GLM-5.3-Flash. Not GLM-4 and not `glm4`.
    #[strum(serialize = "glm5-next")]
    Glm5Next,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
/// Architectures whose GGUFs load far enough to be identified but have no graph
/// here yet. Naming them beats "Unknown GGUF architecture": the file is valid and
/// the operator's next step is a different engine or a different quant repo, not
/// a re-download. Each entry pairs the `general.architecture` value with the
/// model family and the components a loader still has to grow.
const RECOGNIZED_UNIMPLEMENTED: &[(&str, &str, &str)] = &[
    (
        "deepseek4",
        "DeepSeek-V4",
        "an MLA output LoRA (`o_lora_rank`/`o_groups`), a sqrt-softplus router, \
     hash layers, a sparse-attention indexer, and MXFP4 expert tensors",
    ),
];

impl GGUFArchitecture {
    /// `Some` when `general.architecture` names a checkpoint this build can
    /// identify and must not run on an older graph.
    pub(crate) fn unimplemented_reason(value: &str) -> Option<anyhow::Error> {
        let lowered = value.to_ascii_lowercase();
        let (_, family, missing) = RECOGNIZED_UNIMPLEMENTED
            .iter()
            .find(|(arch, _, _)| *arch == lowered)?;
        Some(anyhow::anyhow!(
            "GGUF architecture `{value}` ({family}) is recognized but not implemented: it needs {missing}. Supported architectures: {}.",
            Self::supported()
        ))
    }

    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        let lowered = value.as_ref().to_ascii_lowercase();
        if let Some(error) = Self::unimplemented_reason(value.as_ref()) {
            return Err(error);
        }
        Self::from_str(&lowered)
            .with_context(|| {
                format!(
                    "Unknown GGUF architecture `{value}`. Supported architectures: {}.",
                    Self::supported()
                )
            })
            .map_err(anyhow::Error::msg)
    }

    /// Whether this architecture's loader can be given a `PagedAttentionConfig`.
    ///
    /// Multi-head Latent Attention GGUFs are evaluated dense here — the loader reconstructs
    /// full per-head K/V from the compressed latent and caches that, and there is no paged
    /// kernel for it — so handing one a paged config fails the load outright. Callers should
    /// not have to know which architectures those are: the answer belongs with the loaders,
    /// and the match is exhaustive so a new architecture cannot be added without deciding.
    pub fn supports_paged_attention(self) -> bool {
        match self {
            // MLA + sigmoid-gated MoE, run dense; see `gguf::glm_moe`.
            Self::GlmDsa => false,
            // Compressed KV and a lagged hyper-connection. No paged kernel.
            Self::DeepSeek41 => false,
            // KDA recurrence and compressed DSA. No paged kernel.
            Self::Glm5Next => false,
            Self::Llama
            | Self::Mpt
            | Self::Gptneox
            | Self::Gptj
            | Self::Gpt2
            | Self::Bloom
            | Self::Falcon
            | Self::Mamba
            | Self::Rwkv
            | Self::Phi2
            | Self::Phi3
            | Self::Starcoder2
            | Self::Qwen2
            | Self::Qwen3
            | Self::Qwen3MoE
            | Self::Mistral3 => true,
        }
    }

    /// The `general.architecture` values this build can actually run, for error messages.
    fn supported() -> String {
        Self::iter()
            .map(|arch| arch.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::GGUFArchitecture;

    /// A DeepSeek-V4 GGUF is a valid file this build cannot run yet. The error has to say
    /// that, name the family, and list what *is* supported — "Unknown GGUF architecture"
    /// reads like a corrupt download and sends the operator to re-fetch 100+ GB.
    #[test]
    fn deepseek4_is_reported_as_recognized_but_unimplemented() {
        let err = GGUFArchitecture::from_value("deepseek4")
            .expect_err("deepseek4 has no loader in this build");
        let msg = err.to_string();
        assert!(msg.contains("DeepSeek-V4"), "{msg}");
        assert!(msg.contains("not implemented"), "{msg}");
        assert!(!msg.contains("Unknown"), "{msg}");
        // The supported list is what tells the operator where to go instead.
        assert!(msg.contains("glm-dsa"), "{msg}");
    }

    #[test]
    fn deepseek41_resolves_to_its_own_graph() {
        let arch = GGUFArchitecture::from_value("deepseek41").expect("deepseek41 loads");
        assert!(
            matches!(arch, GGUFArchitecture::DeepSeek41),
            "deepseek41 must not alias another graph: {arch}"
        );
        let name = arch.to_string();
        assert_eq!(name, "deepseek41");
        assert!(!name.contains("deepseekv3"), "{name}");
        assert!(!arch.supports_paged_attention());
        assert!(GGUFArchitecture::from_str("deepseek41").is_ok());
        assert!(GGUFArchitecture::unimplemented_reason("deepseek41").is_none());
    }

    #[test]
    fn glm5_next_resolves_to_its_own_graph() {
        let arch = GGUFArchitecture::from_value("glm5-next").expect("glm5-next loads");
        assert!(
            matches!(arch, GGUFArchitecture::Glm5Next),
            "glm5-next must not alias another graph: {arch}"
        );
        let name = arch.to_string();
        assert_eq!(name, "glm5-next");
        assert!(!name.contains("glm4"), "{name}");
        assert!(!arch.supports_paged_attention());
        assert!(GGUFArchitecture::from_str("glm5-next").is_ok());
        assert!(GGUFArchitecture::unimplemented_reason("glm5-next").is_none());
        assert!(GGUFArchitecture::unimplemented_reason("llama").is_none());
        assert!(GGUFArchitecture::unimplemented_reason("glm4").is_none());
        assert!(GGUFArchitecture::unimplemented_reason("qwen4exp").is_none());
    }

    #[test]
    fn an_unknown_architecture_lists_the_supported_ones() {
        let err = GGUFArchitecture::from_value("not-a-real-arch")
            .expect_err("an invented architecture cannot parse");
        let msg = err.to_string();
        assert!(msg.contains("Unknown GGUF architecture"), "{msg}");
        assert!(msg.contains("llama"), "{msg}");
        assert!(msg.contains("glm-dsa"), "{msg}");
    }

    /// The dense-attention rule has to live with the loaders, not with callers: a caller
    /// holding a list of architecture strings goes stale silently in both directions when
    /// the engine gains an architecture or grows a paged kernel.
    #[test]
    fn mla_architectures_report_no_paged_attention_support() {
        assert!(!GGUFArchitecture::GlmDsa.supports_paged_attention());
        assert!(GGUFArchitecture::Llama.supports_paged_attention());
        assert!(GGUFArchitecture::Qwen3MoE.supports_paged_attention());
    }

    #[test]
    fn supported_architectures_still_parse_case_insensitively() {
        assert!(matches!(
            GGUFArchitecture::from_value("GLM-DSA").expect("glm-dsa is supported"),
            GGUFArchitecture::GlmDsa
        ));
        assert!(matches!(
            GGUFArchitecture::from_value("llama").expect("llama is supported"),
            GGUFArchitecture::Llama
        ));
    }
}

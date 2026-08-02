mod chat_template;
mod content;
mod gguf_tokenizer;
pub(crate) mod glm_moe;
use strum::{EnumIter, EnumString, IntoEnumIterator};

use anyhow::{Context, Result};
pub(crate) use chat_template::get_gguf_chat_template;
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};
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
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
/// Architectures whose GGUFs load far enough to be identified but have no graph
/// here yet. Naming them beats "Unknown GGUF architecture": the file is valid and
/// the operator's next step is a different engine or a different quant repo, not
/// a re-download. Each entry pairs the `general.architecture` value with the
/// model family and the components a loader still has to grow.
const RECOGNIZED_UNIMPLEMENTED: &[(&str, &str, &str)] = &[(
    "deepseek4",
    "DeepSeek-V4",
    "an MLA output LoRA (`o_lora_rank`/`o_groups`), a sqrt-softplus router, \
     hash layers, a sparse-attention indexer, and MXFP4 expert tensors",
)];

impl GGUFArchitecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        let lowered = value.as_ref().to_ascii_lowercase();
        if let Some((_, family, missing)) = RECOGNIZED_UNIMPLEMENTED
            .iter()
            .find(|(arch, _, _)| *arch == lowered)
        {
            anyhow::bail!(
                "GGUF architecture `{value}` ({family}) is recognized but not implemented: it needs {missing}. Supported architectures: {}.",
                Self::supported()
            );
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
    fn an_unknown_architecture_lists_the_supported_ones() {
        let err = GGUFArchitecture::from_value("not-a-real-arch")
            .expect_err("an invented architecture cannot parse");
        let msg = err.to_string();
        assert!(msg.contains("Unknown GGUF architecture"), "{msg}");
        assert!(msg.contains("llama"), "{msg}");
        assert!(msg.contains("glm-dsa"), "{msg}");
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

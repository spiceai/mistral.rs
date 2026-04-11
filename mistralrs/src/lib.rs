//! This crate is the Rust SDK for `mistral.rs`, providing an asynchronous interface for LLM inference.
//!
//! The Rust SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a high-performance
//! LLM inference engine supporting text, multimodal, speech, image generation, and embedding models.
//!
//! For loading multiple models simultaneously, use [`MultiModelBuilder`].
//! The returned [`Model`] supports `_with_model` method variants and runtime
//! model management (unload/reload).
//!
//! ```no_run
//! use mistralrs::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};
//!
//! #[tokio::main]
//! async fn main() -> mistralrs::error::Result<()> {
//!     let model = ModelBuilder::new("Qwen/Qwen3-4B")
//!         .with_auto_isq(IsqBits::Four)
//!         .build()
//!         .await?;
//!
//!     let response = model.chat("What is Rust's ownership model?").await?;
//!     println!("{response}");
//!     Ok(())
//! }
//! ```
//!
//! ## Capabilities
//!
//! | Capability | Builder | Example |
//! |---|---|---|
//! | Any model (auto-detect) | [`ModelBuilder`] | `examples/getting_started/text_generation/` |
//! | Text generation | [`TextModelBuilder`] | `examples/getting_started/text_generation/` |
//! | Multimodal (image+text) | [`MultimodalModelBuilder`] | `examples/getting_started/multimodal/` |
//! | GGUF quantized models | [`GgufModelBuilder`] | `examples/getting_started/gguf/` |
//! | Image generation | [`DiffusionModelBuilder`] | `examples/models/diffusion/` |
//! | Speech synthesis | [`SpeechModelBuilder`] | `examples/models/speech/` |
//! | Embeddings | [`EmbeddingModelBuilder`] | `examples/getting_started/embedding/` |
//! | Structured output | [`Model::generate_structured`] | `examples/advanced/json_schema/` |
//! | Tool calling | [`Tool`], [`ToolChoice`] | `examples/advanced/tools/` |
//! | Agents | [`AgentBuilder`] | `examples/advanced/agent/` |
//! | Multi-model | [`MultiModelBuilder`] | `examples/advanced/multi_model/` |
//! | LoRA / X-LoRA | [`LoraModelBuilder`], [`XLoraModelBuilder`] | `examples/advanced/lora/` |
//! | AnyMoE | [`AnyMoeModelBuilder`] | `examples/advanced/anymoe/` |
//! | MCP client | [`McpClientConfig`] | `examples/advanced/mcp_client/` |
//!
//! ## Model Loading
//!
//! All models are created through builder structs that follow a consistent pattern:
//!
//! ```no_run
//!    use anyhow::Result;
//!    use mistralrs::{
//!        ChatCompletionChunkResponse, ChunkChoice, Delta, IsqType, PagedAttentionMetaBuilder,
//!        Response, TextMessageRole, TextMessages, TextModelBuilder,
//!    };
//!
//! Use [`ModelBuilder::with_auto_isq`] for automatic platform-optimal quantization (e.g., `with_auto_isq(IsqBits::Four)`),
//! or [`ModelBuilder::with_isq`] with a specific [`IsqType`]: `Q4_0`, `Q4_1`, `Q4K`, `Q5_0`, `Q5_1`, `Q5K`,
//! `Q6K`, `Q8_0`, `Q8_1`, `HQQ4`, `HQQ8`, and more.
//!
//! ## Choosing a Request Type
//!
//!        let mut stream = model.stream_chat_request(messages).await?;

//!        while let Some(chunk) = stream.next().await {
//!            if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
//!                if let Some(ChunkChoice {
//!                    delta:
//!                        Delta {
//!                            content: Some(content),
//!                            ..
//!                        },
//!                    ..
//!                }) = choices.first()
//!                {
//!                    print!("{}", content);
//!                };
//!            }
//!        }
//!        Ok(())
//!    }
//! ```
//!
//! ## MCP example
//!
//! The MCP client integrates seamlessly with mistral.rs model builders:
//!
//! ```rust,no_run
//! use mistralrs::{TextModelBuilder, IsqType, McpClientConfig, McpServerConfig, McpServerSource};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mcp_config = McpClientConfig {
//!         servers: vec![/* your server configs */],
//!         auto_register_tools: true,
//!         tool_timeout_secs: Some(30),
//!         max_concurrent_calls: Some(5),
//!     };
//!     
//!     let model = TextModelBuilder::new("path/to/model".to_string())
//!         .with_isq(IsqType::Q8_0)
//!         .with_mcp_client(mcp_config)  // MCP tools automatically registered
//!         .build()
//!         .await?;
//!     
//!     // MCP tools are now available for automatic tool calling
//!     Ok(())
//! }
//! ```

mod agent;
mod anymoe;
mod auto_model;
pub mod blocking;
mod diffusion_model;
mod embedding_model;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod isq_setting;
mod lora_model;
mod messages;
mod model;
pub mod model_builder_trait;
mod speculative;
mod speech_model;
mod text_model;
mod xlora_model;

pub use agent::{
    Agent, AgentBuilder, AgentConfig, AgentEvent, AgentResponse, AgentStep, AgentStopReason,
    AgentStream, AsyncToolCallback, ToolCallbackType, ToolResult,
};
pub use anymoe::AnyMoeModelBuilder;
pub use diffusion_model::DiffusionModelBuilder;
pub use embedding_model::{EmbeddingModelBuilder, UqffEmbeddingModelBuilder};
pub use gguf::GgufModelBuilder;
pub use gguf_lora_model::GgufLoraModelBuilder;
pub use gguf_xlora_model::GgufXLoraModelBuilder;
pub use lora_model::LoraModelBuilder;
pub use messages::{
    EmbeddingRequest, EmbeddingRequestBuilder, EmbeddingRequestInput, RequestBuilder, RequestLike,
    TextMessageRole, TextMessages, VisionMessages,
};
pub use mistralrs_core::{
    McpClient, McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo,
};
pub use mistralrs_core::{SearchCallback, SearchResult, ToolCallback};
pub use model::{best_device, Model};
pub use model_builder_trait::{AnyModelBuilder, MultiModelBuilder};
pub use speculative::TextSpeculativeBuilder;
pub use speech_model::SpeechModelBuilder;
pub use text_model::{PagedAttentionMetaBuilder, TextModelBuilder, UqffTextModelBuilder};
pub use vision_model::{UqffVisionModelBuilder, VisionModelBuilder};
pub use xlora_model::XLoraModelBuilder;

pub use candle_core::{DType, Device, Result, Tensor};
pub use candle_nn::loss::cross_entropy as cross_entropy_loss;

/// Low-level types and internals re-exported from `mistralrs_core`.
///
/// Most users don't need these types directly. They're available for advanced
/// use cases like custom pipelines, device mapping, or direct engine access.
pub mod core;

// ========== Response Types ==========
pub use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Choice, ChunkChoice, CompletionResponse,
    Delta, Logprobs, Response, ResponseMessage, TopLogprob, Usage,
};

// ========== Request Types ==========
pub use mistralrs_core::{Constraint, LlguidanceGrammar, MessageContent, NormalRequest, Request};

// ========== Sampling ==========
pub use mistralrs_core::{DrySamplingParams, SamplingParams, StopTokens};

// ========== Tool Types ==========
pub use mistralrs_core::{
    CalledFunction, Function, Tool, ToolCallResponse, ToolCallType, ToolChoice, ToolType,
};

// ========== Config Types ==========
pub use mistralrs_core::{
    DefaultSchedulerMethod, IsqType, MemoryGpuConfig, ModelDType, PagedAttentionConfig,
    SchedulerConfig, WebSearchOptions,
};

// ========== Audio Types ==========
pub use mistralrs_core::AudioInput;

// ========== Custom Logits ==========
pub use mistralrs_core::CustomLogitsProcessor;

// ========== Model Category ==========
pub use mistralrs_core::ModelCategory;

// ========== Search Types ==========
pub use mistralrs_core::{SearchEmbeddingModel, SearchFunctionParameters};

// ========== Speech Types ==========
pub use mistralrs_core::{speech_utils, SpeechLoaderType};

// ========== AnyMoe Types ==========
pub use mistralrs_core::{AnyMoeConfig, AnyMoeExpertType};

// ========== Diffusion Types ==========
pub use mistralrs_core::{
    DiffusionGenerationParams, DiffusionLoaderType, ImageGenerationResponseFormat,
};

// ========== Speculative Types ==========
pub use mistralrs_core::SpeculativeConfig;

// ========== Device Mapping ==========
pub use mistralrs_core::{AutoDeviceMapParams, DeviceMapSetting};

// ========== Topology ==========
pub use mistralrs_core::{LayerTopology, Topology};

// ========== Token Source ==========
pub use mistralrs_core::TokenSource;

// ========== Engine (Advanced) ==========
pub use mistralrs_core::{MistralRs, RequestMessage, ResponseOk};

// ========== Utilities ==========
pub use mistralrs_core::{initialize_logging, paged_attn_supported, parse_isq_value};

// ========== llguidance ==========
pub use mistralrs_core::llguidance;

// Re-export the tool proc macro for ergonomic tool definition
pub use mistralrs_macros::tool;

// Re-export schemars for use in tool definitions
pub use schemars;

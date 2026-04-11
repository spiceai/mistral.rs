use mistralrs_core::{GGUFLoaderBuilder, GGUFSpecificConfig, Ordering};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_gguf_loader, maybe_initialize_logging,
    },
    GgufModelBuilder, Model,
};

/// Wrapper of [`GgufModelBuilder`] for LoRA models.
pub struct GgufLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    lora_model_id: String,
    ordering: Ordering,
}

impl GgufLoraModelBuilder {
    /// Create a GGUF LoRA builder from a [`GgufModelBuilder`], LoRA model ID, and ordering.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        lora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            gguf_model,
            lora_model_id: lora_model_id.to_string(),
            ordering,
        }
    }

    /// Load the GGUF LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let gguf_model = self.gguf_model.clone();
        let config = GGUFSpecificConfig {
            topology: self.gguf_model.topology,
        };

        maybe_initialize_logging(self.gguf_model.with_logging);

        let loader = GGUFLoaderBuilder::new(
            self.gguf_model.chat_template,
            self.gguf_model.tok_model_id,
            self.gguf_model.model_id,
            self.gguf_model.files,
            config,
            self.gguf_model.no_kv_cache,
            self.gguf_model.jinja_explicit,
        )
        .with_lora(self.lora_model_id, self.ordering)
        .build();

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_gguf_loader(gguf_model, loader).await?;

        let scheduler_method = match self.gguf_model.paged_attn_cfg {
            Some(_) => {
                let config = pipeline
                    .lock()
                    .await
                    .get_metadata()
                    .cache_config
                    .as_ref()
                    .unwrap()
                    .clone();

                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: self.gguf_model.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.gguf_model.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(
            pipeline,
            scheduler_method,
            self.gguf_model.throughput_logging,
            self.gguf_model.search_embedding_model,
        );
        if let Some(cb) = self.gguf_model.search_callback.clone() {
            runner = runner.with_search_callback(cb);
        }
        for (name, cb) in &self.gguf_model.tool_callbacks {
            runner = runner.with_tool_callback(name.clone(), cb.clone());
        }
        runner = runner
            .with_no_kv_cache(self.gguf_model.no_kv_cache)
            .with_no_prefix_cache(self.gguf_model.prefix_cache_n.is_none());

        if let Some(n) = self.gguf_model.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build().await))
    }
}

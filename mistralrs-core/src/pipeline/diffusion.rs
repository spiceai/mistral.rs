use super::loaders::{DiffusionModelPaths, DiffusionModelPathsInner};
use super::{
    AdapterActivationMixin, AnyMoePipelineMixin, Cache, CacheManagerMixin, DiffusionLoaderType,
    DiffusionModel, DiffusionModelLoader, FluxLoader, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::diffusion_models::processor::{DiffusionProcessor, ModelInputs};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::ChatTemplate;
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::{DeviceMapMetadata, PagedAttentionConfig, Pipeline, TryIntoDType};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::{DynamicImage, RgbImage};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::io;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct DiffusionPipeline {
    model: Box<dyn DiffusionModel + Send + Sync>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: Cache,
}

/// A loader for a vision (non-quantized) model.
pub struct DiffusionLoader {
    inner: Box<dyn DiffusionModelLoader>,
    model_id: String,
    config: DiffusionSpecificConfig,
    kind: ModelKind,
}

#[derive(Default)]
/// A builder for a loader for a vision (non-quantized) model.
pub struct DiffusionLoaderBuilder {
    model_id: Option<String>,
    config: DiffusionSpecificConfig,
    kind: ModelKind,
}

#[derive(Clone, Default)]
/// Config specific to loading a vision model.
pub struct DiffusionSpecificConfig {
    pub use_flash_attn: bool,
}

impl DiffusionLoaderBuilder {
    pub fn new(config: DiffusionSpecificConfig, model_id: Option<String>) -> Self {
        Self {
            config,
            model_id,
            kind: ModelKind::Normal,
        }
    }

    pub fn build(self, loader: DiffusionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn DiffusionModelLoader> = match loader {
            DiffusionLoaderType::Flux => Box::new(FluxLoader { offload: false }),
            DiffusionLoaderType::FluxOffloaded => Box::new(FluxLoader { offload: true }),
        };
        Box::new(DiffusionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
        })
    }
}

impl Loader for DiffusionLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = {
            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(&token_source)?)
                .build()?;
            let revision = revision.unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                self.model_id.clone(),
                RepoType::Model,
                revision.clone(),
            ));
            let model_id = std::path::Path::new(&self.model_id);
            let filenames = self.inner.get_model_paths(&api, model_id)?;
            let config_filenames = self.inner.get_config_filenames(&api, model_id)?;
            Ok(Box::new(DiffusionModelPaths(DiffusionModelPathsInner {
                config_filenames,
                filenames,
            })))
        };
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths = &paths
            .as_ref()
            .as_any()
            .downcast_ref::<DiffusionModelPaths>()
            .expect("Path downcast failed.")
            .0;

        // Otherwise, the device mapper will print it
        if mapper.is_dummy() {
            info!(
                "Loading model `{}` on {}.",
                self.get_id(),
                device.device_pretty_repr()
            );
        } else {
            anyhow::bail!("Device mapping is not supported for Diffusion models.");
        }

        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for Diffusion models.");
        }

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for Diffusion models, disabling it.");

            paged_attn_config = None;
        }

        let configs = paths
            .config_filenames
            .iter()
            .map(std::fs::read_to_string)
            .collect::<io::Result<Vec<_>>>()?;

        let mapper = mapper.into_mapper(usize::MAX, device, None)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let model = match self.kind {
            ModelKind::Normal => {
                let vbs = paths
                    .filenames
                    .iter()
                    .zip(self.inner.force_cpu_vb())
                    .map(|(path, force_cpu)| {
                        from_mmaped_safetensors(
                            vec![path.clone()],
                            Vec::new(),
                            Some(dtype),
                            if force_cpu { &Device::Cpu } else { device },
                            silent,
                            None,
                            |_| true,
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;

                self.inner.load(
                    configs,
                    self.config.use_flash_attn,
                    vbs,
                    crate::pipeline::NormalLoadingMetadata {
                        mapper,
                        loading_isq: false,
                        real_device: device.clone(),
                    },
                    attention_mechanism,
                    silent,
                )?
            }
            _ => unreachable!(),
        };

        let max_seq_len = model.max_seq_len();
        Ok(Arc::new(Mutex::new(DiffusionPipeline {
            model,
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                tok_trie: None,
                is_xlora: false,
                num_hidden_layers: 1, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: self.kind.clone(),
                has_no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                prompt_batchsize: None,
            }),
            dummy_cache: Cache::new(0, false),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for DiffusionPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(DiffusionProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for DiffusionPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Diffusion models do not support ISQ for now.")
    }
}

impl CacheManagerMixin for DiffusionPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence], _modify_draft_cache: bool) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence], _modify_draft_cache: bool) {}
    fn set_none_cache(&self, _reset_non_granular: bool, _modify_draft_cache: bool) {}
    fn cache(&self) -> &Cache {
        &self.dummy_cache
    }
}

impl AdapterActivationMixin for DiffusionPipeline {
    fn activate_adapters(&mut self, _adapters: Vec<String>) -> Result<usize> {
        anyhow::bail!("Diffusion models do not support adapter activation.");
    }
}

impl MetadataMixin for DiffusionPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for DiffusionPipeline {
    fn forward_inputs(&mut self, inputs: Box<dyn Any>) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs { prompts, params } = *inputs.downcast().expect("Downcast failed.");
        let img = self.model.forward(prompts, params)?.to_dtype(DType::U8)?;
        let (_b, c, h, w) = img.dims4()?;
        let mut images = Vec::new();
        for b_img in img.chunk(img.dim(0)?, 0)? {
            let flattened = b_img.squeeze(0)?.permute((1, 2, 0))?.flatten_all()?;
            if c != 3 {
                candle_core::bail!("Expected 3 channels in image output");
            }
            #[allow(clippy::cast_possible_truncation)]
            images.push(DynamicImage::ImageRgb8(
                RgbImage::from_raw(w as u32, h as u32, flattened.to_vec1::<u8>()?).ok_or(
                    candle_core::Error::Msg("RgbImage has invalid capacity.".to_string()),
                )?,
            ));
        }
        Ok(ForwardInputsResult::Image { images })
    }
    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManager,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("`sample_causal_gen` is incompatible with `DiffusionPipeline`");
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Diffusion
    }
}

impl AnyMoePipelineMixin for DiffusionPipeline {}

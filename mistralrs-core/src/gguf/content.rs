use std::{
    collections::HashMap,
    fs,
    io::{Read, Seek},
};

use anyhow::Context;
use candle_core::{
    quantized::{
        gguf_file::{self, TensorInfo, Value},
        GgmlDType, QTensor,
    },
    Device, Result,
};
use indexmap::IndexMap;
use tracing::{debug, info};

use crate::DEBUG;

use super::GGUFArchitecture;

/// List of all GgmlDType variants from Candle.
/// This should be kept in sync with candle_core::quantized::GgmlDType.
/// If Candle adds new dtype variants, add them here to include in error messages.
/// Reference: candle-core/src/quantized/mod.rs in the Candle repository.
const KNOWN_DTYPES: &[GgmlDType] = &[
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8K,
    // Add newer ones here if Candle adds more
];

fn get_supported_gguf_dtypes() -> String {
    KNOWN_DTYPES
        .iter()
        .map(|dt| format!("{:?}", dt))
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_gguf_value(value: &Value) -> String {
    match value {
        Value::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        Value::Bool(b) => b.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::String(x) => x.to_string(),
        Value::U8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
    }
}

// Internal invariant: contents and readers must be paired.
/// This abstracts the files for a GGUF model and enables multiple files to be used.
pub struct Content<'a, R: std::io::Seek + std::io::Read> {
    contents: Vec<gguf_file::Content>,
    readers: &'a mut [&'a mut R],
    arch: GGUFArchitecture,
    all_metadata: HashMap<String, Value>,
}

impl<'a, R: std::io::Seek + std::io::Read> Content<'a, R> {
    /// Create a `Content` from a set of file readers.
    pub fn from_readers(readers: &'a mut [&'a mut R]) -> Result<Self> {
        let mut contents = Vec::new();
        let n_readers = readers.len();
        for (i, reader) in readers.iter_mut().enumerate() {
            match gguf_file::Content::read(reader) {
                Ok(c) => {
                    contents.push(c);
                }
                Err(e) => {
                    let error_msg = format!("{}", e);
                    if error_msg.contains("unknown dtype for tensor") {
                        {
                            candle_core::bail!(
                                "Critical failure loading model part {}\n\
                                Verify you are using a supported quantization type\n\
                                Supported types: {}\n\
                                Candle error: {}",
                                i,
                                get_supported_gguf_dtypes(),
                                e
                            );
                        }
                    }
                    candle_core::bail!(
                        "Critical failure loading model part {}!\n\
                        Check whether your current quantization format is supported: {}",
                        i,
                        e
                    );
                }
            }
        }
        let n_splits = contents
            .iter()
            .filter_map(|ct| {
                ct.metadata
                    .get("split.count")
                    .map(|val| val.to_u64().unwrap())
            })
            .fold(Vec::new(), |mut accum, x| {
                if !accum.contains(&x) {
                    accum.push(x);
                }
                accum
            });
        if n_splits.len() > 1 {
            candle_core::bail!("GGUF files have differing `split.count` values: {n_splits:?}. Perhaps the GGUF files do not match?");
        }
        #[allow(clippy::cast_possible_truncation)]
        if !n_splits.is_empty() && n_readers != n_splits[0] as usize {
            candle_core::bail!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            );
        } else if n_splits.len() == 1 {
            info!("GGUF file has been split into {} shards", n_splits[0]);
        }

        let mut arch = None;
        for ct in &contents {
            if !ct.metadata.contains_key("general.architecture") {
                continue;
            }

            arch = Some(
                ct.metadata["general.architecture"]
                    .to_string()
                    .context("Model metadata should have declared an architecture")
                    .and_then(GGUFArchitecture::from_value)
                    .unwrap(),
            );
        }
        let arch = arch.expect("GGUF files must specify `general.architecture`");

        let mut all_metadata = HashMap::new();
        for content in &contents {
            all_metadata.extend(content.metadata.clone())
        }

        Ok(Self {
            contents,
            readers,
            arch,
            all_metadata,
        })
    }

    pub fn arch(&self) -> GGUFArchitecture {
        self.arch
    }

    /// Retrieve a tensor info, searching through each content.
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        for ct in &self.contents {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return Ok(tensor_info);
            }
        }
        candle_core::bail!("Cannot find tensor info for {name}")
    }

    /// Retrieve a tensor, searching through each content.
    pub fn tensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        for (ct, reader) in self.contents.iter().zip(self.readers.iter_mut()) {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return tensor_info.read(reader, ct.tensor_data_offset, device);
            }
        }
        candle_core::bail!("Cannot find tensor info for {name}")
    }

    /// Retrieve a tensor sharded along its outer (dim-0) axis: only indices
    /// `[start, start + len)` are read from disk and materialized on `device`, so a rank
    /// never stages the whole tensor — essential on unified-memory GPUs where host and
    /// device share the same DRAM.
    ///
    /// This is the loading primitive for tensor parallelism over GGUF weights. Candle
    /// reverses GGUF's `ne` order, so dim 0 is the slowest-varying axis and each dim-0
    /// index occupies a contiguous byte range; the slice therefore needs no dequantization
    /// and no quant-block re-alignment (asserted below). Two uses:
    ///   - stacked routed-expert tensors, where dim 0 is the expert index
    ///     (`[n_experts, out, in]`), giving expert parallelism;
    ///   - column-parallel linears, where dim 0 is the output-feature axis
    ///     (`[out, in]`), giving each rank a slice of the output features.
    ///
    /// Requesting the full extent falls back to an ordinary read.
    pub fn tensor_dim0_shard(
        &mut self,
        name: &str,
        start: usize,
        len: usize,
        device: &Device,
    ) -> Result<QTensor> {
        self.read_dim0_range(name, start, len, true, device)
    }

    fn read_dim0_range(
        &mut self,
        name: &str,
        start: usize,
        len: usize,
        keep_outer: bool,
        device: &Device,
    ) -> Result<QTensor> {
        for (ct, reader) in self.contents.iter().zip(self.readers.iter_mut()) {
            if let Some(info) = ct.tensor_infos.get(name) {
                let dims = info.shape.dims();
                if dims.is_empty() {
                    candle_core::bail!("cannot shard {name}: tensor has no dimensions");
                }
                if !keep_outer && dims.len() < 2 {
                    candle_core::bail!(
                        "cannot index {name} along dim 0: rank {} leaves no trailing dimensions",
                        dims.len()
                    );
                }
                let outer = dims[0];
                if start + len > outer {
                    candle_core::bail!(
                        "shard [{start}, {}) of {name} is out of range for outer dimension {outer}",
                        start + len
                    );
                }
                if keep_outer && start == 0 && len == outer {
                    return info.read(reader, ct.tensor_data_offset, device);
                }
                let block_size = info.ggml_dtype.block_size();
                let type_size = info.ggml_dtype.type_size();
                let elems_per_index = info.shape.elem_count() / outer;
                if elems_per_index % block_size != 0 {
                    candle_core::bail!(
                        "cannot shard {name}: per-index element count {elems_per_index} is not divisible by block size {block_size}"
                    );
                }
                let bytes_per_index = elems_per_index / block_size * type_size;
                let shard_start = ct
                    .tensor_data_offset
                    .saturating_add(info.offset)
                    .saturating_add((start * bytes_per_index) as u64);
                let shard_bytes = len * bytes_per_index;

                let file_size = reader.seek(std::io::SeekFrom::End(0))?;
                if shard_start.saturating_add(shard_bytes as u64) > file_size {
                    candle_core::bail!(
                        "shard of {name} needs {shard_bytes} bytes at offset {shard_start}, exceeds file size {file_size}"
                    );
                }
                let mut raw = vec![0u8; shard_bytes];
                reader.seek(std::io::SeekFrom::Start(shard_start))?;
                reader.read_exact(&mut raw)?;

                let shard_dims = if keep_outer {
                    let mut d = dims.to_vec();
                    d[0] = len;
                    d
                } else {
                    dims[1..].to_vec()
                };
                return candle_core::quantized::ggml_file::qtensor_from_ggml(
                    info.ggml_dtype,
                    &raw,
                    shard_dims,
                    device,
                );
            }
        }
        candle_core::bail!("Cannot find tensor info for {name}")
    }

    /// Check for a tensor, searching through each content.
    pub fn has_tensor(&self, name: &str) -> bool {
        for ct in self.contents.iter() {
            if ct.tensor_infos.contains_key(name) {
                return true;
            }
        }
        false
    }

    /// Print metadata for these contents.
    /// This will also log tensor name, shape and dtype to `mistralrs_gguf_tensors.txt` is DEBUG is enabled.
    pub fn print_metadata(&self) -> anyhow::Result<()> {
        // Find the ct with general.architecture
        let mut keys = Vec::new();
        let mut metadatas = Vec::new();
        let mut tensors = Vec::new();
        for ct in &self.contents {
            keys.extend(ct.metadata.keys());
            metadatas.push(&ct.metadata);

            if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
                for (name, info) in &ct.tensor_infos {
                    tensors.push(format!(
                        "name = `{name}`, shape = {:?}, dtype = {:?}",
                        info.shape.clone(),
                        info.ggml_dtype
                    ));
                }
            }
        }

        debug!("Model config:");
        keys.sort();
        let mut output_keys = IndexMap::new();
        for name in keys {
            if !name.contains("tokenizer") {
                for metadata in &metadatas {
                    if let Some(val) = metadata.get(name) {
                        output_keys.insert(name, parse_gguf_value(val));
                    }
                }
            }
        }
        for (name, val) in output_keys {
            debug!("{name}: {val}");
        }

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            fs::write(
                "mistralrs_gguf_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!("Debug is enabled, wrote the names and information about each tensor to `mistralrs_gguf_tensors.txt`.");
        }

        anyhow::Ok(())
    }

    /// Get all metadatas
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}

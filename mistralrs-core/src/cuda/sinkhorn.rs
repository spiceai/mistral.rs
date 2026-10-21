//! Fused CUDA sinkhorn for the DeepSeek-V4 hyper-connection mixer.
//!
//! [`crate::gguf::dsv4_hc::sinkhorn`] runs as about `6 * iters` tiny tensor ops on a
//! `[n, hc, hc]` matrix; with 20 iterations and two mixers per layer, that is most of
//! the kernel launches in a DeepSeek-V4.1 decode step. This runs the same sequence as
//! one kernel.

use candle_core::{
    backend::BackendStorage, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Tensor,
};

/// The largest `hc` the kernel holds in registers (`SINKHORN_MAX_HC` in `sinkhorn.cu`).
pub(crate) const MAX_HC: usize = 16;

struct SinkhornOp {
    iters: usize,
    eps: f32,
}

impl CustomOp1 for SinkhornOp {
    fn name(&self) -> &'static str {
        "dsv4-sinkhorn"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused sinkhorn requires CUDA storage")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if storage.dtype() != DType::F32 {
            candle_core::bail!("the fused sinkhorn takes f32, got {:?}", storage.dtype());
        }
        if !layout.is_contiguous() {
            candle_core::bail!("the fused sinkhorn takes a contiguous `[n, hc, hc]` tensor");
        }
        let dims = layout.shape().dims();
        let (hc, src) = match dims {
            [.., hc, src] => (*hc, *src),
            _ => candle_core::bail!("the fused sinkhorn takes `[n, hc, hc]`, got {dims:?}"),
        };
        if hc != src || hc == 0 || hc > MAX_HC {
            candle_core::bail!(
                "the fused sinkhorn takes square matrices of at most {MAX_HC}, got {dims:?}"
            );
        }
        let elements = layout.shape().elem_count();
        let n = elements / (hc * hc);
        let iters = i32::try_from(self.iters).map_err(candle_core::Error::wrap)?;
        let device = storage.device();
        let stream = device.cuda_stream();
        let input = storage.as_cuda_slice::<f32>()?;
        let input = input.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            crate::cuda::ffi::dsv4_sinkhorn_f32(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *mut core::ffi::c_void,
                n as i64,
                hc as i32,
                iters,
                self.eps,
                stream.cu_stream() as i64,
            )
        };
        drop(output_guard);
        drop(input_guard);
        if status != 0 {
            candle_core::bail!("dsv4_sinkhorn_f32 failed with status {status}");
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

/// Runs the sinkhorn mixer on a CUDA `[n, hc, hc]` f32 tensor as one kernel.
pub(crate) fn sinkhorn(logits: &Tensor, iters: usize, eps: f32) -> Result<Tensor> {
    logits
        .contiguous()?
        .apply_op1_no_bwd(&SinkhornOp { iters, eps })
}

//! Runs the ring backend's collectives between real ranks and checks every element.
//!
//! Start one process per rank with `RING_CONFIG` pointing at that rank's ring config:
//!
//! ```text
//! RING_CONFIG=rank0.json cargo run --release -p mistralrs-quant --features ring --example ring_collectives
//! ```
//!
//! Each rank sum-reduces and gathers tensors from 1 element to 20 MiB, contributing values
//! that make every expected element exact in f32, and exits non-zero on the first wrong
//! element. `ring_collectives mismatch` gives the last rank a tensor twice as long, which
//! every rank must refuse rather than reduce.

use std::sync::Arc;

use candle_core::{Device, Tensor};
use mistralrs_quant::{distributed::AllGather, Comm, Id, RingConfig, SumAllReduce};

const SIZES: [usize; 6] = [1, 5_120, 262_144, 786_432, 1_000_003, 5_242_880];
const ITERATIONS: usize = 20;

/// Rank `rank`'s contribution to element `i` of iteration `step`: small integers, so every
/// sum is exact.
fn value(rank: usize, step: usize, i: usize) -> f32 {
    ((rank + 1) * 1000 + (step * 7 + i) % 997) as f32
}

fn main() -> candle_core::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("mistralrs_quant=info"))
        .init();
    let mismatch = std::env::args().nth(1).as_deref() == Some("mismatch");
    let config = RingConfig::load();
    let (rank, world) = (config.rank, config.world_size);
    let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, rank, world)?);
    let reduce = SumAllReduce::new(&comm);
    let gather = AllGather::new(&comm, 0);

    if mismatch {
        // Every rank holds one 1 MiB chunk of `rank + 1`; the last rank holds a second chunk
        // of 100s. A rank that sums that second chunk in instead of refusing reports it.
        let chunk = 262_144;
        let mut mine = vec![(rank + 1) as f32; chunk];
        if rank == world - 1 {
            mine.extend(std::iter::repeat_n(100.0, chunk));
        }
        let len = mine.len();
        let xs = Tensor::from_vec(mine, len, &Device::Cpu)?;
        let outcome = reduce
            .sum_all_reduce(&xs)
            .and_then(|sum| Ok(sum.to_vec1::<f32>()?[0]));
        println!("rank {rank} mismatch: first element {outcome:?}");
        std::process::exit(i32::from(outcome.is_ok()));
    }

    for &len in &SIZES {
        let started = std::time::Instant::now();
        for step in 0..ITERATIONS {
            let mine: Vec<f32> = (0..len).map(|i| value(rank, step, i)).collect();
            let xs = Tensor::from_vec(mine.clone(), len, &Device::Cpu)?;

            let sum = reduce.sum_all_reduce(&xs)?.to_vec1::<f32>()?;
            for (i, got) in sum.iter().enumerate() {
                let want: f32 = (0..world).map(|r| value(r, step, i)).sum();
                if *got != want {
                    candle_core::bail!("rank {rank}: sum of {len} elements, step {step}: element {i} is {got}, want {want}");
                }
            }

            let all = gather.all_gather(&xs)?.to_vec1::<f32>()?;
            for (i, got) in all.iter().enumerate() {
                let want = value(i / len, step, i % len);
                if *got != want {
                    candle_core::bail!("rank {rank}: gather of {len} elements, step {step}: element {i} is {got}, want {want}");
                }
            }
        }
        println!(
            "rank {rank}: {len} elements x {ITERATIONS}: sums and gathers exact ({:.1} ms per sum+gather)",
            started.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64
        );
    }
    Ok(())
}

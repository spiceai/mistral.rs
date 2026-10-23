//! Runs the ring backend's collectives between real ranks and checks every element.
//!
//! Start one process per rank with `RING_CONFIG` pointing at that rank's ring config:
//!
//! ```text
//! RING_CONFIG=rank0.json cargo run --release -p mistralrs-quant --features ring --example ring_collectives
//! ```
//!
//! Each rank sum-reduces and gathers tensors from none to 20 MiB, contributing values
//! that make every expected element exact in f32, and exits non-zero on the first wrong
//! element. `ring_collectives mismatch` gives the last rank a tensor twice as long, a whole
//! 1 MiB chunk more than the others, which every rank must refuse rather than reduce;
//! `mismatch-small` does the same with one element against two. In both, each rank prints how long it took to answer, then keeps its process
//! and connections open for 15 s, so a rank that only learns of the failure when a peer exits
//! shows it. `ring_collectives latency` times 10,000 sum-reduces of a decode step's 20 KiB and
//! prints their P50, P99 and P99.9.

use std::sync::Arc;

use candle_core::{Device, Tensor};
use mistralrs_quant::{distributed::AllGather, Comm, Id, RingConfig, SumAllReduce};

const SIZES: [usize; 7] = [0, 1, 5_120, 262_144, 786_432, 1_000_003, 5_242_880];
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
    let mode = std::env::args().nth(1);
    let mismatch = matches!(mode.as_deref(), Some("mismatch" | "mismatch-small"));
    let config = RingConfig::load();
    let (rank, world) = (config.rank, config.world_size);
    let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, rank, world)?);
    let reduce = SumAllReduce::new(&comm);
    let gather = AllGather::new(&comm, 0);

    if mode.as_deref() == Some("latency") {
        // The all-reduce of a decode step, 5120 f32 (20 KiB), timed one at a time.
        const LEN: usize = 5_120;
        const RUNS: usize = 10_000;
        let xs = Tensor::from_vec(vec![1f32; LEN], LEN, &Device::Cpu)?;
        let mut micros = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let started = std::time::Instant::now();
            reduce.sum_all_reduce(&xs)?;
            micros.push(started.elapsed().as_secs_f64() * 1e6);
        }
        micros.sort_by(f64::total_cmp);
        let at = |q: f64| micros[((RUNS as f64 * q).ceil() as usize).clamp(1, RUNS) - 1];
        println!(
            "rank {rank}: {RUNS} sums of {LEN} elements: p50 {:.1} us, p99 {:.1} us, p99.9 {:.1} us",
            at(0.5),
            at(0.99),
            at(0.999)
        );
        return Ok(());
    }

    if mismatch {
        // Every rank holds one chunk of `rank + 1`, 1 MiB or, in `mismatch-small`, one
        // element; the last rank holds a second chunk of 100s. A rank that sums that second
        // chunk in instead of refusing reports it.
        let chunk = if mode.as_deref() == Some("mismatch-small") {
            1
        } else {
            262_144
        };
        let started = std::time::Instant::now();
        let mut mine = vec![(rank + 1) as f32; chunk];
        if rank == world - 1 {
            mine.extend(std::iter::repeat_n(100.0, chunk));
        }
        let len = mine.len();
        let xs = Tensor::from_vec(mine, len, &Device::Cpu)?;
        let outcome = reduce
            .sum_all_reduce(&xs)
            .and_then(|sum| Ok(sum.to_vec1::<f32>()?[0]));
        println!(
            "rank {rank} mismatch after {} ms: first element {outcome:?}",
            started.elapsed().as_millis()
        );
        std::thread::sleep(std::time::Duration::from_secs(15));
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

//! Benchmarks comparing Vec-based Orderbook vs FastOrderbook performance
//!
//! Run with: cargo bench --package polysniper-core

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use polysniper_core::{
    orderbook::{FastOrderbook, OrderbookDelta, PackedPriceLevel, SharedOrderbook},
    Orderbook, PriceLevel,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_decimal::Decimal;

/// Generate a random legacy orderbook with the given number of levels
fn generate_legacy_orderbook(rng: &mut StdRng, levels: usize) -> Orderbook {
    let mut bids = Vec::with_capacity(levels);
    let mut asks = Vec::with_capacity(levels);

    // Generate bids (descending from 0.50)
    for i in 0..levels {
        let price = Decimal::new(5000 - (i as i64) * 10, 4); // 0.5000, 0.4990, ...
        let size = Decimal::new(rng.gen_range(100..1000), 0);
        bids.push(PriceLevel { price, size });
    }

    // Generate asks (ascending from 0.51)
    for i in 0..levels {
        let price = Decimal::new(5100 + (i as i64) * 10, 4); // 0.5100, 0.5110, ...
        let size = Decimal::new(rng.gen_range(100..1000), 0);
        asks.push(PriceLevel { price, size });
    }

    Orderbook {
        token_id: "bench_token".to_string(),
        market_id: "bench_market".to_string(),
        bids,
        asks,
        timestamp: chrono::Utc::now(),
    }
}

/// Generate random delta updates
fn generate_delta(rng: &mut StdRng, updates: usize) -> OrderbookDelta {
    let mut delta = OrderbookDelta::new(chrono::Utc::now().timestamp_millis());

    for _ in 0..updates / 2 {
        // Random bid update
        let price = rng.gen_range(4500..5000);
        let size = if rng.gen_bool(0.1) {
            0 // 10% chance of removal
        } else {
            rng.gen_range(100_000_000..1_000_000_000)
        };
        delta.add_bid(price, size);

        // Random ask update
        let price = rng.gen_range(5100..5600);
        let size = if rng.gen_bool(0.1) {
            0
        } else {
            rng.gen_range(100_000_000..1_000_000_000)
        };
        delta.add_ask(price, size);
    }

    delta
}

/// Benchmark: Clone vs Reference access patterns
fn bench_orderbook_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_access");
    let mut rng = StdRng::seed_from_u64(42);

    for levels in [10, 50, 100].iter() {
        let legacy = generate_legacy_orderbook(&mut rng, *levels);
        let fast = FastOrderbook::from_legacy(&legacy);
        let shared = SharedOrderbook::new(fast.clone());

        // Benchmark: Clone entire legacy orderbook (current pattern)
        group.bench_with_input(
            BenchmarkId::new("legacy_clone", levels),
            &legacy,
            |b, ob| {
                b.iter(|| {
                    let cloned = black_box(ob.clone());
                    black_box(cloned.best_bid())
                })
            },
        );

        // Benchmark: Reference access with FastOrderbook
        group.bench_with_input(
            BenchmarkId::new("fast_reference", levels),
            &fast,
            |b, ob| {
                b.iter(|| {
                    let best = black_box(ob.best_bid());
                    black_box(best)
                })
            },
        );

        // Benchmark: SharedOrderbook get() (zero-copy)
        group.bench_with_input(
            BenchmarkId::new("shared_get", levels),
            &shared,
            |b, ob| {
                b.iter(|| {
                    let inner = black_box(ob.get());
                    black_box(inner.best_bid())
                })
            },
        );

        // Benchmark: Top N access - legacy requires iteration
        group.bench_with_input(
            BenchmarkId::new("legacy_top_5", levels),
            &legacy,
            |b, ob| {
                b.iter(|| {
                    let top: Vec<_> = black_box(ob.bids.iter().take(5).collect());
                    black_box(top)
                })
            },
        );

        // Benchmark: Top N access - FastOrderbook slice (zero-copy)
        group.bench_with_input(
            BenchmarkId::new("fast_top_5", levels),
            &fast,
            |b, ob| {
                b.iter(|| {
                    let top = black_box(ob.top_bids(5));
                    black_box(top)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Orderbook update throughput
fn bench_orderbook_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_update");
    let mut rng = StdRng::seed_from_u64(42);

    for levels in [50, 100].iter() {
        // Benchmark: Legacy orderbook rebuild (typical pattern)
        let legacy = generate_legacy_orderbook(&mut rng, *levels);

        group.bench_with_input(
            BenchmarkId::new("legacy_rebuild", levels),
            &legacy,
            |b, base_ob| {
                b.iter_batched(
                    || {
                        // Setup: create a new orderbook to simulate update
                        let mut new_ob = base_ob.clone();
                        // Simulate price changes
                        for level in new_ob.bids.iter_mut().take(5) {
                            level.size += Decimal::ONE;
                        }
                        new_ob
                    },
                    |new_ob| {
                        // Measure: full replacement (typical pattern)
                        black_box(new_ob)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Benchmark: FastOrderbook delta update
        let fast = FastOrderbook::from_legacy(&legacy);
        let delta = generate_delta(&mut rng, 10);

        group.bench_with_input(
            BenchmarkId::new("fast_delta", levels),
            &(fast.clone(), delta.clone()),
            |b, (base_ob, delta)| {
                b.iter_batched(
                    || base_ob.clone(),
                    |mut ob| {
                        ob.apply_delta(delta);
                        black_box(ob)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark: Price lookup performance
fn bench_price_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("price_lookup");
    let mut rng = StdRng::seed_from_u64(42);

    for levels in [50, 100].iter() {
        let legacy = generate_legacy_orderbook(&mut rng, *levels);
        let fast = FastOrderbook::from_legacy(&legacy);

        // Benchmark: Linear search in legacy orderbook
        let search_price = Decimal::new(4950, 4); // 0.4950
        group.bench_with_input(
            BenchmarkId::new("legacy_linear", levels),
            &(&legacy, search_price),
            |b, (ob, price)| {
                b.iter(|| {
                    let found = ob.bids.iter().find(|l| l.price == *price);
                    black_box(found)
                })
            },
        );

        // Benchmark: Binary search in FastOrderbook
        let search_price_i64 = 4950; // Same price as i64
        group.bench_with_input(
            BenchmarkId::new("fast_binary", levels),
            &(&fast, search_price_i64),
            |b, (ob, price)| {
                b.iter(|| {
                    let found = ob.bid_at_price(*price);
                    black_box(found)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    let mut rng = StdRng::seed_from_u64(42);

    // Benchmark: Create new legacy orderbook
    group.bench_function("legacy_create_100", |b| {
        b.iter(|| {
            let ob = generate_legacy_orderbook(&mut StdRng::seed_from_u64(42), 100);
            black_box(ob)
        })
    });

    // Benchmark: Create new FastOrderbook (pre-allocated)
    group.bench_function("fast_create_100", |b| {
        b.iter(|| {
            let ob = FastOrderbook::new("token".to_string(), "market".to_string());
            black_box(ob)
        })
    });

    // Benchmark: Convert legacy to fast
    let legacy = generate_legacy_orderbook(&mut rng, 100);
    group.bench_function("legacy_to_fast", |b| {
        b.iter(|| {
            let fast = FastOrderbook::from_legacy(&legacy);
            black_box(fast)
        })
    });

    // Benchmark: Convert fast back to legacy
    let fast = FastOrderbook::from_legacy(&legacy);
    group.bench_function("fast_to_legacy", |b| {
        b.iter(|| {
            let legacy = fast.to_legacy();
            black_box(legacy)
        })
    });

    group.finish();
}

/// Benchmark: COW (Copy-on-Write) overhead
fn bench_cow_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cow_overhead");
    let mut rng = StdRng::seed_from_u64(42);

    let legacy = generate_legacy_orderbook(&mut rng, 100);
    let fast = FastOrderbook::from_legacy(&legacy);

    // Benchmark: SharedOrderbook clone (cheap - Arc increment)
    let shared = SharedOrderbook::new(fast.clone());
    group.bench_function("shared_clone", |b| {
        b.iter(|| {
            let cloned = shared.clone();
            black_box(cloned)
        })
    });

    // Benchmark: SharedOrderbook get_mut when unique (no copy)
    group.bench_function("shared_get_mut_unique", |b| {
        b.iter_batched(
            || SharedOrderbook::new(fast.clone()),
            |mut ob| {
                let inner = ob.get_mut();
                inner.set_timestamp(12345);
                black_box(ob)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark: SharedOrderbook get_mut when shared (triggers copy)
    group.bench_function("shared_get_mut_copy", |b| {
        b.iter_batched(
            || {
                let shared = SharedOrderbook::new(fast.clone());
                let _clone = shared.clone(); // Create second reference
                shared
            },
            |mut ob| {
                let inner = ob.get_mut(); // Triggers copy
                inner.set_timestamp(12345);
                black_box(ob)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_orderbook_access,
    bench_orderbook_update,
    bench_price_lookup,
    bench_memory_allocation,
    bench_cow_overhead,
);

criterion_main!(benches);

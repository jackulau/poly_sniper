//! Benchmarks comparing broadcast vs lock-free event bus implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polysniper_core::{HeartbeatEvent, SystemEvent};
use polysniper_data::{BroadcastEventBus, LockFreeEventBus};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn create_test_event() -> SystemEvent {
    SystemEvent::Heartbeat(HeartbeatEvent {
        source: "bench".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

/// Benchmark single-threaded publish latency
fn bench_publish_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("publish_latency");
    group.throughput(Throughput::Elements(1));

    // Broadcast bus
    group.bench_function("broadcast", |b| {
        let bus = BroadcastEventBus::new();
        let _sub = bus.subscribe();
        b.iter(|| {
            bus.publish(black_box(create_test_event()));
        });
    });

    // Lock-free bus
    group.bench_function("lock_free", |b| {
        let bus = LockFreeEventBus::new();
        let _sub = bus.subscribe();
        b.iter(|| {
            bus.publish(black_box(create_test_event()));
        });
    });

    group.finish();
}

/// Benchmark high-throughput publish with multiple subscribers
fn bench_multi_subscriber_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_subscriber_throughput");

    for num_subscribers in [1, 2, 4, 8].iter() {
        group.throughput(Throughput::Elements(100));

        // Broadcast bus
        group.bench_with_input(
            BenchmarkId::new("broadcast", num_subscribers),
            num_subscribers,
            |b, &num_subs| {
                let bus = BroadcastEventBus::new();
                let _subs: Vec<_> = (0..num_subs).map(|_| bus.subscribe()).collect();
                b.iter(|| {
                    for _ in 0..100 {
                        bus.publish(black_box(create_test_event()));
                    }
                });
            },
        );

        // Lock-free bus
        group.bench_with_input(
            BenchmarkId::new("lock_free", num_subscribers),
            num_subscribers,
            |b, &num_subs| {
                let bus = LockFreeEventBus::new();
                let _subs: Vec<_> = (0..num_subs).map(|_| bus.subscribe()).collect();
                b.iter(|| {
                    for _ in 0..100 {
                        bus.publish(black_box(create_test_event()));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent publish and subscribe
fn bench_concurrent_pubsub(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_pubsub");
    group.measurement_time(Duration::from_secs(5));

    // Lock-free bus with concurrent consumers
    group.bench_function("lock_free_concurrent", |b| {
        b.iter_custom(|iters| {
            let bus = Arc::new(LockFreeEventBus::new());
            let num_events = iters as usize;

            // Create subscribers
            let subs: Vec<_> = (0..4).map(|_| bus.subscribe()).collect();

            // Spawn consumer threads
            let handles: Vec<_> = subs
                .into_iter()
                .map(|sub| {
                    thread::spawn(move || {
                        let mut count = 0;
                        while count < num_events {
                            if sub.try_recv().is_some() {
                                count += 1;
                            }
                            thread::yield_now();
                        }
                    })
                })
                .collect();

            // Time the publish phase
            let start = std::time::Instant::now();
            for _ in 0..num_events {
                bus.publish(create_test_event());
            }

            // Wait for consumers
            for handle in handles {
                handle.join().unwrap();
            }

            start.elapsed()
        });
    });

    group.finish();
}

/// Benchmark subscribe overhead
fn bench_subscribe_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("subscribe_overhead");

    group.bench_function("broadcast", |b| {
        let bus = BroadcastEventBus::new();
        b.iter(|| {
            let _sub = black_box(bus.subscribe());
        });
    });

    group.bench_function("lock_free", |b| {
        let bus = LockFreeEventBus::new();
        b.iter(|| {
            let _sub = black_box(bus.subscribe());
        });
    });

    group.finish();
}

/// Benchmark try_recv latency (non-blocking receive)
fn bench_try_recv_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("try_recv_latency");

    // Lock-free try_recv with event available
    group.bench_function("lock_free_with_event", |b| {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe();
        b.iter(|| {
            bus.publish(create_test_event());
            black_box(sub.try_recv())
        });
    });

    // Lock-free try_recv with empty queue
    group.bench_function("lock_free_empty", |b| {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe();
        b.iter(|| black_box(sub.try_recv()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_publish_latency,
    bench_multi_subscriber_throughput,
    bench_concurrent_pubsub,
    bench_subscribe_overhead,
    bench_try_recv_latency,
);

criterion_main!(benches);

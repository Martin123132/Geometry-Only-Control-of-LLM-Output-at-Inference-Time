from __future__ import annotations

import importlib.util
from pathlib import Path


def load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "benchmark_candidate_pool.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_candidate_pool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_frozen_exp20_ledger_has_expected_scale_shape():
    benchmark = load_benchmark_module()

    ledger = benchmark.load_ledger(benchmark.DEFAULT_LEDGER)

    assert len(ledger["candidates"]) == 257
    assert len(ledger["references"]) == 75
    assert len(ledger["cases"]) == 53


def test_candidate_pool_benchmark_reports_compression_timings_and_sizes():
    benchmark = load_benchmark_module()
    candidates = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is London.",
    ]

    report = benchmark.run_benchmark(
        candidates,
        ["The capital of France is Paris."],
        cases=1,
        iterations=1,
        warmup=0,
    )

    assert report["configuration"]["use_embeddings"] is False
    assert report["pool"]["total_candidates"] == 3
    assert report["pool"]["unique_pool_groups"] == 2
    assert report["pool"]["duplicate_candidates"] == 1
    assert report["pool"]["evaluations_avoided_by_duplicate_grouping"] == 1
    assert report["timings"]["regulation"]["samples"] == 1
    assert report["timings"]["regulation"]["median_submitted_candidates_per_second"] > 0
    assert report["timings"]["regulation"]["median_unique_pool_groups_per_second"] > 0
    assert report["memory"]["peak_bytes"] > 0
    assert report["report_sizes"]["json_bytes"] > 0
    assert report["report_sizes"]["markdown_bytes"] > 0
    assert report["report_sizes"]["csv_bytes"] > 0

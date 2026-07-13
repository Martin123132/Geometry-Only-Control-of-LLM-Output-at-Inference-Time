from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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


def test_paired_comparison_alternates_order_and_requires_exact_parity():
    benchmark = load_benchmark_module()
    calls = []
    candidates = [
        "The capital of France is Paris.",
        "The capital of France is London.",
    ]
    references = ["The capital of France is Paris."]

    def baseline(items, source, *, use_embeddings):
        calls.append("baseline")
        return benchmark.regulate_candidates(
            items,
            source,
            use_embeddings=use_embeddings,
        )

    def candidate(items, source, *, use_embeddings):
        calls.append("candidate")
        return benchmark.regulate_candidates(
            items,
            source,
            use_embeddings=use_embeddings,
        )

    report = benchmark.run_paired_comparison(
        candidates,
        references,
        baseline_regulator=baseline,
        candidate_regulator=candidate,
        pairs=4,
        warmup=0,
        baseline_label="before",
        candidate_label="after",
        max_regression_percent=1000.0,
    )

    assert calls == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert report["implementations"] == {
        "baseline": "before",
        "candidate": "after",
    }
    assert report["behavior"]["exact_report_parity"] is True
    assert report["behavior"]["mismatch_pairs"] == []
    assert report["timings"]["baseline"]["samples"] == 4
    assert report["timings"]["candidate"]["samples"] == 4
    assert len(report["paired"]["deltas_ms"]) == 4
    assert report["acceptance"]["accepted"] is True


def test_paired_comparison_rejects_behavior_mismatch():
    benchmark = load_benchmark_module()
    references = ["The capital of France is Paris."]

    def baseline(items, source, *, use_embeddings):
        return benchmark.regulate_candidates(
            items,
            source,
            use_embeddings=use_embeddings,
        )

    def changed(_items, source, *, use_embeddings):
        return benchmark.regulate_candidates(
            ["The capital of France is London."],
            source,
            use_embeddings=use_embeddings,
        )

    report = benchmark.run_paired_comparison(
        ["The capital of France is Paris."],
        references,
        baseline_regulator=baseline,
        candidate_regulator=changed,
        pairs=1,
        warmup=0,
        max_regression_percent=1000.0,
    )

    assert report["behavior"]["exact_report_parity"] is False
    assert report["behavior"]["mismatch_pairs"] == [0]
    assert report["acceptance"]["behavior_parity_pass"] is False
    assert report["acceptance"]["accepted"] is False


def test_load_callable_resolves_regulator_entrypoint():
    benchmark = load_benchmark_module()

    loaded = benchmark.load_callable(
        "mbt_ai_tools.mbt.regulator:regulate_candidates"
    )

    assert loaded is benchmark.regulate_candidates


def test_paired_comparison_rejects_same_callable_for_both_roles():
    benchmark = load_benchmark_module()

    with pytest.raises(ValueError, match="callables must be distinct"):
        benchmark.run_paired_comparison(
            ["The capital of France is Paris."],
            ["The capital of France is Paris."],
            baseline_regulator=benchmark.regulate_candidates,
            candidate_regulator=benchmark.regulate_candidates,
            pairs=1,
            warmup=0,
        )

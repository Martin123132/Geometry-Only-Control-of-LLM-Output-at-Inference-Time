from mbt_ai_tools import evaluate_candidate, extract_relations, regulate_candidates


def test_mixed_candidate_pool_emits_supported_candidate_without_embeddings():
    references = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
    ]

    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        references,
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The capital of France is Paris."
    assert result.evaluations[0].safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in result.evaluations[0].clamp_summary


def test_all_bad_candidate_pool_blocks_without_embeddings():
    references = ["The capital of France is Paris."]

    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Lyon.",
            "The capital of France is Berlin.",
        ],
        references,
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None


def test_relation_direction_and_shared_subject_coordination():
    references = [
        "Photosynthesis releases oxygen.",
        "Photosynthesis stores light energy.",
    ]

    assert ("photosynthesis", "release", "oxygen") in extract_relations(
        "Photosynthesis releases oxygen and stores light energy."
    )
    valid = evaluate_candidate(
        "Photosynthesis releases oxygen and stores light energy.",
        references,
        use_embeddings=False,
    )
    invalid = evaluate_candidate(
        "Photosynthesis converts oxygen into carbon dioxide and water.",
        references,
        use_embeddings=False,
    )

    assert valid.safe_to_emit is True
    assert invalid.safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in invalid.clamp_summary


def test_negated_positive_support_clamp():
    evaluation = evaluate_candidate(
        "Water is not liquid at room temperature.",
        ["Water is liquid at room temperature."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary

import builtins

import numpy as np
import pytest

from mbt_ai_tools import evaluate_candidate, extract_relations, regulate_candidates
from mbt_ai_tools.mbt.regulator import _extract_negated_relations


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


def test_candidate_pool_selection_diagnostics_rank_safe_alternatives():
    result = regulate_candidates(
        [
            "Paris is the capital city of France.",
            "The capital of France is Paris.",
            "The capital of France is London.",
        ],
        [
            "Paris is the capital city of France.",
            "The capital of France is Paris.",
        ],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Paris is the capital city of France."
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [
        True,
        True,
        False,
    ]
    assert [evaluation.selection_status for evaluation in result.evaluations] == [
        "emitted",
        "safe_alternative",
        "blocked_before_ranking",
    ]
    assert [evaluation.selection_rank for evaluation in result.evaluations] == [
        1,
        2,
        None,
    ]
    assert "lowest-score safe candidate" in result.evaluations[0].selection_reason
    assert "retained for audit" in result.evaluations[1].selection_reason
    assert "dropped from emission ranking" in result.evaluations[2].selection_reason


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


def test_negated_contraction_is_blocked_without_explicit_token_overlap():
    evaluation = evaluate_candidate(
        "Water isn't liquid at room temperature.",
        ["Water is liquid at room temperature."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_negated_auxiliary_verb_is_blocked():
    evaluation = evaluate_candidate(
        "The Sun does not orbit the Earth.",
        ["The Sun orbits the Earth."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_negated_contraction_without_space_is_normalized():
    evaluation = evaluate_candidate(
        "The Sun doesn't orbit the Earth.",
        ["The Sun orbits the Earth."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_negation_on_unrelated_clause_should_not_block_when_relation_not_supported():
    evaluation = evaluate_candidate(
        "Water is a liquid at room temperature and this is not the.",
        ["Water is liquid at room temperature."],
        use_embeddings=False,
    )

    assert "negated_positive_support_clamp" not in evaluation.clamp_summary
    assert evaluation.safe_to_emit is True


def test_negated_future_modal_is_blocked():
    evaluation = evaluate_candidate(
        "The Sun will not orbit the Earth.",
        ["The Sun orbits the Earth."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_negated_has_no_is_blocked():
    evaluation = evaluate_candidate(
        "Earth has no atmosphere.",
        ["Earth has an atmosphere."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_missing_sentence_transformers_dependency_is_communicated_in_regulator(monkeypatch):
    original_import = builtins.__import__

    def import_without_sentence_transformers(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "sentence_transformers":
            raise ModuleNotFoundError(
                "No module named 'sentence_transformers'", name="sentence_transformers"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", import_without_sentence_transformers)
    from mbt_ai_tools.mbt.embeddings import load_embedder

    load_embedder.cache_clear()

    with pytest.raises(ModuleNotFoundError, match="sentence-transformers is required"):
        evaluate_candidate("Water is liquid.", ["Water is liquid."])


def test_negated_positive_support_clamp():
    evaluation = evaluate_candidate(
        "Water is not liquid at room temperature.",
        ["Water is liquid at room temperature."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_supported_reference_negation_blocks_positive_counterclaim():
    result = regulate_candidates(
        [
            "The dataset contains city names and phone numbers.",
            "The dataset contains city names but no phone numbers.",
        ],
        ["The dataset contains city names but does not contain phone numbers."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The dataset contains city names but no phone numbers."
    assert result.evaluations[0].safe_to_emit is False
    assert "negated_reference_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True
    assert "negated_positive_support_clamp" not in result.evaluations[1].clamp_summary


def test_supported_reference_negation_keeps_exact_negative_reference_safe():
    evaluation = evaluate_candidate(
        "Mars has two moons and does not have a dense ring system.",
        ["Mars has two moons and does not have a dense ring system."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is True
    assert "exact_reference_member" in evaluation.clamp_summary
    assert "negated_positive_support_clamp" not in evaluation.clamp_summary
    assert "negated_reference_relation_clamp" not in evaluation.clamp_summary


def test_exclusion_supports_equivalent_negative_candidate():
    result = regulate_candidates(
        [
            "The trial included adults and children.",
            "The trial included adults and did not include children.",
        ],
        ["The trial included adults and excluded children."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The trial included adults and did not include children."
    assert result.evaluations[0].safe_to_emit is False
    assert "negated_reference_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_temporal_year_value_swap_is_blocked():
    result = regulate_candidates(
        [
            "In 2020, rainfall was 45 mm and in 2021 it was 30 mm.",
            "Rainfall was 30 mm in 2020 and 45 mm in 2021.",
        ],
        ["In 2020, rainfall was 30 mm. In 2021, rainfall was 45 mm."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Rainfall was 30 mm in 2020 and 45 mm in 2021."
    assert result.evaluations[0].safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_temporal_version_binding_swap_is_blocked():
    result = regulate_candidates(
        [
            "Version 1.2 added import support and version 1.3 added export support.",
            "Version 1.2 added export support and version 1.3 added import support.",
        ],
        ["Version 1.2 added export support. Version 1.3 added import support."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "Version 1.2 added export support and version 1.3 added import support."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_temporal_start_end_date_swap_is_blocked():
    result = regulate_candidates(
        [
            "The lease starts on August 31 and ends on March 1.",
            "The lease starts on March 1 and ends on August 31.",
        ],
        ["The lease starts on March 1 and ends on August 31."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The lease starts on March 1 and ends on August 31."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_relation_composition_shared_subject_chain_is_blocked():
    references = [
        "Photosynthesis uses carbon dioxide.",
        "Photosynthesis releases oxygen.",
        "Photosynthesis stores chemical energy.",
    ]

    assert ("photosynthesis", "use", "carbon dioxide") in extract_relations(
        "Photosynthesis uses carbon dioxide, releases oxygen, and stores chemical energy."
    )
    assert ("photosynthesis", "release", "oxygen") in extract_relations(
        "Photosynthesis uses carbon dioxide, releases oxygen, and stores chemical energy."
    )
    result = regulate_candidates(
        [
            "Photosynthesis releases carbon dioxide and uses oxygen.",
            "Photosynthesis uses carbon dioxide, releases oxygen, and stores chemical energy.",
        ],
        references,
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "Photosynthesis uses carbon dioxide, releases oxygen, and stores chemical energy."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_relation_composition_cross_subject_verb_swap_is_blocked():
    result = regulate_candidates(
        [
            "Libraries preserve records and archives lend books.",
            "Libraries lend books and archives preserve records.",
        ],
        [
            "Libraries lend books.",
            "Archives preserve records.",
        ],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Libraries lend books and archives preserve records."
    assert result.evaluations[0].safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_relation_composition_transform_swap_is_blocked():
    result = regulate_candidates(
        [
            "Evaporation turns vapor into liquid water and condensation turns liquid water into vapor.",
            "Evaporation turns liquid water into vapor and condensation turns vapor into liquid water.",
        ],
        [
            "Evaporation turns liquid water into vapor.",
            "Condensation turns vapor into liquid water.",
        ],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "Evaporation turns liquid water into vapor and condensation turns vapor into liquid water."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_negation_scope_gate_until_binding_is_blocked():
    result = regulate_candidates(
        [
            "The south gate opens at 08:00 and the north gate opens at 10:00.",
            "The north gate opens at 08:00 and the south gate waits until 10:00.",
        ],
        ["The north gate opens at 08:00; the south gate does not open until 10:00."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "The north gate opens at 08:00 and the south gate waits until 10:00."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_negation_scope_sensor_ellipsis_preserves_lettered_subjects():
    result = regulate_candidates(
        [
            "Sensor B measures humidity and Sensor A does not.",
            "Sensor A measures humidity and Sensor B does not.",
        ],
        ["Sensor A measures humidity; Sensor B does not measure humidity."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Sensor A measures humidity and Sensor B does not."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_negation_scope_axis_ellipsis_is_bound_to_prior_event():
    result = regulate_candidates(
        [
            "Axis Y moved during calibration while Axis X did not.",
            "Axis X moved during calibration while Axis Y did not.",
        ],
        ["Axis X moved during calibration; Axis Y did not move."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Axis X moved during calibration while Axis Y did not."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_supported_negation_policy_compact_form_blocks_positive_reversal():
    result = regulate_candidates(
        [
            "The policy allows remote work and overseas payroll.",
            "The policy allows remote work but not overseas payroll.",
            "The policy blocks remote work.",
        ],
        ["The policy allows remote work but does not allow overseas payroll."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The policy allows remote work but not overseas payroll."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True
    assert result.evaluations[2].safe_to_emit is False


def test_supported_negation_prepositional_contrast_object_is_preserved():
    result = regulate_candidates(
        [
            "The medicine is approved for adults and children.",
            "The medicine is approved for adults, not children.",
            "The medicine is not approved for adults.",
        ],
        ["The medicine is approved for adults and is not approved for children."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The medicine is approved for adults, not children."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True
    assert result.evaluations[2].safe_to_emit is False


def test_alias_binding_acquisition_direction_is_preserved():
    result = regulate_candidates(
        [
            "Intel acquired Red Hat.",
            "IBM acquired Red Hat.",
            "Red Hat acquired IBM.",
        ],
        ["International Business Machines, also called IBM, acquired Red Hat."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "IBM acquired Red Hat."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True
    assert result.evaluations[2].safe_to_emit is False


def test_alias_binding_location_direction_is_preserved():
    result = regulate_candidates(
        [
            "NYC is in California.",
            "NYC is in New York State.",
            "New York State is in NYC.",
        ],
        ["New York City, abbreviated NYC, is in New York State."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "NYC is in New York State."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True
    assert result.evaluations[2].safe_to_emit is False


def test_qualifier_overclaim_sea_level_scope_is_preserved():
    result = regulate_candidates(
        [
            "Water always boils at 100 degrees Celsius.",
            "At sea level, water boils at 100 degrees Celsius.",
        ],
        ["Water boils at 100 degrees Celsius at sea level."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "At sea level, water boils at 100 degrees Celsius."
    assert result.evaluations[0].safe_to_emit is False
    assert "overclaim_flag" in result.evaluations[0].clamp_summary
    assert result.evaluations[1].safe_to_emit is True


def test_temporal_office_opened_dates_preserve_subject_binding():
    result = regulate_candidates(
        [
            "The Paris office opened in 2018 and the Berlin office opened in 2020.",
            "The Berlin office opened in 2018 and the Paris office opened in 2020.",
        ],
        ["The Berlin office opened in 2018. The Paris office opened in 2020."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "The Berlin office opened in 2018 and the Paris office opened in 2020."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_pronoun_review_binding_preserves_original_object():
    result = regulate_candidates(
        [
            "Linus wrote the parser and Ada reviewed it.",
            "Ada wrote the parser and Linus reviewed it.",
        ],
        ["Ada wrote the parser. Linus reviewed the parser."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Ada wrote the parser and Linus reviewed it."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_lettered_tank_binding_preserves_valve_direction():
    assert ("blue valve", "feed", "tank a") in extract_relations(
        "The blue valve feeds Tank A and the red valve feeds Tank B."
    )

    result = regulate_candidates(
        [
            "The red valve feeds Tank A and the blue valve feeds Tank B.",
            "The blue valve feeds Tank A and the red valve feeds Tank B.",
        ],
        ["The blue valve feeds Tank A. The red valve feeds Tank B."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The blue valve feeds Tank A and the red valve feeds Tank B."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_supervision_chain_preserves_role_order():
    result = regulate_candidates(
        [
            "Omar supervises Kim and Kim supervises Maya.",
            "Maya supervises Kim and Kim supervises Omar.",
        ],
        ["Maya supervises Kim. Kim supervises Omar."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Maya supervises Kim and Kim supervises Omar."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_all_bad_numbered_boiler_swaps_block():
    result = regulate_candidates(
        [
            "The left pump feeds Boiler 2.",
            "The right pump feeds Boiler 1.",
            "Boiler 1 feeds the left pump.",
        ],
        ["The left pump feeds Boiler 1. The right pump feeds Boiler 2."],
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [
        False,
        False,
        False,
    ]


def test_exp22_plan_cost_comparison_preserves_lower_value_direction():
    assert ("plan a", "cheaperthan", "plan b") in extract_relations(
        "Plan A costs 40 dollars. Plan B costs 55 dollars."
    )

    result = regulate_candidates(
        [
            "Plan B is cheaper than Plan A.",
            "Plan A is cheaper than Plan B.",
        ],
        ["Plan A costs 40 dollars. Plan B costs 55 dollars."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Plan A is cheaper than Plan B."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_model_latency_comparison_preserves_lower_latency_direction():
    result = regulate_candidates(
        [
            "Model Y is faster than Model X.",
            "Model X is faster than Model Y.",
        ],
        ["Model X has 80 ms latency. Model Y has 120 ms latency."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Model X is faster than Model Y."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_battery_capacity_comparison_preserves_higher_value_direction():
    result = regulate_candidates(
        [
            "Battery Beta has higher capacity than Battery Alpha.",
            "Battery Alpha has higher capacity than Battery Beta.",
        ],
        ["Battery Alpha stores 3000 mAh. Battery Beta stores 2500 mAh."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Battery Alpha has higher capacity than Battery Beta."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_weekday_exception_preserves_closed_day_scope():
    result = regulate_candidates(
        [
            "The museum is open on Monday.",
            "The museum is closed on Monday and open on other weekdays.",
        ],
        ["The museum is open every weekday except Monday."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The museum is closed on Monday and open on other weekdays."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_file_format_exception_blocks_excluded_encrypted_json():
    result = regulate_candidates(
        [
            "The importer accepts encrypted JSON files.",
            "The importer accepts CSV and JSON files but not encrypted JSON.",
        ],
        ["The importer accepts CSV and JSON files, except encrypted JSON."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The importer accepts CSV and JSON files but not encrypted JSON."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_all_bad_negated_form_exception_blocks_reversals():
    result = regulate_candidates(
        [
            "The form requires a witness and does not require a signature.",
            "The form does not require a signature.",
        ],
        ["The form requires a signature and does not require a witness."],
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [False, False]


def test_exp22_backup_migration_temporal_order_blocks_reversal():
    result = regulate_candidates(
        [
            "The backup completed before the migration.",
            "The backup completed after the migration.",
        ],
        ["The backup completed after the migration."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The backup completed after the migration."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_inspection_repair_temporal_order_preserves_event_binding():
    result = regulate_candidates(
        [
            "The repair happened before launch and the inspection happened after launch.",
            "The inspection happened before launch and the repair happened after launch.",
        ],
        ["The inspection happened before launch and the repair happened after launch."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "The inspection happened before launch and the repair happened after launch."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp22_all_bad_alias_role_swap_blocks_report_as_actor():
    result = regulate_candidates(
        [
            "Dr. Chen signed the report.",
            "The report signed Mira Rao.",
        ],
        ["Dr. Rao, also called Mira Rao, signed the report."],
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [False, False]


@pytest.mark.parametrize(
    "reference,candidate",
    [
        ("Water is liquid at room temperature.", "Water is not liquid at room temperature."),
        ("Water is liquid at room temperature.", "Water isn't liquid at room temperature."),
        ("The Sun orbits the Earth.", "The Sun does not orbit the Earth."),
        ("The Sun orbits the Earth.", "The Sun doesn't orbit the Earth."),
        ("The Sun orbits the Earth.", "The Sun will not orbit the Earth."),
        ("Earth has an atmosphere.", "Earth has no atmosphere."),
        ("Plants use sunlight.", "Plants do not use sunlight."),
        ("Plants use sunlight.", "Plants won't use sunlight."),
        ("DNA contains genes.", "DNA cannot contain genes."),
        ("Mammals need oxygen.", "Mammals shouldn't need oxygen."),
    ],
)
def test_negations_without_embeddings_remain_blocked(reference, candidate):
    evaluation = evaluate_candidate(
        candidate,
        [reference],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "negated_positive_support_clamp" in evaluation.clamp_summary


def test_multi_word_relation_direction_swap_is_blocked():
    evaluation = evaluate_candidate(
        "Sunlight uses solar panels.",
        ["Solar panels use sunlight."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "known_participant_unsupported_relation_clamp" in evaluation.clamp_summary


@pytest.mark.parametrize(
    "reference,candidate",
    [
        ("The model output is a candidate result.", "The model output is the final truth."),
        (
            "The method has preliminary support.",
            "The method is complete and experimentally verified.",
        ),
        ("The answer is a candidate result.", "The answer is automatically the final answer."),
    ],
)
def test_overclaim_boundaries_block_without_embeddings(reference, candidate):
    evaluation = evaluate_candidate(
        candidate,
        [reference],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is False
    assert "overclaim_flag" in evaluation.clamp_summary


def test_supported_capital_paraphrase_reference_no_false_relation_clamp():
    evaluation = evaluate_candidate(
        "Paris is the capital city of France.",
        ["The capital of France is Paris."],
        use_embeddings=False,
    )

    assert evaluation.safe_to_emit is True
    assert "known_participant_unsupported_relation_clamp" not in evaluation.clamp_summary


def test_regulate_candidates_with_embedded_manifold_smoke(monkeypatch):
    def fake_embed_texts(texts, *args, **kwargs):
        outputs = []
        for text in texts:
            if text == "The capital of France is Paris.":
                outputs.append(np.array([0.0, 0.0, 0.0]))
            else:
                outputs.append(np.array([4.0, 0.0, 0.0]))
        return np.asarray(outputs)

    monkeypatch.setattr("mbt_ai_tools.mbt.regulator.embed_texts", fake_embed_texts)

    result = regulate_candidates(
        ["The capital of France is Paris.", "The capital of France is London."],
        ["The capital of France is Paris."],
    )

    assert result.action == "emit"
    assert result.emitted_text == "The capital of France is Paris."


def test_multi_word_capital_relations_are_not_truncated():
    assert ("mexico", "capital", "mexico city") in extract_relations(
        "The capital of Mexico is Mexico City."
    )
    assert ("argentina", "capital", "buenos aires") in extract_relations(
        "Buenos Aires is the capital city of Argentina."
    )


def test_exp23_runner_finish_order_preserves_rank_binding():
    result = regulate_candidates(
        [
            "Runner B finished first and Runner A finished second.",
            "Runner A finished first and Runner B finished second.",
        ],
        ["Runner A finished first. Runner B finished second."],
        use_embeddings=False,
    )

    assert ("runner a", "ordinalrank", "first") in extract_relations(
        "Runner A finished first. Runner B finished second."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Runner A finished first and Runner B finished second."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_priority_labels_preserve_number_binding():
    result = regulate_candidates(
        [
            "Priority 2 is critical and Priority 1 is warning.",
            "Priority 1 is critical and Priority 2 is warning.",
        ],
        [
            "Priority 1 is critical. Priority 2 is warning. "
            "Priority 3 is informational."
        ],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Priority 1 is critical and Priority 2 is warning."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_pipeline_stage_values_preserve_ordinal_binding():
    result = regulate_candidates(
        [
            "The first stage is review and the second stage is intake.",
            "The first stage is intake and the second stage is review.",
        ],
        [
            "The first stage is intake. The second stage is review. "
            "The third stage is approval."
        ],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The first stage is intake and the second stage is review."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_lab_exception_preserves_excluded_interns():
    result = regulate_candidates(
        [
            "Interns can enter the lab.",
            "Interns cannot enter the lab, and supervisors can enter after hours.",
        ],
        ["All staff can enter the lab except interns; supervisors can enter after hours."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == (
        "Interns cannot enter the lab, and supervisors can enter after hours."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_animated_gif_exception_preserves_static_allowance():
    result = regulate_candidates(
        [
            "The importer accepts animated GIFs.",
            "The importer accepts static GIFs but not animated GIFs.",
        ],
        ["The importer accepts images except animated GIFs; it accepts static GIFs."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The importer accepts static GIFs but not animated GIFs."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_clinic_exception_preserves_walkin_exclusion():
    result = regulate_candidates(
        [
            "The clinic treats all adults including emergency walk-ins.",
            "The clinic treats adults except emergency walk-ins.",
        ],
        ["The clinic treats adults except emergency walk-ins; children require referral."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The clinic treats adults except emergency walk-ins."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_alarm_conditional_blocks_unconditional_lock_claim():
    result = regulate_candidates(
        [
            "The door lock always engages.",
            "The door lock engages when the alarm is armed.",
        ],
        ["If the alarm is armed, the door lock engages."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The door lock engages when the alarm is armed."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_switch_motor_all_bad_conditionals_block():
    result = regulate_candidates(
        [
            "The motor stops when the switch is on.",
            "The switch stops when the motor is off.",
        ],
        ["If the switch is off, the motor stops."],
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [False, False]


def test_exp24_card_pin_multi_antecedent_condition_preserves_joint_scope():
    result = regulate_candidates(
        [
            "The door opens when the card is active.",
            "The door opens when the card is active and the PIN is correct.",
        ],
        ["If the card is active and the PIN is correct, the door opens."],
        use_embeddings=False,
    )

    assert ("door", "openwhen", "cardactivepincorrect") in extract_relations(
        "If the card is active and the PIN is correct, the door opens."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The door opens when the card is active and the PIN is correct."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_valve_pressure_multi_antecedent_condition_preserves_joint_scope():
    result = regulate_candidates(
        [
            "The pump starts when pressure is below 5 bar.",
            "The pump starts when the valve is open and pressure is below 5 bar.",
        ],
        ["If the valve is open and pressure is below 5 bar, the pump starts."],
        use_embeddings=False,
    )

    assert ("pump", "startwhen", "valveopenpressurebelow5bar") in extract_relations(
        "If the valve is open and pressure is below 5 bar, the pump starts."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "The pump starts when the valve is open and pressure is below 5 bar."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_admin_owner_permission_preserves_allowed_roles():
    result = regulate_candidates(
        [
            "Viewers may export the file.",
            "Admins and owners may export the file.",
        ],
        ["If the user is an admin or owner, export is allowed."],
        use_embeddings=False,
    )

    assert ("admin", "mayexport", "file") in extract_relations(
        "If the user is an admin or owner, export is allowed."
    )
    assert ("owner", "mayexport", "file") in extract_relations(
        "Admins and owners may export the file."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Admins and owners may export the file."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_rao_alias_permission_preserves_approver_binding():
    result = regulate_candidates(
        [
            "Lee may approve invoices.",
            "Dr. Rao may approve invoices.",
        ],
        ["Mira Rao, also called Dr. Rao, may approve invoices. Lee may view invoices."],
        use_embeddings=False,
    )

    assert ("dr rao", "mayapprove", "invoices") in extract_relations(
        "Mira Rao, also called Dr. Rao, may approve invoices. Lee may view invoices."
    )
    assert ("lee", "mayview", "invoices") in extract_relations(
        "Mira Rao, also called Dr. Rao, may approve invoices. Lee may view invoices."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Dr. Rao may approve invoices."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_backup_alias_permission_preserves_device_binding():
    result = regulate_candidates(
        [
            "Device B may restart the router.",
            "Device A may restart the router.",
        ],
        ["Device A is the backup unit. The backup unit may restart the router."],
        use_embeddings=False,
    )

    assert ("device a", "mayrestart", "router") in extract_relations(
        "Device A is the backup unit. The backup unit may restart the router."
    )
    assert ("device b", "mayrestart", "router") in extract_relations(
        "Device B may restart the router."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Device A may restart the router."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_ledger_permission_scope_blocks_wrong_actor_and_direction():
    result = regulate_candidates(
        [
            "Team Blue may edit the ledger.",
            "The ledger may edit Team Red.",
        ],
        ["Only Team Red may edit the ledger; Team Blue may view it."],
        use_embeddings=False,
    )

    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [False, False]


def test_exp23_trial_aggregate_binding_preserves_group_counts():
    result = regulate_candidates(
        [
            "The trial enrolled 10 adults and 40 children.",
            "The trial enrolled 40 adults and 10 children.",
        ],
        ["The trial enrolled 40 adults and 10 children."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "The trial enrolled 40 adults and 10 children."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp23_applicant_age_range_preserves_under_bound_paraphrase():
    result = regulate_candidates(
        [
            "Applicants aged 17 are eligible.",
            "Applicants aged 18 through 64 are eligible.",
        ],
        ["Applicants must be at least 18 and under 65."],
        use_embeddings=False,
    )

    assert result.action == "emit"
    assert result.emitted_text == "Applicants aged 18 through 64 are eligible."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_tank_liter_to_milliliter_bound_preserves_upper_limit():
    result = regulate_candidates(
        [
            "The tank holds 2500 milliliters.",
            "The tank holds up to 2000 milliliters.",
        ],
        ["The tank holds at most 2 liters."],
        use_embeddings=False,
    )

    assert ("tank", "holdsatmost", "2000milliliters") in extract_relations(
        "The tank holds at most 2 liters."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The tank holds up to 2000 milliliters."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_cable_meter_to_centimeter_bound_preserves_exclusive_limit():
    result = regulate_candidates(
        [
            "The cable may be 300 centimeters long.",
            "The cable must be less than 300 centimeters long.",
        ],
        ["The cable must be shorter than 3 meters."],
        use_embeddings=False,
    )

    assert ("cable", "maxlengthexclusive", "300centimeters") in extract_relations(
        "The cable must be shorter than 3 meters."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The cable must be less than 300 centimeters long."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_service_window_preserves_exclusive_end_time():
    result = regulate_candidates(
        [
            "The service window includes 17:00.",
            "The service window runs from 09:00 until before 17:00.",
        ],
        ["The service window starts at 09:00 and ends before 17:00."],
        use_embeddings=False,
    )

    assert ("service window", "endsbefore", "1700") in extract_relations(
        "The service window starts at 09:00 and ends before 17:00."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The service window runs from 09:00 until before 17:00."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_employee_exception_chain_preserves_multiple_exclusions():
    result = regulate_candidates(
        [
            "Contractors may enter.",
            "Contractors and interns may not enter, and security may enter after hours.",
        ],
        ["All employees may enter except contractors and interns; security may enter after hours."],
        use_embeddings=False,
    )

    assert ("contractors", "is", "enter") in extract_relations("Contractors may enter.")
    assert ("contractors", "exceptfrom", "enter") in extract_relations(
        "All employees may enter except contractors and interns; security may enter after hours."
    )
    assert ("interns", "exceptfrom", "enter") in extract_relations(
        "All employees may enter except contractors and interns; security may enter after hours."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "Contractors and interns may not enter, and security may enter after hours."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_image_exception_chain_preserves_png_allowance_and_exclusions():
    result = regulate_candidates(
        [
            "The importer accepts SVG files.",
            "The importer accepts PNG files but not SVG or animated GIFs.",
        ],
        ["The importer accepts images except SVG and animated GIFs; it accepts PNG."],
        use_embeddings=False,
    )

    assert ("importer", "accept", "svg") in extract_relations(
        "The importer accepts SVG files."
    )
    assert ("importer", "exceptfrom", "animated gifs") in extract_relations(
        "The importer accepts images except SVG and animated GIFs; it accepts PNG."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The importer accepts PNG files but not SVG or animated GIFs."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_clinic_exception_chain_preserves_multiple_visitor_exclusions():
    result = regulate_candidates(
        [
            "The clinic treats all adults including walk-ins.",
            "The clinic treats adults except walk-ins and unregistered visitors.",
        ],
        ["The clinic treats adults except walk-ins and unregistered visitors."],
        use_embeddings=False,
    )

    assert ("clinic", "treat", "walkins") in extract_relations(
        "The clinic treats all adults including walk-ins."
    )
    assert ("clinic", "exceptfrom", "unregistered visitors") in extract_relations(
        "The clinic treats adults except walk-ins and unregistered visitors."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "The clinic treats adults except walk-ins and unregistered visitors."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_sensor_monitoring_compact_binding_preserves_assignments():
    result = regulate_candidates(
        [
            "Sensor red monitors humidity and sensor blue monitors heat.",
            "Sensor red monitors heat and sensor blue monitors humidity.",
        ],
        ["Sensor red monitors heat and sensor blue monitors humidity."],
        use_embeddings=False,
    )

    assert ("sensor red", "monitor", "heat") in extract_relations(
        "Sensor red monitors heat and sensor blue monitors humidity."
    )
    assert ("sensor red", "monitor", "humidity") in extract_relations(
        "Sensor red monitors humidity and sensor blue monitors heat."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Sensor red monitors heat and sensor blue monitors humidity."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp24_dose_range_all_bad_reuse_blocks_boundary_candidates():
    result = regulate_candidates(
        [
            "A 4 mg dose is allowed.",
            "A 10 mg dose is allowed.",
        ],
        ["The dose must be at least 5 mg and below 10 mg."],
        use_embeddings=False,
    )

    assert ("dose", "allowedrange", "gte5mglt10mg") in extract_relations(
        "The dose must be at least 5 mg and below 10 mg."
    )
    assert ("dose", "allowedamount", "10mg") in extract_relations(
        "A 10 mg dose is allowed."
    )
    assert result.action == "block"
    assert result.emitted_text is None
    assert [evaluation.safe_to_emit for evaluation in result.evaluations] == [False, False]


def test_exp25_container_dimension_binding_preserves_weight_and_volume():
    result = regulate_candidates(
        [
            "The container weighs 2 kg and holds 3 liters.",
            "The container weighs 3 kg and holds 2 liters.",
        ],
        ["The container weighs 3 kg and holds 2 liters."],
        use_embeddings=False,
    )

    assert ("container", "weighs", "3kg") in extract_relations(
        "The container weighs 3 kg and holds 2 liters."
    )
    assert ("container", "holds", "2liters") in extract_relations(
        "The container weighs 3 kg and holds 2 liters."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The container weighs 3 kg and holds 2 liters."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_drone_dimension_binding_preserves_range_and_payload():
    result = regulate_candidates(
        [
            "The drone flies 4 km and carries 12 kg.",
            "The drone flies 12 km and carries 4 kg.",
        ],
        ["The drone flies 12 km and carries 4 kg."],
        use_embeddings=False,
    )

    assert ("drone", "flies", "12km") in extract_relations(
        "The drone flies 12 km and carries 4 kg."
    )
    assert ("drone", "carries", "4kg") in extract_relations(
        "The drone flies 12 km and carries 4 kg."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The drone flies 12 km and carries 4 kg."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_battery_dimension_binding_preserves_power_and_duration():
    result = regulate_candidates(
        [
            "The battery charges at 5 watts and lasts 20 hours.",
            "The battery charges at 20 watts and lasts 5 hours.",
        ],
        ["The battery charges at 20 watts and lasts 5 hours."],
        use_embeddings=False,
    )

    assert ("battery", "chargesat", "20watts") in extract_relations(
        "The battery charges at 20 watts and lasts 5 hours."
    )
    assert ("battery", "lasts", "5hours") in extract_relations(
        "The battery charges at 20 watts and lasts 5 hours."
    )
    assert result.action == "emit"
    assert result.emitted_text == "The battery charges at 20 watts and lasts 5 hours."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_reviewer_threshold_preserves_minimum_approval_count():
    result = regulate_candidates(
        [
            "One reviewer approval is enough for release.",
            "Release requires approval from two or more reviewers.",
        ],
        ["At least two reviewers must approve before release."],
        use_embeddings=False,
    )

    assert ("release", "minapprovals", "2reviewers") in extract_relations(
        "At least two reviewers must approve before release."
    )
    assert ("release", "minapprovals", "1reviewer") in extract_relations(
        "One reviewer approval is enough for release."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Release requires approval from two or more reviewers."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_failed_check_threshold_preserves_maximum_count():
    result = regulate_candidates(
        [
            "Two failed checks are allowed.",
            "At most one failed check is allowed.",
        ],
        ["No more than one failed check is allowed."],
        use_embeddings=False,
    )

    assert ("failed check", "maxallowed", "1") in extract_relations(
        "No more than one failed check is allowed."
    )
    assert ("failed check", "allowedcount", "2") in extract_relations(
        "Two failed checks are allowed."
    )
    assert result.action == "emit"
    assert result.emitted_text == "At most one failed check is allowed."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_signature_threshold_preserves_exact_count():
    result = regulate_candidates(
        [
            "Two signatures are required.",
            "Three signatures are required.",
        ],
        ["Exactly three signatures are required."],
        use_embeddings=False,
    )

    assert ("signatures", "exactrequired", "3") in extract_relations(
        "Exactly three signatures are required."
    )
    assert ("signatures", "exactrequired", "2") in extract_relations(
        "Two signatures are required."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Three signatures are required."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_badge_room_nested_condition_preserves_both_antecedents():
    result = regulate_candidates(
        [
            "The robot enters when the badge is valid.",
            "The robot enters when the badge is valid and the room is unlocked.",
        ],
        ["If the badge is valid, then if the room is unlocked, the robot enters."],
        use_embeddings=False,
    )

    assert ("robot", "enterwhen", "badgevalidroomunlocked") in extract_relations(
        "If the badge is valid, then if the room is unlocked, the robot enters."
    )
    assert ("robot", "enterwhen", "badgevalid") in extract_relations(
        "The robot enters when the badge is valid."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "The robot enters when the badge is valid and the room is unlocked."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_sensor_alarm_nested_condition_preserves_both_antecedents():
    result = regulate_candidates(
        [
            "The alarm stays off when sensor B is silent.",
            "The alarm stays off when sensor A is armed and sensor B is silent.",
        ],
        ["If sensor A is armed, then if sensor B is silent, the alarm stays off."],
        use_embeddings=False,
    )

    assert ("alarm", "staysoffwhen", "sensoraarmedsensorbsilent") in extract_relations(
        "If sensor A is armed, then if sensor B is silent, the alarm stays off."
    )
    assert ("alarm", "staysoffwhen", "sensorbsilent") in extract_relations(
        "The alarm stays off when sensor B is silent."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "The alarm stays off when sensor A is armed and sensor B is silent."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_verified_owner_nested_condition_preserves_role_scope():
    result = regulate_candidates(
        [
            "Verified viewers may export.",
            "Verified owners may export.",
        ],
        [
            "If the account is verified, then if the requester is an owner, export is allowed."
        ],
        use_embeddings=False,
    )

    assert ("verified owner", "mayexport", "export") in extract_relations(
        "If the account is verified, then if the requester is an owner, export is allowed."
    )
    assert ("verified viewer", "mayexport", "export") in extract_relations(
        "Verified viewers may export."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Verified owners may export."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_exporter_scoped_exception_preserves_encrypted_csv_denial():
    result = regulate_candidates(
        [
            "Encrypted CSV is supported.",
            "JSON is supported, but encrypted CSV is not.",
        ],
        ["The exporter supports CSV except encrypted CSV; JSON is supported."],
        use_embeddings=False,
    )

    assert ("exporter", "exceptfrom", "encrypted csv") in extract_relations(
        "The exporter supports CSV except encrypted CSV; JSON is supported."
    )
    assert ("encrypted csv", "is", "supported") in _extract_negated_relations(
        "The exporter supports CSV except encrypted CSV; JSON is supported."
    )
    assert result.action == "emit"
    assert result.emitted_text == "JSON is supported, but encrypted CSV is not."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_staff_scoped_exception_preserves_auditor_escort_scope():
    result = regulate_candidates(
        [
            "Visitors may enter with escort.",
            "Visitors and contractors may not enter, and auditors may enter with escort.",
        ],
        [
            "All staff may enter except visitors and contractors; auditors may enter with escort."
        ],
        use_embeddings=False,
    )

    assert ("visitors", "is", "enter") in _extract_negated_relations(
        "All staff may enter except visitors and contractors; auditors may enter with escort."
    )
    assert ("auditors", "is", "enter with escort") in extract_relations(
        "All staff may enter except visitors and contractors; auditors may enter with escort."
    )
    assert result.action == "emit"
    assert result.emitted_text == (
        "Visitors and contractors may not enter, and auditors may enter with escort."
    )
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_valve_pump_temporal_role_binding_preserves_actor_object_order():
    result = regulate_candidates(
        [
            "Mira repaired the pump before Sol inspected the valve.",
            "Sol repaired the pump before Mira inspected the valve.",
        ],
        ["Sol repaired the pump before Mira inspected the valve."],
        use_embeddings=False,
    )

    assert ("sol", "repairbefore", "pump") in extract_relations(
        "Sol repaired the pump before Mira inspected the valve."
    )
    assert ("mira", "inspectafter", "valve") in extract_relations(
        "Mira inspected the valve after Sol repaired the pump."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Sol repaired the pump before Mira inspected the valve."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_draft_temporal_role_binding_preserves_approve_reject_order():
    result = regulate_candidates(
        [
            "Ana rejected draft A after Jo approved draft B.",
            "Jo rejected draft A before Ana approved draft B.",
        ],
        ["Ana approved draft B after Jo rejected draft A."],
        use_embeddings=False,
    )

    assert ("ana", "approveafter", "drafta") not in extract_relations(
        "Jo rejected draft A before Ana approved draft B."
    )
    assert ("ana", "approveafter", "draftb") in extract_relations(
        "Ana approved draft B after Jo rejected draft A."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Jo rejected draft A before Ana approved draft B."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_vial_temporal_role_binding_preserves_label_seal_order():
    result = regulate_candidates(
        [
            "Noor sealed vial 2 after Iris labeled vial 1.",
            "Noor labeled vial 1 before Iris sealed vial 2.",
        ],
        ["Iris sealed vial 2 after Noor labeled vial 1."],
        use_embeddings=False,
    )

    assert ("iris", "sealafter", "vial 2") in extract_relations(
        "Iris sealed vial 2 after Noor labeled vial 1."
    )
    assert ("noor", "labelbefore", "vial 1") in extract_relations(
        "Noor labeled vial 1 before Iris sealed vial 2."
    )
    assert result.action == "emit"
    assert result.emitted_text == "Noor labeled vial 1 before Iris sealed vial 2."
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is True


def test_exp25_route_dock_all_bad_token_reuse_blocks_swaps():
    result = regulate_candidates(
        [
            "Route A goes to dock 7 and Route B goes to dock 4.",
            "Dock 4 goes to Route B.",
        ],
        ["Route A goes to dock 4; Route B goes to dock 7."],
        use_embeddings=False,
    )

    assert ("routea", "goesto", "dock4") in extract_relations(
        "Route A goes to dock 4; Route B goes to dock 7."
    )
    assert ("routea", "goesto", "dock7") in extract_relations(
        "Route A goes to dock 7 and Route B goes to dock 4."
    )
    assert result.action == "block"
    assert result.emitted_text is None
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is False


def test_exp25_service_deploy_all_bad_token_reuse_blocks_permission_swaps():
    result = regulate_candidates(
        [
            "Team Red may deploy service X.",
            "Service X may deploy Team Blue.",
        ],
        ["Only Team Blue may deploy service X; Team Red may monitor it."],
        use_embeddings=False,
    )

    assert ("teamblue", "maydeploy", "servicex") in extract_relations(
        "Only Team Blue may deploy service X; Team Red may monitor it."
    )
    assert ("teamred", "maydeploy", "servicex") in extract_relations(
        "Team Red may deploy service X."
    )
    assert result.action == "block"
    assert result.emitted_text is None
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is False


def test_exp25_freezer_all_bad_token_reuse_blocks_threshold_swaps():
    result = regulate_candidates(
        [
            "Freezer A must stay above 2 C and Freezer B below -10 C.",
            "Freezer B must stay below -10 C.",
        ],
        ["Freezer A must stay below -10 C; Freezer B must stay above 2 C."],
        use_embeddings=False,
    )

    assert ("freezera", "staybelow", "10c") in extract_relations(
        "Freezer A must stay below -10 C; Freezer B must stay above 2 C."
    )
    assert ("freezerb", "staybelow", "10c") in extract_relations(
        "Freezer B must stay below -10 C."
    )
    assert result.action == "block"
    assert result.emitted_text is None
    assert result.evaluations[0].safe_to_emit is False
    assert result.evaluations[1].safe_to_emit is False

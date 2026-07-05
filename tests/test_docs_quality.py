from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest


def load_docs_quality_module():
    module_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "docs_quality.py"
    )
    spec = importlib.util.spec_from_file_location("docs_quality", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def docs_quality():
    return load_docs_quality_module()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_pyproject(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_changelog(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_manifest(tmp_path: Path) -> Path:
    manifest = {
        "project": {
            "name": "manifold-guard",
            "package_urls": {
                "homepage": "https://github.com/Martin123132/ManifoldGuard",
                "source": "https://github.com/Martin123132/ManifoldGuard",
                "documentation": "https://github.com/Martin123132/ManifoldGuard/blob/main/README.md",
                "issues": "https://github.com/Martin123132/ManifoldGuard/issues",
                "changelog": "https://github.com/Martin123132/ManifoldGuard/blob/main/CHANGELOG.md",
            },
        }
    }
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def parse_ci_jobs(workflow_text: str) -> dict[str, str]:
    jobs_match = re.search(
        r"^(?P<indent>[ \t]*)jobs:\s*$",
        workflow_text,
        flags=re.MULTILINE,
    )
    if jobs_match is None:
        return {}
    jobs_indent = len(jobs_match.group("indent"))
    matches = list(
        re.finditer(
            r"^[ \t]{" + str(jobs_indent + 2) + r"}(?P<name>[A-Za-z0-9_-]+):\s*$",
            workflow_text[jobs_match.end() :],
            flags=re.MULTILINE,
        )
    )
    if not matches:
        return {}
    blocks: dict[str, str] = {}
    jobs_text = workflow_text[jobs_match.end() :]
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(jobs_text)
        blocks[match.group("name")] = jobs_text[start:end]
    return blocks


def _parse_if_expressions(workflow_text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"^\s+if:\s*\$\{\{(.*?)\}\}\s*$",
            workflow_text,
            flags=re.MULTILINE,
        )
    ]


def _if_expression_references(
    expression: str, allowed_contexts: set[str]
) -> None:
    # Roughly parse top-level contexts used by github expression variables, e.g.
    # github.event_name, inputs.run_tests, matrix.mode
    refs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.", expression)
    invalid = sorted({ref for ref in refs if ref not in allowed_contexts})
    if invalid:
        raise AssertionError(
            f"Unsupported if-expression contexts: {invalid} in '{expression}'"
        )


def _if_expression_uses_supported_syntax(expression: str) -> None:
    stripped = expression.strip()
    if stripped == "":
        raise AssertionError("Empty if-expression")

    function_call_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped)
    if function_call_match:
        raise AssertionError(
            f"Unsupported function-call style expression: '{expression}'"
        )

    token_pattern = re.compile(
        r"""
        (?P<ws>\s+)
        |(?P<number>\d+(?:\.\d+)?)
        |(?P<literal>true|false|null)
        |(?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
        |(?P<op>==|!=|<=|>=|&&|\|\||<|>|!|\(|\))
        |(?P<var>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
        """,
        re.VERBOSE,
    )

    position = 0
    for match in token_pattern.finditer(stripped):
        if match.start() != position:
            unsupported = stripped[position : match.start()]
            if unsupported.strip():
                raise AssertionError(
                    f"Unsupported characters in if-expression: '{expression}'"
                )
        position = match.end()

    if position != len(stripped):
        unsupported = stripped[position:]
        if unsupported.strip():
            raise AssertionError(
                f"Unsupported trailing characters in if-expression: '{expression}'"
            )


def _assert_expression_is_wrapped_in_github_syntax(expression: str) -> None:
    assert expression.count("${{") == 1
    assert expression.count("}}") == 1


def base_manifest_dict() -> dict:
    return {
        "project": {
            "package_version": "0.1.0",
            "name": "manifold-guard",
            "package_urls": {
                "homepage": "https://github.com/Martin123132/ManifoldGuard",
                "source": "https://github.com/Martin123132/ManifoldGuard",
                "documentation": "https://github.com/Martin123132/ManifoldGuard/blob/main/README.md",
                "issues": "https://github.com/Martin123132/ManifoldGuard/issues",
                "changelog": "https://github.com/Martin123132/ManifoldGuard/blob/main/CHANGELOG.md",
            },
        },
        "install_modes": [
            {
                "name": "core",
                "command": "python -m pip install -e . --no-deps",
            },
            {
                "name": "embeddings",
                "command": "python -m pip install -e .[embeddings]",
            },
        ],
        "ci_policy": [
            {
                "name": "core",
                "install": "python -m pip install -e . --no-deps",
            },
            {
                "name": "embeddings",
                "install": "python -m pip install -e .[embeddings]",
            },
        ],
        "runtime_guarantees": [
            {
                "name": "offline_first_core",
                "statement": "Core regulation works without sentence-transformers when use_embeddings=false or --no-embeddings is selected.",
            },
            {
                "name": "explicit_embeddings_extra",
                "statement": "Embedding-backed operation is opt-in through pip install -e .[embeddings].",
            },
            {
                "name": "clear_missing_dependency_error",
                "statement": "Embedding paths fail with an actionable message when sentence-transformers is unavailable.",
            },
            {
                "name": "token_shock_embedding_only",
                "statement": "CLI token-shock reporting is embedding-backed and is rejected when --no-embeddings is selected.",
            },
            {
                "name": "no_api_surface_change",
                "statement": "Current hardening work preserves existing regulation and token-shock API signatures.",
            },
        ],
        "support_boundaries": [
            "ManifoldGuard regulates candidate outputs against supplied reference structure.",
            "ManifoldGuard is not a universal fact checker.",
            "Token-shock reporting is available only in embedding-enabled mode.",
        ],
    }


def test_validate_package_urls_passes_when_aligned(docs_quality, tmp_path: Path):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.urls]
        Homepage = "https://github.com/Martin123132/ManifoldGuard"
        Repository = "https://github.com/Martin123132/ManifoldGuard"
        Documentation = "https://github.com/Martin123132/ManifoldGuard/blob/main/README.md"
        Source = "https://github.com/Martin123132/ManifoldGuard"
        Issues = "https://github.com/Martin123132/ManifoldGuard/issues"
        Changelog = "https://github.com/Martin123132/ManifoldGuard/blob/main/CHANGELOG.md"
        """
    )

    manifest_path = build_manifest(tmp_path)
    assert docs_quality.validate_package_urls(
        manifest_path, tmp_path / "pyproject.toml"
    ) is None


def test_validate_package_urls_flags_missing_manifest_keys(docs_quality, tmp_path: Path):
    manifest = {
        "project": {
            "name": "manifold-guard",
            "package_urls": {
                "homepage": "https://example.com",
            },
        }
    }
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.urls]
        Homepage = "https://example.com"
        Source = "https://example.com/src"
        Documentation = "https://example.com/readme"
        Issues = "https://example.com/issues"
        Changelog = "https://example.com/changelog"
        """
    )

    with pytest.raises(RuntimeError, match="missing required keys"):
        docs_quality.validate_package_urls(manifest_path, tmp_path / "pyproject.toml")


def test_validate_package_urls_flags_drift(docs_quality, tmp_path: Path):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.urls]
        Homepage = "https://github.com/Martin123132/ManifoldGuard"
        Source = "https://github.com/Martin123132/ManifoldGuard"
        Documentation = "https://github.com/Martin123132/ManifoldGuard/blob/main/README.md"
        Issues = "https://github.com/Martin123132/ManifoldGuard/issues"
        Changelog = "https://github.com/Martin123132/ManifoldGuard/blob/main/CHANGELOG.md"
        """
    )
    manifest = {
        "project": {
            "name": "manifold-guard",
            "package_urls": {
                "homepage": "https://github.com/Martin123132/ManifoldGuard",
                "source": "https://github.com/Martin123132/ManifoldGuard",
                "documentation": "https://github.com/Martin123132/ManifoldGuard/blob/main/README.md",
                "issues": "https://github.com/Martin123132/ManifoldGuard/issues",
                "changelog": "https://github.com/Martin123132/ManifoldGuard/blob/main/CHANGELOG.md?changed",
            },
        }
    }
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="Manifest package URL drift detected"):
        docs_quality.validate_package_urls(manifest_path, tmp_path / "pyproject.toml")


def test_validate_version_contract_ensures_versions_are_in_sync(docs_quality, tmp_path: Path):
    manifest = base_manifest_dict()
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "manifold-guard"
        version = "0.1.0"
        """
    )

    assert docs_quality.validate_version_contract(manifest_path, tmp_path / "pyproject.toml") is None


def test_validate_version_contract_rejects_drift(docs_quality, tmp_path: Path):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.1.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "manifold-guard"
        version = "0.2.0"
        """
    )

    with pytest.raises(RuntimeError, match="does not match"):
        docs_quality.validate_version_contract(manifest_path, tmp_path / "pyproject.toml")


def test_validate_embedding_extra_passes_when_pinned(docs_quality, tmp_path: Path):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.optional-dependencies]
        embeddings = ["sentence-transformers>=2.6.0,<3", "torch"]
        """
    )

    assert (
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml") is None
    )


def test_validate_embedding_extra_rejects_missing_extra(tmp_path: Path, docs_quality):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        """
    )

    with pytest.raises(RuntimeError, match="embeddings"):
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml")


def test_validate_embedding_extra_rejects_missing_sentence_transformer_constraint(
    docs_quality, tmp_path: Path
):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.optional-dependencies]
        embeddings = ["torch", "transformers"]
        """
    )

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml")


def test_validate_embedding_extra_rejects_wrong_sentence_transformer_range(
    docs_quality, tmp_path: Path
):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.optional-dependencies]
        embeddings = ["sentence-transformers>=2.4.0,<4", "torch"]
        """
    )

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml")


def test_validate_embedding_extra_accepts_formatted_spec_range(docs_quality, tmp_path: Path):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.optional-dependencies]
        embeddings = ["sentence-transformers >= 2.6.0, < 3", "torch"]
        """
    )

    assert (
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml") is None
    )


def test_validate_embedding_extra_rejects_unbounded_sentence_transformers(docs_quality, tmp_path: Path):
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project.optional-dependencies]
        embeddings = ["sentence-transformers", "torch"]
        """
    )

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        docs_quality.validate_embedding_extra(tmp_path / "pyproject.toml")


def test_validate_license_notice_and_contact_accepts_company_policy(docs_quality, tmp_path: Path):
    license_path = tmp_path / "LICENSE"
    license_path.write_text(
        "\n".join(
            [
                "Required Notice: Copyright (c) 2026 TWO HANDS NETWORK LTD.",
                "Required Notice: ManifoldGuard is source-available for personal and non-commercial use. Commercial use requires a separate written license from the licensor.",
                "",
                "# PolyForm Noncommercial License 1.0.0",
                "",
                "Contact us",
                "For collaboration, information on existing products, or other enquiries, please contact (via Email):",
                "",
                "Glyn : glyn@twohandsnetwork.co.uk",
            ]
        ),
        encoding="utf-8",
    )

    assert docs_quality.validate_license_notice_and_contact(license_path) is None


def test_validate_license_notice_and_contact_flags_missing_contact(docs_quality, tmp_path: Path):
    license_path = tmp_path / "LICENSE"
    license_path.write_text(
        "\n".join(
            [
                "Required Notice: Copyright (c) 2026 TWO HANDS NETWORK LTD.",
                "Required Notice: ManifoldGuard is source-available for personal and non-commercial use. Commercial use requires a separate written license from the licensor.",
                "",
                "# PolyForm Noncommercial License 1.0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="License notice/contact contract"):
        docs_quality.validate_license_notice_and_contact(license_path)


def test_validate_release_changelog_alignment_passes_for_matching_version(
    docs_quality,
    tmp_path: Path,
):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.1.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        version = "0.1.0"
        """
    )
    write_changelog(
        tmp_path / "CHANGELOG.md",
        """
        # Changelog

        ## Unreleased

        - ...

        ## 0.1.0 Release Candidate 2 - 2026-06-11

        - Added release alignment checks.
        """,
    )

    manifest_path.parent.joinpath("mbt_ai_tools").mkdir()
    (manifest_path.parent / "mbt_ai_tools" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )

    assert (
        docs_quality.validate_release_changelog_alignment(
            manifest_path, tmp_path / "pyproject.toml"
        )
        is None
    )


def test_validate_release_changelog_alignment_rejects_missing_release_heading(
    docs_quality,
    tmp_path: Path,
):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.2.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        version = "0.2.0"
        """
    )
    write_changelog(
        tmp_path / "CHANGELOG.md",
        """
        # Changelog

        ## 0.1.0 Release Candidate 2 - 2026-06-11

        - Existing release.
        """,
    )

    manifest_path.parent.joinpath("mbt_ai_tools").mkdir()
    (manifest_path.parent / "mbt_ai_tools" / "__init__.py").write_text(
        '__version__ = "0.2.0"\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Missing changelog release entry"):
        docs_quality.validate_release_changelog_alignment(
            manifest_path, tmp_path / "pyproject.toml"
        )


def test_validate_distribution_version_skips_when_distribution_not_installed(
    docs_quality,
    tmp_path: Path,
    monkeypatch,
):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.1.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "manifold-guard"
        version = "0.1.0"
        """,
    )
    def _missing_package(_: str = "manifold-guard") -> str:
        raise docs_quality.PackageNotFoundError("manifold-guard")

    monkeypatch.setattr(
        docs_quality,
        "get_distribution_version",
        _missing_package,
    )

    assert (
        docs_quality.validate_distribution_version(
            manifest_path, tmp_path / "pyproject.toml"
        )
        is None
    )


def test_validate_distribution_version_detects_mismatch_when_installed(docs_quality, tmp_path: Path, monkeypatch):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.1.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "manifold-guard"
        version = "0.1.0"
        """,
    )
    monkeypatch.setattr(
        docs_quality,
        "get_distribution_version",
        lambda name="manifold-guard": "0.2.0",
    )

    with pytest.raises(RuntimeError, match="does not match manifest package_version"):
        docs_quality.validate_distribution_version(
            manifest_path, tmp_path / "pyproject.toml"
        )


def test_validate_distribution_version_passes_for_matching_installed_version(
    docs_quality,
    tmp_path: Path,
    monkeypatch,
):
    manifest = base_manifest_dict()
    manifest["project"]["package_version"] = "0.1.0"
    manifest_path = tmp_path / "docs_product_readiness_manifest.json"
    write_json(manifest_path, manifest)
    write_pyproject(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "manifold-guard"
        version = "0.1.0"
        """,
    )
    monkeypatch.setattr(
        docs_quality,
        "get_distribution_version",
        lambda name="manifold-guard": "0.1.0",
    )

    assert (
        docs_quality.validate_distribution_version(
            manifest_path, tmp_path / "pyproject.toml"
        )
        is None
    )


def test_validate_install_modes_passes_for_valid_contract(docs_quality):
    manifest = base_manifest_dict()
    assert docs_quality.validate_install_modes(manifest) is None


def test_validate_install_modes_rejects_missing_core_definition(docs_quality):
    manifest = base_manifest_dict()
    manifest["install_modes"] = [
        {
            "name": "embeddings",
            "command": "python -m pip install -e .[embeddings]",
        }
    ]

    with pytest.raises(RuntimeError, match="install_modes missing required names"):
        docs_quality.validate_install_modes(manifest)


def test_validate_install_modes_rejects_duplicate_ci_policy_names(docs_quality):
    manifest = base_manifest_dict()
    manifest["ci_policy"].append({"name": "core", "install": "python -m pip install -e . --no-deps"})

    with pytest.raises(RuntimeError, match="Duplicate ci_policy entry name"):
        docs_quality.validate_install_modes(manifest)


def test_validate_install_modes_rejects_ci_policy_missing_fields(docs_quality):
    manifest = base_manifest_dict()
    manifest["ci_policy"] = [{"name": "core"}, {"name": "embeddings", "install": "python -m pip install -e .[embeddings]"}]

    with pytest.raises(TypeError, match="missing required field: install"):
        docs_quality.validate_install_modes(manifest)


def test_validate_install_modes_rejects_ci_policy_command_drift(docs_quality):
    manifest = base_manifest_dict()
    manifest["ci_policy"][0]["install"] = "python3 -m pip install -e . --no-deps"
    manifest["ci_policy"][1]["install"] = "pip install -e .[embeddings]"

    with pytest.raises(RuntimeError, match="core ci_policy install command"):
        docs_quality.validate_install_modes(manifest)


def test_validate_install_modes_rejects_mismatched_mode_names(docs_quality):
    manifest = base_manifest_dict()
    manifest["ci_policy"].append({"name": "experimental", "install": "python -m pip install -e .[embeddings]"})

    with pytest.raises(RuntimeError, match="must match exactly"):
        docs_quality.validate_install_modes(manifest)

    manifest = base_manifest_dict()
    manifest["install_modes"].append({"name": "staging", "command": "python -m pip install -e . --no-deps"})
    with pytest.raises(RuntimeError, match="must match exactly"):
        docs_quality.validate_install_modes(manifest)


def test_validate_runtime_guarantees_passes_for_manifest_contract(docs_quality):
    assert docs_quality.validate_runtime_guarantees(base_manifest_dict()) is None


def test_validate_runtime_guarantees_passes_for_repo_manifest(docs_quality):
    manifest_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "product_readiness_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert docs_quality.validate_runtime_guarantees(manifest) is None


def test_explain_report_example_is_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])

    assert "examples/explain_report.md" in manifest["examples"]
    assert "examples/explain_report.md" in docs_quality.parse_args().required_example_paths


def test_personal_corpus_template_is_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])

    assert "examples/personal_corpus_template.jsonl" in manifest["examples"]
    assert (
        "examples/personal_corpus_template.jsonl"
        in docs_quality.parse_args().required_example_paths
    )


def test_benchmark_guide_is_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])

    assert "docs/benchmark.md" in manifest["documentation"]
    assert "docs/benchmark.md" in docs_quality.parse_args().required_markdown


def test_getting_started_guide_is_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])

    assert "docs/getting_started.md" in manifest["documentation"]
    assert "docs/getting_started.md" in docs_quality.parse_args().required_markdown


def test_cli_quickstart_and_demo_corpus_are_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])
    args = docs_quality.parse_args()

    assert "docs/cli_quickstart.md" in manifest["documentation"]
    assert "docs/cli_quickstart.md" in args.required_markdown
    assert "examples/demo_corpus.jsonl" in manifest["examples"]
    assert "examples/demo_corpus.jsonl" in args.required_example_paths


def test_extending_guide_is_manifested_and_required(docs_quality, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "docs" / "product_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.argv", ["docs_quality.py"])

    assert "docs/extending.md" in manifest["documentation"]
    assert "docs/extending.md" in docs_quality.parse_args().required_markdown


def test_validate_runtime_guarantees_rejects_missing_token_shock_policy(docs_quality):
    manifest = base_manifest_dict()
    manifest["runtime_guarantees"] = [
        item
        for item in manifest["runtime_guarantees"]
        if item["name"] != "token_shock_embedding_only"
    ]

    with pytest.raises(RuntimeError, match="runtime_guarantees missing required names"):
        docs_quality.validate_runtime_guarantees(manifest)


def test_validate_runtime_guarantees_rejects_duplicate_names(docs_quality):
    manifest = base_manifest_dict()
    manifest["runtime_guarantees"].append(
        {
            "name": "offline_first_core",
            "statement": "Duplicate guarantee.",
        }
    )

    with pytest.raises(RuntimeError, match="Duplicate runtime_guarantees entry name"):
        docs_quality.validate_runtime_guarantees(manifest)


def test_validate_runtime_guarantees_rejects_missing_token_shock_boundary(docs_quality):
    manifest = base_manifest_dict()
    manifest["support_boundaries"] = [
        item for item in manifest["support_boundaries"] if "Token-shock reporting" not in item
    ]

    with pytest.raises(RuntimeError, match="support_boundaries"):
        docs_quality.validate_runtime_guarantees(manifest)


def _write_install_docs(path: Path, *, malformed: bool = False) -> None:
    if malformed:
        path.write_text(
            "Install modes:\n- Offline baseline: pip install -e .\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        "\n".join(
            [
                "Install modes:",
                "- Offline baseline (default): `python -m pip install -e . --no-deps`",
                "- Optional semantic mode: `python -m pip install -e .[embeddings]`",
                "",
                "If `sentence-transformers` is unavailable, use offline literal/relation-only regulation with `--no-embeddings` / `use_embeddings=False`.",
            ]
        ),
        encoding="utf-8",
    )


def test_validate_install_guidance_requires_required_contract(docs_quality, tmp_path: Path):
    readme = tmp_path / "README.md"
    claims = tmp_path / "CLAIMS.md"
    replication = tmp_path / "REPLICATION.md"
    for path in (readme, claims, replication):
        _write_install_docs(path)

    assert (
        docs_quality.validate_install_guidance([readme, claims, replication]) is None
    )


def test_validate_install_guidance_flags_incomplete_contract(docs_quality, tmp_path: Path):
    readme = tmp_path / "README.md"
    claims = tmp_path / "CLAIMS.md"
    replication = tmp_path / "REPLICATION.md"
    _write_install_docs(readme)
    _write_install_docs(claims)
    _write_install_docs(replication, malformed=True)

    with pytest.raises(RuntimeError, match="Installation guidance contract"):
        docs_quality.validate_install_guidance([readme, claims, replication])


def test_validate_token_shock_guidance_requires_embedding_policy(docs_quality, tmp_path: Path):
    docs = [
        tmp_path / "README.md",
        tmp_path / "report_schema.md",
        tmp_path / "quality_gates.md",
        tmp_path / "release_checklist.md",
    ]
    for path in docs:
        path.write_text(
            "Token-shock guidance: `--token-shock` is embedding-backed.",
            encoding="utf-8",
        )

    assert docs_quality.validate_token_shock_guidance(docs) is None

    docs[-1].write_text(
        "Token-shock guidance without dependency mode.",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Token-shock guidance contract"):
        docs_quality.validate_token_shock_guidance(docs)


def test_ci_dispatch_inputs_are_present_in_tests_workflow():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "tests.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow_text
    assert "run_installability:" in workflow_text
    assert "run_preflight:" in workflow_text
    assert "run_tests:" in workflow_text
    assert "default: true" in workflow_text


def test_ci_dispatch_defaults_are_true_and_boolean_type():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "tests.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    defaults = re.findall(
        r"run_(installability|preflight|tests):\n\s*description: .*?\n\s*required:\s*false\n\s*default:\s*true\n\s*type:\s*boolean",
        workflow_text,
        flags=re.DOTALL,
    )
    assert defaults == ["installability", "preflight", "tests"]


def test_ci_dispatch_inputs_are_documented_in_release_checklist():
    checklist_path = (
        Path(__file__).resolve().parent.parent / "docs" / "release_checklist.md"
    )
    checklist_text = checklist_path.read_text(encoding="utf-8")

    assert "`run_installability`" in checklist_text
    assert "`run_preflight`" in checklist_text
    assert "`run_tests`" in checklist_text
    assert "workflow_dispatch:" in checklist_text
    assert "default: true" in checklist_text


def test_ci_dispatch_inputs_are_present_in_docs_quality_workflow():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "docs-quality.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "run_core_mode:" in workflow_text
    assert "run_embeddings_mode:" in workflow_text
    assert "run_schema_checks:" in workflow_text
    assert "type: boolean" in workflow_text


def test_docs_quality_dispatch_inputs_have_explicit_defaults_and_types():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "docs-quality.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    defaults = re.findall(
        r"run_(core_mode|embeddings_mode|schema_checks):\n\s*description: .*?\n\s*required:\s*false\n\s*default:\s*true\n\s*type:\s*boolean",
        workflow_text,
        flags=re.DOTALL,
    )
    assert defaults == ["core_mode", "embeddings_mode", "schema_checks"]


def test_docs_quality_workflow_matrix_mode_is_core_and_embeddings():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "docs-quality.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    matrix_match = re.search(
        r"matrix:\s*\n\s*mode:\s*\[(.*?)\]",
        workflow_text,
        flags=re.DOTALL,
    )
    assert matrix_match is not None
    matrix_values = matrix_match.group(1)
    assert "core" in matrix_values and "embeddings" in matrix_values
    assert matrix_values.count("core") == 1
    assert matrix_values.count("embeddings") == 1


def test_docs_quality_workflow_controls_and_scope_logging_are_assertable():
    workflow_path = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "docs-quality.yml"
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert (
        "echo \"run_core_mode=${{ inputs.run_core_mode }}\""
        in workflow_text
    )
    assert (
        "echo \"run_embeddings_mode=${{ inputs.run_embeddings_mode }}\""
        in workflow_text
    )
    assert (
        "echo \"run_schema_checks=${{ inputs.run_schema_checks }}\""
        in workflow_text
    )
    assert "if: ${{ github.event_name != 'workflow_dispatch' || (matrix.mode == 'core' && inputs.run_core_mode == true) || (matrix.mode == 'embeddings' && inputs.run_embeddings_mode == true) }}" in workflow_text
    assert "if: ${{ matrix.mode == 'embeddings' && (github.event_name != 'workflow_dispatch' || inputs.run_embeddings_mode == true) }}" in workflow_text
    assert "if: ${{ github.event_name != 'workflow_dispatch' || inputs.run_schema_checks == true }}" in workflow_text


@pytest.mark.parametrize(
    "workflow_name,allowed_contexts",
    [
        (
            ".github/workflows/tests.yml",
            {"github", "inputs", "matrix"},
        ),
        (
            ".github/workflows/docs-quality.yml",
            {"github", "inputs", "matrix"},
        ),
    ],
)
def test_ci_if_expressions_use_allowed_contexts_and_syntax(
    workflow_name, allowed_contexts
):
    workflow_path = (
        Path(__file__).resolve().parent.parent / workflow_name
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")
    if_expressions = _parse_if_expressions(workflow_text)

    assert if_expressions, f"{workflow_name} has no parsed if expressions"
    for raw_expression in if_expressions:
        _assert_expression_is_wrapped_in_github_syntax(f"${{{{ {raw_expression} }}}}")
        _if_expression_references(raw_expression, allowed_contexts)
        _if_expression_uses_supported_syntax(raw_expression)


@pytest.mark.parametrize(
    "workflow_name,expected_jobs",
    [
        (
            ".github/workflows/tests.yml",
            {"installability", "preflight", "test"},
        ),
        (
            ".github/workflows/docs-quality.yml",
            {"docs-quality"},
        ),
    ],
)
def test_ci_jobs_match_expected_names(workflow_name, expected_jobs):
    workflow_path = Path(__file__).resolve().parent.parent / workflow_name
    workflow_text = workflow_path.read_text(encoding="utf-8")
    job_blocks = parse_ci_jobs(workflow_text)
    observed_jobs = set(job_blocks.keys())
    assert expected_jobs.issubset(observed_jobs), (
        f"{workflow_name} missing expected jobs: {sorted(expected_jobs - observed_jobs)}"
    )


def test_ci_workflow_contract_validation_on_repo_root(docs_quality):
    assert docs_quality.validate_ci_workflow_contracts(
        Path(__file__).resolve().parent.parent
    ) is None


def test_validate_ci_workflow_contracts_rejects_missing_required_jobs(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard minimal
        on: [push]
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard docs quality minimal
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="tests workflow missing required jobs"):
        docs_quality.validate_ci_workflow_contracts(tmp_path)


def test_validate_ci_workflow_contracts_rejects_malformed_matrix_mode_declaration(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard malformed modes
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings, legacy]
            steps: []
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings]
            steps: []
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings]
            steps: []
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard docs quality malformed modes
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings, legacy]
            steps: []
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported matrix.mode declaration"):
        docs_quality.validate_ci_workflow_contracts(tmp_path)


def test_validate_ci_workflow_contracts_rejects_required_job_without_matrix_block(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard missing matrix in one job
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            steps:
              - run: echo ok
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings]
            steps: []
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings]
            steps: []
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard missing matrix doc
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: [core, embeddings]
            steps: []
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing matrix.mode"):
        docs_quality.validate_ci_workflow_contracts(tmp_path)


def test_validate_ci_workflow_contracts_accepts_matrix_mode_declaration_with_quotes_and_comment(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard quoted modes
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: ["core", "embeddings"]  # quoted mode values
            steps:
              - run: echo ok
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: ["core", "embeddings"]  # quoted mode values
            steps:
              - run: echo ok
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: ["core", "embeddings"]  # quoted mode values
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard quoted modes docs quality
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode: ["core", "embeddings"]  # quoted mode values
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )

    assert docs_quality.validate_ci_workflow_contracts(tmp_path) is None


def test_validate_ci_workflow_contracts_accepts_multiline_matrix_mode_declaration(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard multiline modes
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
                  - embeddings
            steps: []
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
                  - embeddings
            steps: []
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
                  - embeddings
            steps: []
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard multiline modes docs quality
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
                  - embeddings
            steps: []
        """,
        encoding="utf-8",
    )

    assert docs_quality.validate_ci_workflow_contracts(tmp_path) is None


def test_validate_ci_workflow_contracts_accepts_multiline_matrix_mode_with_comment(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard multiline modes with comments
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:  # explicit list values
                  - core
                  - embeddings
            steps:
              - run: echo ok
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:  # explicit list values
                  - core
                  - embeddings
            steps:
              - run: echo ok
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:  # explicit list values
                  - core
                  - embeddings
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard multiline modes with comments docs quality
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:  # explicit list values
                  - core
                  - embeddings
            steps:
              - run: echo ok
        """,
        encoding="utf-8",
    )

    assert docs_quality.validate_ci_workflow_contracts(tmp_path) is None


def test_validate_ci_workflow_contracts_rejects_partial_multiline_matrix_mode(tmp_path: Path, docs_quality):
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "tests.yml").write_text(
        """
        name: ManifoldGuard multiline partial mode
        on: [push]
        jobs:
          installability:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
            steps: []
          preflight:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
            steps: []
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
            steps: []
        """,
        encoding="utf-8",
    )
    (workflow_root / "docs-quality.yml").write_text(
        """
        name: ManifoldGuard multiline partial mode docs quality
        on: [push]
        jobs:
          docs-quality:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                mode:
                  - core
            steps: []
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported matrix.mode declaration"):
        docs_quality.validate_ci_workflow_contracts(tmp_path)


@pytest.mark.parametrize(
    "expression",
    [
        "inputs['run_tests'] == true",
        "hashFiles('**/*')",
        "startsWith(github.event_name, 'pull_request')",
    ],
)
def test_ci_if_expression_syntax_rejects_unsupported_patterns(expression):
    with pytest.raises(
        AssertionError,
        match="Unsupported|Unsupported function-call style expression",
    ):
        _if_expression_uses_supported_syntax(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "secrets.GITHUB_TOKEN == ''",
        "env.PYTHONPATH == ''",
        "runner.os == 'Linux'",
    ],
)
def test_ci_if_expression_references_reject_disallowed_contexts(expression):
    with pytest.raises(AssertionError, match="Unsupported if-expression contexts"):
        _if_expression_references(
            expression, {"github", "inputs", "matrix"}
        )


def _workflow_has_import_entrypoint_checks(workflow_text: str) -> bool:
    return (
        "from mbt_ai_tools.cli import main" in workflow_text
        and "from mbt_ai_tools import extract_relations, evaluate_candidate, regulate_candidates, token_shock_map"
        in workflow_text
    )


def _matrix_mode_install_blocks(workflow_text: str) -> list[str]:
    lines = workflow_text.splitlines()
    blocks: list[str] = []
    i = 0
    shell_if_core = re.compile(
        r'if\s+\[\s*["\']\$\{\{\s*matrix\.mode\s*\}\}["\']\s*(?:=|==|!=)\s*["\']core["\']\s*\]\s*;?\s*then',
        flags=re.IGNORECASE,
    )
    shell_if_embeddings = re.compile(
        r'if\s+\[\s*["\']\$\{\{\s*matrix\.mode\s*\}\}["\']\s*(?:=|==|!=)\s*["\']embeddings["\']\s*\]\s*;?\s*then',
        flags=re.IGNORECASE,
    )
    while i < len(lines):
        line = lines[i]
        if shell_if_core.search(line) or shell_if_embeddings.search(line):
            block_lines = [line]
            j = i + 1
            while j < len(lines):
                block_lines.append(lines[j])
                if lines[j].strip() == "fi":
                    break
                j += 1
            blocks.append("\n".join(block_lines))
            i = j + 1
            continue
        i += 1
    return blocks


def _workflow_mode_condition_counts(workflow_text: str) -> tuple[int, int]:
    core_refs = len(
        re.findall(r"matrix\.mode\s*==\s*['\"]core['\"]", workflow_text)
    )
    core_refs += len(
        re.findall(
            r'["\']\$\{\{\s*matrix\.mode\s*\}\}["\']\s*(?:=|==|!=)\s*["\']core["\']',
            workflow_text,
        )
    )
    embeddings_refs = len(
        re.findall(r"matrix\.mode\s*==\s*['\"]embeddings['\"]", workflow_text)
    )
    embeddings_refs += len(
        re.findall(
            r'["\']\$\{\{\s*matrix\.mode\s*\}\}["\']\s*(?:=|==|!=)\s*["\']embeddings["\']',
            workflow_text,
        )
    )
    return core_refs, embeddings_refs


@pytest.mark.parametrize(
    "workflow_name",
    [
        ".github/workflows/tests.yml",
        ".github/workflows/docs-quality.yml",
    ],
)
def test_ci_workflows_require_entrypoint_import_smoke_after_install(workflow_name):
    workflow_path = (
        Path(__file__).resolve().parent.parent / workflow_name
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")
    assert _workflow_has_import_entrypoint_checks(workflow_text), (
        f"{workflow_name} missing import smoke checks for public entrypoints."
    )


@pytest.mark.parametrize(
    "workflow_name",
    [
        ".github/workflows/tests.yml",
        ".github/workflows/docs-quality.yml",
    ],
)
def test_ci_workflows_require_matrix_mode_install_contract(workflow_name):
    workflow_path = Path(__file__).resolve().parent.parent / workflow_name
    workflow_text = workflow_path.read_text(encoding="utf-8")
    blocks = _matrix_mode_install_blocks(workflow_text)

    assert blocks, f"{workflow_name} has no matrix-mode install blocks"

    required_blocks = [
        block
        for block in blocks
        if "python -m pip install -e . --no-deps" in block
        and "python -m pip install -e .[embeddings]" in block
    ]
    assert required_blocks, (
        f"{workflow_name} matrix-mode install block missing core and embeddings commands."
    )


@pytest.mark.parametrize(
    "workflow_name",
    [
        ".github/workflows/tests.yml",
        ".github/workflows/docs-quality.yml",
    ],
)
def test_ci_workflows_reference_both_matrix_modes(workflow_name):
    workflow_path = Path(__file__).resolve().parent.parent / workflow_name
    workflow_text = workflow_path.read_text(encoding="utf-8")
    core_refs, embeddings_refs = _workflow_mode_condition_counts(workflow_text)

    assert core_refs > 0 and embeddings_refs > 0, (
        f"{workflow_name} must reference both matrix modes in workflow conditionals."
    )


@pytest.mark.parametrize(
    "workflow_name",
    [
        ".github/workflows/tests.yml",
        ".github/workflows/docs-quality.yml",
    ],
)
def test_ci_jobs_include_scope_summary_step(workflow_name):
    workflow_path = (
        Path(__file__).resolve().parent.parent / workflow_name
    )
    workflow_text = workflow_path.read_text(encoding="utf-8")
    job_blocks = parse_ci_jobs(workflow_text)
    missing_summary = [
        name
        for name, block in job_blocks.items()
        if "name: Summarize execution scope" not in block
    ]
    assert not missing_summary, f"{workflow_name} jobs missing scope summary step: {missing_summary}"

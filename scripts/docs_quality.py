#!/usr/bin/env python
"""Run repository documentation and report-contract quality checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
import ast
from importlib.metadata import PackageNotFoundError, version as get_distribution_version
try:
    import tomllib
except ImportError:  # pragma: no cover - compatibility fallback for Python <3.11
    tomllib = None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_urls_section_fallback(text: str) -> dict:
    in_urls_section = False
    urls = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_urls_section = stripped == "[project.urls]"
            continue
        if not in_urls_section or "=" not in stripped:
            continue
        key, value = (segment.strip() for segment in stripped.split("=", 1))
        urls[key] = _strip_quotes(value)
    return urls


def _parse_project_section_fallback(text: str) -> dict:
    in_project_section = False
    values = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_section = stripped == "[project]"
            continue
        if not in_project_section or "=" not in stripped:
            continue
        key, value = (segment.strip() for segment in stripped.split("=", 1))
        parsed = _strip_quotes(value)
        if parsed:
            values[key] = parsed
    return values


def _parse_optional_dependencies_section_fallback(text: str) -> dict:
    in_opt_section = False
    extras: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_opt_section = stripped == "[project.optional-dependencies]"
            continue
        if not in_opt_section:
            continue
        if "=" not in stripped:
            continue
        key, value = (segment.strip() for segment in stripped.split("=", 1))
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            continue
        if isinstance(parsed, list):
            extras[key] = [str(item) for item in parsed]
    return extras


def parse_pyproject(text: str) -> dict:
    if tomllib is None:
        return {
            "project": {
                "urls": _parse_urls_section_fallback(text),
                "version": _parse_project_section_fallback(text).get("version"),
                "optional-dependencies": _parse_optional_dependencies_section_fallback(text),
            }
        }
    return tomllib.loads(text)


def parse_ci_jobs(workflow_text: str) -> dict[str, str]:
    jobs_match = re.search(
        r"^(?P<indent>[ \t]*)jobs:\s*$", workflow_text, flags=re.MULTILINE
    )
    if jobs_match is None:
        return {}
    jobs_indent = len(jobs_match.group("indent"))
    jobs_text = workflow_text[jobs_match.end() :]
    matches = list(
        re.finditer(
            r"^[ \t]{" + str(jobs_indent + 2) + r"}(?P<name>[A-Za-z0-9_-]+):\s*$",
            jobs_text,
            flags=re.MULTILINE,
        )
    )
    if not matches:
        return {}
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(jobs_text)
        blocks[match.group("name")] = jobs_text[start:end]
    return blocks


def validate_ci_workflow_contracts(project_root: Path) -> None:
    tests_workflow = project_root / ".github" / "workflows" / "tests.yml"
    docs_workflow = project_root / ".github" / "workflows" / "docs-quality.yml"
    if not tests_workflow.exists():
        raise FileNotFoundError(f"Missing tests workflow: {tests_workflow}")
    if not docs_workflow.exists():
        raise FileNotFoundError(f"Missing docs-quality workflow: {docs_workflow}")

    tests_text = tests_workflow.read_text(encoding="utf-8")
    docs_text = docs_workflow.read_text(encoding="utf-8")

    tests_jobs = parse_ci_jobs(tests_text)
    docs_jobs = parse_ci_jobs(docs_text)
    if not {"installability", "preflight", "test"}.issubset(tests_jobs):
        raise RuntimeError(
            "tests workflow missing required jobs: installability/preflight/test"
        )
    if not {"docs-quality"}.issubset(docs_jobs):
        raise RuntimeError("docs-quality workflow missing required job: docs-quality")

    def _extract_matrix_modes(workflow_text: str) -> list[list[str]]:
        modes = []
        lines = workflow_text.splitlines()
        for index, line in enumerate(lines):
            matrix_match = re.match(r"^([ \t]*)matrix:\s*$", line)
            if not matrix_match:
                continue
            matrix_indent = len(matrix_match.group(1))
            mode_pattern = re.compile(
                r"^[ \t]{" + str(matrix_indent + 2) + r",}mode:\s*(.*?)\s*$"
            )
            for line_index in range(index + 1, len(lines)):
                follow = lines[line_index]
                stripped_follow = follow.strip()
                if stripped_follow == "":
                    continue
                if re.match(r"^\S", follow):
                    break
                if len(follow) - len(follow.lstrip(" \t")) <= matrix_indent:
                    break

                mode_match = mode_pattern.match(follow)
                if not mode_match:
                    continue

                mode_expression = mode_match.group(1)
                mode_expression = re.sub(r"\s*#.*$", "", mode_expression).strip()
                mode_key_indent = len(follow) - len(follow.lstrip(" \t"))
                parsed: list[str] = []

                if mode_expression:
                    parsed_mode_values = None
                    try:
                        parsed_mode_values = ast.literal_eval(mode_expression)
                    except (SyntaxError, ValueError):
                        parsed_mode_values = None

                    if isinstance(parsed_mode_values, list):
                        parsed = [str(item) for item in parsed_mode_values]
                    elif (
                        mode_expression.startswith("[") and mode_expression.endswith("]")
                    ):
                        inner = mode_expression[1:-1]
                        parsed = [
                            item.strip().strip("'\"")
                            for item in inner.split(",")
                            if item.strip()
                        ]
                else:
                    block_indent = mode_key_indent + 2
                    for item_index in range(line_index + 1, len(lines)):
                        item_line = lines[item_index]
                        item_stripped = item_line.strip()
                        if item_stripped == "" or item_stripped.startswith("#"):
                            continue
                        if (
                            len(item_line) - len(item_line.lstrip(" \t"))
                            <= mode_key_indent
                        ):
                            break

                        item_match = re.match(
                            rf"^[ \t]{{{block_indent},}}-\s*(.*?)\s*$",
                            item_line,
                        )
                        if not item_match:
                            break
                        raw_item = item_match.group(1)
                        raw_item = re.sub(r"\s*#.*$", "", raw_item).strip()
                        if raw_item:
                            parsed.append(raw_item.strip("'\""))

                if parsed:
                    modes.append([mode for mode in parsed if mode])
        return modes

    def _required_jobs_have_valid_matrix_modes(
        workflow_name: str,
        jobs: dict[str, str],
        required_jobs: set[str],
    ) -> None:
        for job_name in sorted(required_jobs):
            block = jobs.get(job_name)
            if block is None:
                raise RuntimeError(f"{workflow_name} missing required job: {job_name}")
            mode_blocks = _extract_matrix_modes(block)
            if not mode_blocks:
                raise RuntimeError(
                    f"{workflow_name}:{job_name} is missing matrix.mode declaration."
                )
            for mode_values in mode_blocks:
                if set(mode_values) != {"core", "embeddings"} or len(mode_values) != 2:
                    raise RuntimeError(
                        f"{workflow_name}:{job_name} has unsupported matrix.mode declaration."
                    )

    _required_jobs_have_valid_matrix_modes(
        "tests workflow",
        tests_jobs,
        {"installability", "preflight", "test"},
    )
    _required_jobs_have_valid_matrix_modes(
        "docs-quality workflow",
        docs_jobs,
        {"docs-quality"},
    )

    if "workflow_dispatch:" in tests_text:
        if "name: Summarize execution scope" not in tests_text:
            raise RuntimeError("tests workflow missing execution scope summary step")
        for marker in (
            "matrix.mode == 'core'",
            "matrix.mode == 'embeddings'",
        ):
            if marker not in tests_text:
                raise RuntimeError(
                    f"tests workflow missing matrix mode conditional: {marker}"
                )

    if "workflow_dispatch:" in docs_text:
        if "name: Summarize execution scope" not in docs_text:
            raise RuntimeError("docs-quality workflow missing execution scope summary step")
        for marker in (
            "matrix.mode == 'core'",
            "matrix.mode == 'embeddings'",
        ):
            if marker not in docs_text:
                raise RuntimeError(
                    f"docs-quality workflow missing matrix mode conditional: {marker}"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate repository documentation contracts and report schema fixtures."
    )
    parser.add_argument(
        "--manifest",
        default="docs/product_readiness_manifest.json",
        help="Path to product readiness manifest.",
    )
    parser.add_argument(
        "--schema",
        default="docs/report_schema.json",
        help="Path to report JSON schema.",
    )
    parser.add_argument(
        "--required-markdown",
        nargs="*",
        default=[
            "README.md",
            "CLAIMS.md",
            "REPLICATION.md",
            "RELEASE_PROCESS.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "SUPPORT.md",
            "CODE_OF_CONDUCT.md",
            "docs/cli_quickstart.md",
            "docs/getting_started.md",
            "docs/extending.md",
            "docs/report_schema.md",
            "docs/report_schema.json",
            "docs/quality_gates.md",
            "docs/release_checklist.md",
            "docs/benchmark.md",
            "docs/evaluation_report.md",
            "docs/exp21_challenge.md",
            "docs/package_publishing.md",
            "docs/package_installation.md",
            "docs/v0.1.6_plan.md",
            "docs/v0.1.8_scale_baseline.md",
        ],
        help="Markdown or JSON schema docs required for read checks.",
    )
    parser.add_argument(
        "--required-example-paths",
        nargs="*",
        default=[
            "examples/batch_input.jsonl",
            "examples/demo_corpus.jsonl",
            "examples/cli_json_report.md",
            "examples/explain_report.md",
            "examples/markdown_audit_report.md",
            "examples/csv_audit_report.csv",
            "examples/personal_corpus_template.jsonl",
            "examples/regression_corpus.jsonl",
            "examples/single_report_example.json",
            "examples/batch_report_example.jsonl",
        ],
        help="Example files required for docs quality checks.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to project pyproject file.",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip JSON schema validation pass.",
    )
    return parser.parse_args()


def normalize_url_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def read_pyproject_urls(pyproject_path: Path) -> dict:
    data = parse_pyproject(pyproject_path.read_text(encoding="utf-8"))
    urls = data.get("project", {}).get("urls", {})
    if not isinstance(urls, dict):
        return {}
    return {normalize_url_key(key): str(value) for key, value in urls.items()}


def validate_package_urls(manifest_path: Path, pyproject_path: Path) -> None:
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Missing pyproject file: {pyproject_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_urls = manifest.get("project", {}).get("package_urls", {})
    if not isinstance(manifest_urls, dict):
        raise TypeError("Manifest project.package_urls must be a mapping.")

    normalized_manifest_urls = {
        normalize_url_key(key): str(value) for key, value in manifest_urls.items()
    }

    required_keys = {
        "homepage",
        "source",
        "documentation",
        "issues",
        "changelog",
    }
    missing_manifest = sorted(required_keys - set(normalized_manifest_urls))
    if missing_manifest:
        raise RuntimeError(
            f"Manifest project.package_urls missing required keys: {missing_manifest}"
        )

    pyproject_urls = read_pyproject_urls(pyproject_path)
    missing_pyproject = sorted(required_keys - set(pyproject_urls))
    if missing_pyproject:
        raise RuntimeError(
            f"Pyproject project.urls missing required keys: {missing_pyproject}"
        )

    mismatches = []
    for key in required_keys:
        if normalized_manifest_urls.get(key) != pyproject_urls.get(key):
            mismatches.append(
                f"{key}: manifest={normalized_manifest_urls.get(key)!r}, pyproject={pyproject_urls.get(key)!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Manifest package URL drift detected: " + "; ".join(mismatches)
        )


def parse_version(text: str) -> str | None:
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return match.group(1) if match else None


def validate_version_contract(manifest_path: Path, pyproject_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_version = (
        manifest.get("project", {}).get("package_version") if isinstance(manifest.get("project"), dict) else None
    )

    data = parse_pyproject(pyproject_path.read_text(encoding="utf-8"))
    pyproject_version = data.get("project", {}).get("version")
    if not manifest_version:
        raise RuntimeError("Manifest project.package_version is missing.")
    if manifest_version != pyproject_version:
        raise RuntimeError(
            f"Manifest package_version ({manifest_version}) does not match pyproject version ({pyproject_version})."
        )

    init_file = manifest_path.parent / "mbt_ai_tools" / "__init__.py"
    if init_file.exists():
        init_version = parse_version(init_file.read_text(encoding="utf-8"))
        if init_version and init_version != manifest_version:
            raise RuntimeError(
                f"__init__.py version ({init_version}) does not match manifest package_version ({manifest_version})."
            )
def validate_embedding_extra(pyproject_path: Path) -> None:
    data = parse_pyproject(pyproject_path.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies")
    if not isinstance(extras, dict):
        raise RuntimeError("Missing optional dependency extra: embeddings.")

    embeddings = extras.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError("Missing optional dependency extra: embeddings.")

    if not any(
        isinstance(spec, str)
        and re.fullmatch(
            r"sentence-transformers\s*>=\s*2\.6\.0\s*,\s*<\s*3",
            spec.strip(),
        )
        for spec in embeddings
    ):
        raise RuntimeError(
            "Optional dependency extra 'embeddings' missing sentence-transformers constraint."
        )


def validate_license_notice_and_contact(license_path: Path) -> None:
    text = license_path.read_text(encoding="utf-8")
    required_snippets = [
        "Required Notice: Copyright (c) 2026 TWO HANDS NETWORK LTD.",
        "source-available for personal and non-commercial use",
        "Commercial use",
        "PolyForm Noncommercial License 1.0.0",
        "Contact us",
        "glyn@twohandsnetwork.co.uk",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise RuntimeError(
            "License notice/contact contract missing required text: "
            + ", ".join(missing)
        )


def validate_install_modes(manifest: dict) -> None:
    install_modes = manifest.get("install_modes")
    ci_policy = manifest.get("ci_policy")

    if not isinstance(install_modes, list) or not install_modes:
        raise TypeError("Manifest field 'install_modes' must be a non-empty list.")
    if not isinstance(ci_policy, list) or not ci_policy:
        raise TypeError("Manifest field 'ci_policy' must be a non-empty list.")

    install_modes_by_name = {}
    for mode in install_modes:
        if not isinstance(mode, dict):
            raise TypeError("Each install_modes entry must be an object.")
        name = mode.get("name")
        if not name:
            raise TypeError("Each install_modes entry must include a name.")
        if name in install_modes_by_name:
            raise RuntimeError(f"Duplicate install_modes entry name: {name}")
        install_modes_by_name[name] = mode

    required_install_modes = {"core", "embeddings"}
    missing_install_modes = required_install_modes - set(install_modes_by_name)
    if missing_install_modes:
        raise RuntimeError(
            f"install_modes missing required names: {sorted(missing_install_modes)}"
        )

    for mode_name in required_install_modes:
        command = install_modes_by_name[mode_name].get("command", "")
        if mode_name == "core" and "python -m pip install -e ." not in command:
            raise RuntimeError(
                "core install mode command must include 'python -m pip install -e .'."
            )
        if mode_name == "core" and "--no-deps" not in command:
            raise RuntimeError("core install mode command must include '--no-deps'.")
        if mode_name == "embeddings" and ".[embeddings]" not in command:
            raise RuntimeError(
                "embeddings install mode command must include '.[embeddings]'."
            )

    ci_policy_by_name = {}
    for policy in ci_policy:
        if not isinstance(policy, dict):
            raise TypeError("Each ci_policy entry must be an object.")
        name = policy.get("name")
        if not name:
            raise TypeError("Each ci_policy entry must include a name.")
        if name in ci_policy_by_name:
            raise RuntimeError(f"Duplicate ci_policy entry name: {name}")
        if "install" not in policy:
            raise TypeError(f"ci_policy entry '{name}' missing required field: install")
        ci_policy_by_name[name] = policy

    missing_ci_modes = required_install_modes - set(ci_policy_by_name)
    if missing_ci_modes:
        raise RuntimeError(
            f"ci_policy missing required names: {sorted(missing_ci_modes)}"
        )

    if set(install_modes_by_name) != set(ci_policy_by_name):
        raise RuntimeError(
            "install_modes and ci_policy mode names must match exactly."
        )

    for mode_name in required_install_modes:
        command = ci_policy_by_name[mode_name]["install"]
        if mode_name == "core" and "python -m pip install -e ." not in command:
            raise RuntimeError(
                "core ci_policy install command must include 'python -m pip install -e .'."
            )
        if mode_name == "core" and "--no-deps" not in command:
            raise RuntimeError("core ci_policy install command must include '--no-deps'.")
        if mode_name == "embeddings" and ".[embeddings]" not in command:
            raise RuntimeError(
                "embeddings ci_policy install command must include '.[embeddings]'."
            )


def validate_runtime_guarantees(manifest: dict) -> None:
    guarantees = manifest.get("runtime_guarantees")
    support_boundaries = manifest.get("support_boundaries")
    if not isinstance(guarantees, list) or not guarantees:
        raise TypeError("Manifest field 'runtime_guarantees' must be a non-empty list.")
    if not isinstance(support_boundaries, list) or not support_boundaries:
        raise TypeError("Manifest field 'support_boundaries' must be a non-empty list.")

    guarantee_by_name = {}
    for item in guarantees:
        if not isinstance(item, dict):
            raise TypeError("Each runtime_guarantees entry must be an object.")
        name = item.get("name")
        if not name:
            raise TypeError("Each runtime_guarantees entry must include a name.")
        if name in guarantee_by_name:
            raise RuntimeError(f"Duplicate runtime_guarantees entry name: {name}")
        guarantee_by_name[name] = item
    required_names = {
        "offline_first_core",
        "explicit_embeddings_extra",
        "clear_missing_dependency_error",
        "token_shock_embedding_only",
        "no_api_surface_change",
    }
    missing = sorted(required_names - set(guarantee_by_name))
    if missing:
        raise RuntimeError(f"runtime_guarantees missing required names: {missing}")

    token_shock_statement = str(
        guarantee_by_name["token_shock_embedding_only"].get("statement", "")
    )
    if "--no-embeddings" not in token_shock_statement or "embedding" not in token_shock_statement.lower():
        raise RuntimeError(
            "token_shock_embedding_only guarantee must name embedding mode and --no-embeddings behavior."
        )

    boundary_text = "\n".join(str(item) for item in support_boundaries)
    if "Token-shock reporting" not in boundary_text or "embedding-enabled mode" not in boundary_text:
        raise RuntimeError(
            "support_boundaries must document token-shock as embedding-enabled only."
        )


def validate_release_changelog_alignment(manifest_path: Path, pyproject_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_version = (
        manifest.get("project", {}).get("package_version")
        if isinstance(manifest.get("project"), dict)
        else None
    )
    if not manifest_version:
        raise RuntimeError("Manifest project.package_version is missing.")

    data = parse_pyproject(pyproject_path.read_text(encoding="utf-8"))
    pyproject_version = data.get("project", {}).get("version")
    if manifest_version != pyproject_version:
        raise RuntimeError(
            "Cannot validate changelog until manifest and pyproject versions match."
        )

    changelog_path = manifest_path.parent.parent / "CHANGELOG.md"
    if not changelog_path.exists():
        changelog_path = manifest_path.parent / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    if not re.search(
        rf"^\s*##\s+{re.escape(manifest_version)}\b",
        changelog,
        flags=re.MULTILINE,
    ):
        raise RuntimeError(
            f"Missing changelog release entry for version {manifest_version} in CHANGELOG.md."
        )


def validate_distribution_version(manifest_path: Path, pyproject_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_version = (
        manifest.get("project", {}).get("package_version")
        if isinstance(manifest.get("project"), dict)
        else None
    )
    if not manifest_version:
        raise RuntimeError("Manifest project.package_version is missing.")

    pyproject_data = parse_pyproject(pyproject_path.read_text(encoding="utf-8"))
    pyproject_version = pyproject_data.get("project", {}).get("version")
    if manifest_version != pyproject_version:
        raise RuntimeError(
            "Cannot validate installed distribution until manifest and pyproject versions match."
        )

    project_name = pyproject_data.get("project", {}).get("name")
    distribution_names = [name for name in (project_name, "mbt-ai-tools") if name]

    distribution_version = None
    distribution_name = None
    for candidate_name in dict.fromkeys(distribution_names):
        try:
            distribution_version = get_distribution_version(candidate_name)
            distribution_name = candidate_name
            break
        except PackageNotFoundError:
            continue

    if distribution_version is None:
        return

    if distribution_version != manifest_version:
        raise RuntimeError(
            f"Installed {distribution_name} distribution version "
            f"({distribution_version}) does not match manifest package_version "
            f"({manifest_version})."
        )


def validate_install_guidance(doc_paths: list[Path]) -> None:
    required_checks = [
        (
            "install modes section",
            r"Install modes:",
        ),
        (
            "offline baseline install line",
            r"^\s*-\s*Offline baseline \(default\):\s*`python -m pip install -e \. --no-deps`",
        ),
        (
            "optional semantic install line",
            r"^\s*-\s*Optional semantic mode:\s*`python -m pip install -e \.\[embeddings\]`",
        ),
        (
            "offline fallback sentence",
            r"If `sentence-transformers` is unavailable, use offline literal/relation-only regulation with `--no-embeddings` / `use_embeddings=False`\.",
        ),
    ]

    failures: list[str] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        path_name = str(doc_path)
        for label, pattern in required_checks:
            if not re.search(pattern, text, re.MULTILINE | re.DOTALL):
                failures.append(f"{path_name} missing {label}")

    if failures:
        raise RuntimeError(
            "Installation guidance contract not aligned across docs: "
            + "; ".join(failures)
        )


def validate_token_shock_guidance(doc_paths: list[Path]) -> None:
    failures: list[str] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        normalized = text.lower()
        names_token_shock = "--token-shock" in text or "token_shock" in text or "token-shock" in normalized
        names_embedding_mode = "embedding" in normalized
        if not names_token_shock:
            failures.append(f"{doc_path} missing token-shock guidance")
        if not names_embedding_mode:
            failures.append(f"{doc_path} missing token-shock embedding-mode guidance")

    if failures:
        raise RuntimeError(
            "Token-shock guidance contract not aligned across docs: "
            + "; ".join(failures)
        )


def validate_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []

    required_sections = (
        "documentation",
        "examples",
        "github_project_intake",
        "maintenance_automation",
        "validation_scripts",
    )
    for section in required_sections:
        for path_value in manifest.get(section, []):
            path = Path(path_value["path"] if isinstance(path_value, dict) else path_value)
            if not path.exists():
                missing.append(str(path))

    if missing:
        raise FileNotFoundError(f"Missing manifest-referenced paths: {missing}")
    return manifest


def validate_readable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() in {".md", ".json", ".txt"}:
        path.read_text(encoding="utf-8")
    else:
        path.read_bytes()


def validate_example_freshness(project_root: Path) -> None:
    script_path = project_root / "scripts" / "validate_examples.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing example validator: {script_path}")

    spec = importlib.util.spec_from_file_location("validate_examples", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example validator: {script_path}")
    spec.loader.exec_module(module)
    failures = module.validate_examples(project_root)
    if failures:
        raise RuntimeError(
            "Example output drift detected: " + "; ".join(str(item) for item in failures)
        )


def validate_schema_examples(schema_path: Path, examples):
    from jsonschema import validate

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    failures = 0

    for file_name in examples:
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing schema example: {path}")
        if path.suffix.lower() == ".jsonl":
            for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    validate(instance=json.loads(raw_line), schema=schema)
                except Exception as exc:
                    failures += 1
                    print(f"Schema validation failed for {path}:{line_number}: {exc}")
        else:
            try:
                validate(
                    instance=json.loads(path.read_text(encoding="utf-8")),
                    schema=schema,
                )
            except Exception as exc:
                failures += 1
                print(f"Schema validation failed for {path}: {exc}")

    if failures:
        raise RuntimeError("Schema validation failed for one or more report fixtures.")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    schema_path = Path(args.schema)
    pyproject_path = Path(args.pyproject)

    if not manifest_path.exists():
        print(f"Missing manifest file: {manifest_path}")
        return 1
    if not schema_path.exists():
        print(f"Missing schema file: {schema_path}")
        return 1

    try:
        manifest = validate_manifest(manifest_path)
    except Exception as exc:
        print(exc)
        return 1

    for path in [Path(path) for path in args.required_markdown + args.required_example_paths]:
        try:
            validate_readable(path)
        except Exception as exc:
            print(exc)
            return 1

    try:
        validate_example_freshness(Path("."))
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_package_urls(manifest_path, pyproject_path)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_version_contract(manifest_path, pyproject_path)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_embedding_extra(pyproject_path)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_license_notice_and_contact(manifest_path.parent.parent / "LICENSE")
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_release_changelog_alignment(manifest_path, pyproject_path)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_distribution_version(manifest_path, pyproject_path)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_install_guidance(
            [
                manifest_path.parent.parent / "README.md",
                manifest_path.parent.parent / "CLAIMS.md",
                manifest_path.parent.parent / "REPLICATION.md",
            ]
        )
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_token_shock_guidance(
            [
                manifest_path.parent.parent / "README.md",
                manifest_path.parent / "report_schema.md",
                manifest_path.parent / "quality_gates.md",
                manifest_path.parent / "release_checklist.md",
            ]
        )
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_install_modes(manifest)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_runtime_guarantees(manifest)
    except Exception as exc:
        print(exc)
        return 1

    try:
        validate_ci_workflow_contracts(Path("."))
    except Exception as exc:
        print(exc)
        return 1

    if args.skip_schema:
        return 0

    try:
        examples = manifest.get(
            "json_schema_examples",
            ["examples/single_report_example.json", "examples/batch_report_example.jsonl"],
        )
        validate_schema_examples(schema_path, examples)
    except Exception as exc:
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

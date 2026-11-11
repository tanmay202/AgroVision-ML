"""
Regression Test Suite: Member 1 Portability and Root Orchestrator Repair
========================================================================
Validates:
1. Dynamic Member 1 root resolution.
2. Dynamic monorepo root resolution.
3. Absence of developer-specific / machine-specific hardcoded paths.
4. Correct input/output directory resolution inside Member 1.
5. Invocation and execution from different working directories (root, m1, scripts, tmp).
6. Root orchestrator references only existing Member 1 scripts (no obsolete modules).
7. No accidental Tea pipeline execution in Member 1.
8. No accidental modifications to Member 2.
9. Safe ASCII output (no UnicodeEncodeError on Windows console).
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
M1_ROOT = TEST_DIR.parent
PROJECT_ROOT = M1_ROOT.parent

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(M1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(M1_ROOT / "scripts"))

from shared.paths import (
    get_project_root,
    get_member1_root,
    get_member2_root,
    get_m1_data_dir,
    get_m1_raw_dir,
    get_m1_processed_dir,
    get_m1_models_dir,
    get_m1_scripts_dir,
)


def test_dynamic_member1_root_resolution():
    """Verify dynamic Member 1 root resolution via shared.paths and Path(__file__)."""
    m1_root = get_member1_root()
    assert m1_root.exists(), f"Member 1 root does not exist: {m1_root}"
    assert m1_root.is_dir()
    assert (m1_root / "scripts").is_dir()
    assert (m1_root / "data").is_dir()
    assert (m1_root / "models").is_dir()
    assert m1_root.name == "member1-yield-model"

    # Verify script-relative resolution matches
    script_m1_root = (M1_ROOT / "scripts" / "diagnose_models.py").resolve().parent.parent
    assert script_m1_root == m1_root


def test_dynamic_monorepo_root_resolution():
    """Verify dynamic monorepo root resolution."""
    proj_root = get_project_root()
    assert proj_root.exists()
    assert (proj_root / "run_all.py").is_file()
    assert (proj_root / "member1-yield-model").is_dir()
    assert (proj_root / "member2-price-volatility").is_dir()
    assert (proj_root / "shared").is_dir()


def test_absence_of_machine_specific_paths():
    """Verify that no developer-specific or hardcoded absolute machine paths exist."""
    forbidden_terms = [
        "Arunava",
        r"C:\Arunava",
        r"C:\Users",
        "/home/",
    ]

    files_to_check = []
    # Check all scripts in member1-yield-model/scripts/
    for p in (M1_ROOT / "scripts").glob("*.py"):
        files_to_check.append(p)
    # Check all other tests in member1-yield-model/tests/
    for p in (M1_ROOT / "tests").glob("*.py"):
        if p.name != Path(__file__).name:
            files_to_check.append(p)
    # Check root orchestrator
    files_to_check.append(PROJECT_ROOT / "run_all.py")
    # Check shared paths
    files_to_check.append(PROJECT_ROOT / "shared" / "paths.py")

    violations = []
    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_terms:
            if term.lower() in content.lower():
                violations.append(f"{fpath.name}: contains forbidden term '{term}'")

    assert not violations, f"Machine-specific path violations found:\n" + "\n".join(violations)


def test_correct_input_output_directory_resolution():
    """Verify input/output directory resolution strictly targets Member 1 directories."""
    m1_root = get_member1_root()
    m2_root = get_member2_root()

    data_dir = get_m1_data_dir()
    raw_dir = get_m1_raw_dir()
    processed_dir = get_m1_processed_dir()
    models_dir = get_m1_models_dir()
    scripts_dir = get_m1_scripts_dir()

    assert data_dir.is_relative_to(m1_root)
    assert not data_dir.is_relative_to(m2_root)
    assert raw_dir.is_relative_to(m1_root)
    assert processed_dir.is_relative_to(m1_root)
    assert models_dir.is_relative_to(m1_root)
    assert scripts_dir.is_relative_to(m1_root)

    # Check that paths defined in the repaired scripts resolve correctly
    import clean_rice_yield
    assert Path(clean_rice_yield.INPUT).resolve().is_relative_to(m1_root)
    assert Path(clean_rice_yield.OUTPUT).resolve().is_relative_to(m1_root)

    import aggregate_rice_weather
    assert Path(aggregate_rice_weather.RAW_DIR).resolve().is_relative_to(m1_root)
    assert Path(aggregate_rice_weather.OUTPUT_FILE).resolve().is_relative_to(m1_root)

    import merge_m1_datasets
    assert Path(merge_m1_datasets.YIELD_FILE).resolve().is_relative_to(m1_root)
    assert Path(merge_m1_datasets.WEATHER_FILE).resolve().is_relative_to(m1_root)
    assert Path(merge_m1_datasets.OUTPUT_FILE).resolve().is_relative_to(m1_root)

    import diagnose_models
    assert Path(diagnose_models.BASE).resolve() == m1_root

    import tune_candidates
    assert Path(tune_candidates.BASE).resolve() == m1_root

    import validate_candidates
    assert Path(validate_candidates.BASE).resolve() == m1_root


def test_execution_from_different_working_directories(tmp_path):
    """Verify scripts execute cleanly from repository root, m1 dir, scripts dir, and a foreign tmp dir."""
    working_dirs = [
        PROJECT_ROOT,
        M1_ROOT,
        M1_ROOT / "scripts",
        tmp_path,
    ]

    cli_script = M1_ROOT / "scripts" / "predict_rice_yield.py"

    for cwd in working_dirs:
        cmd = [
            sys.executable,
            str(cli_script),
            "--district", "Hooghly",
            "--season", "Winter",
            "--yield-lag1", "2800.0",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Failed when executing from {cwd}:\n{proc.stderr}\n{proc.stdout}"
        assert "Predicted Yield (kg/ha):" in proc.stdout

    # Test clean_rice_yield function from different working dir into tmp_path
    import clean_rice_yield
    tmp_out = tmp_path / "clean_test.csv"
    res_df = clean_rice_yield.clean_rice_yield(output_path=tmp_out)
    assert tmp_out.exists()
    assert len(res_df) == 1333

    # Test root orchestrator invocation from foreign working directory
    orch_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_all.py"),
        "--member", "1",
    ]
    proc_orch = subprocess.run(
        orch_cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc_orch.returncode == 0, f"run_all.py failed from tmp_path:\n{proc_orch.stderr}\n{proc_orch.stdout}"
    assert "SUCCESS" in proc_orch.stdout

    # Test custom batch output path resolution from foreign directory
    custom_out = tmp_path / "custom_test_preds.csv"
    proc_custom = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_all.py"),
            "--member", "1",
            "--output", str(custom_out),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc_custom.returncode == 0, f"Custom output run failed:\n{proc_custom.stderr}"
    assert custom_out.exists(), f"Output file was not created: {custom_out}"


def test_root_orchestrator_references_only_existing_scripts():
    """Verify run_all.py references only existing Member 1 scripts and no obsolete modules."""
    run_all_content = (PROJECT_ROOT / "run_all.py").read_text(encoding="utf-8")

    # Obsolete module references must be absent
    obsolete_modules = [
        "src.data_loader",
        "src.data_cleaner",
        "src.feature_engineer",
        "src.model_trainer",
        "src.model_evaluator",
    ]
    for obs in obsolete_modules:
        assert obs not in run_all_content, f"Obsolete module reference found in run_all.py: {obs}"

    # Referenced script files must physically exist
    referenced_scripts = [
        "predict_rice_yield.py",
        "train_and_evaluate_improved.py",
        "validate_candidates.py",
        "tune_candidates.py",
        "diagnose_models.py",
    ]
    for script_name in referenced_scripts:
        assert (M1_ROOT / "scripts" / script_name).is_file(), f"Referenced script missing: {script_name}"


def test_no_accidental_tea_execution():
    """Verify Member 1 orchestration does not depend on or execute Tea datasets."""
    run_all_content = (PROJECT_ROOT / "run_all.py").read_text(encoding="utf-8")

    # In Member 1 section of run_all.py, tea_cleaned.csv must not appear
    m1_section = run_all_content.split("def run_member1")[1].split("def run_member2")[0]
    assert "tea_cleaned.csv" not in m1_section
    assert "tea" not in m1_section.lower() or "commodity" in m1_section.lower()


def test_no_accidental_member2_modifications():
    """Verify that Member 1 path resolution and orchestrator do not mutate Member 2 files or Member 1 datasets."""
    m2_artifacts = list((PROJECT_ROOT / "member2-price-volatility" / "artifacts").glob("*"))
    m2_state_before = {p: p.stat().st_mtime for p in m2_artifacts}

    # Record state of Member 1 tracked processed datasets
    m1_processed = M1_ROOT / "data" / "processed"
    m1_datasets = [
        m1_processed / "m1_predictions.csv",
        m1_processed / "rice_yield_wb_clean.csv",
        m1_processed / "weather_rice_wb.csv",
        m1_processed / "test.csv",
    ]
    m1_state_before = {p: p.stat().st_mtime for p in m1_datasets if p.exists()}

    # Run Member 1 inference via orchestrator function
    from run_all import run_member1
    success = run_member1(mode="inference")
    assert success is True

    # Assert no mtime changes in Member 2 artifacts
    for p, before_mtime in m2_state_before.items():
        assert p.stat().st_mtime == before_mtime, f"Member 2 file {p} was modified!"

    # Assert no mtime changes in Member 1 tracked datasets
    for p, before_mtime in m1_state_before.items():
        assert p.stat().st_mtime == before_mtime, f"Member 1 dataset {p.name} was mutated during inference!"


def test_safe_ascii_output():
    """Verify that root orchestrator output uses ASCII-only characters without UnicodeEncodeError."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_all.py"),
        "--member", "1",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    # Must encode cleanly to standard ascii and cp1252 (Windows default)
    try:
        proc.stdout.encode("ascii")
    except UnicodeEncodeError as e:
        pytest.fail(f"run_all.py output contains non-ASCII characters: {e}")

    try:
        proc.stdout.encode("cp1252")
    except UnicodeEncodeError as e:
        pytest.fail(f"run_all.py output fails cp1252 encoding: {e}")

    assert "SUCCESS" in proc.stdout
    assert "SKIPPED" in proc.stdout
    assert "\u2713" not in proc.stdout
    assert "\u2717" not in proc.stdout

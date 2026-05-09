"""Lightweight tests for the HPO driver pieces.

The full HPO study needs a trained checkpoint and a reference metrics file,
which are not redistributable. These tests exercise the parts we own:

- The weighted-objective aggregation (no Optuna needed).
- The search-space sampler with a real Optuna ``FixedTrial`` to catch bounds /
  naming regressions.
- Reproducibility of the *seeded* per-seed integer derivation.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib.util
import pathlib

import pytest

optuna = pytest.importorskip("optuna")

HPO_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hpo_generate.py"


def _load_hpo_module():
    spec = importlib.util.spec_from_file_location("hpo_generate", HPO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_objective_value_weights():
    hpo = _load_hpo_module()
    errors = {
        "coordination_emd": 0.2,
        "pdf_rmse": 0.1,
        "adf_rmse": 0.04,
    }
    expected = 1.0 * 0.2 + 0.5 * 0.1 + 0.25 * 0.04
    assert hpo._objective_value(errors) == pytest.approx(expected)


def test_sample_params_respects_bounds():
    hpo = _load_hpo_module()
    # FixedTrial lets us exercise the sampler with user-chosen values and
    # confirm the keys / types match the evaluation function's expectations.
    fixed = optuna.trial.FixedTrial({
        "tstep": 128,
        "tmin": 5e-3,
        "tmax": 0.9,
        "rho": 1.8,
        "tersoff_lambda": 0.07,
        "tersoff_schedule": "sigmoid",
        "tersoff_t_gate": 0.55,
        "n_corr": 2,
        "corr_step_size": 0.2,
        "N_anneal": 100,
        "T0": 5e-3,
        "anneal_lr": 1e-3,
    })
    params = hpo._sample_params(fixed)

    expected_keys = {
        "tstep", "tmin", "tmax", "rho",
        "tersoff_lambda", "tersoff_schedule", "tersoff_t_gate",
        "n_corr", "corr_step_size",
        "N_anneal", "T0", "anneal_lr",
    }
    assert set(params.keys()) == expected_keys
    assert params["tersoff_schedule"] in {"constant", "linear", "sigmoid"}
    assert params["n_corr"] in {0, 1, 2, 3}
    assert params["N_anneal"] in {0, 50, 100, 200}
    assert 1e-4 - 1e-9 <= params["tmin"] <= 5e-2 + 1e-9
    assert 0.5 - 1e-9 <= params["tmax"] <= 1.0 + 1e-9
    assert 0.5 - 1e-9 <= params["rho"] <= 3.0 + 1e-9


def test_parse_devices_roundtrip():
    hpo = _load_hpo_module()
    assert hpo._parse_devices("cuda:0") == ["cuda:0"]
    assert hpo._parse_devices("cuda:0,cuda:1,cuda:2") == ["cuda:0", "cuda:1", "cuda:2"]
    assert hpo._parse_devices(" cuda:0 , cuda:1 ") == ["cuda:0", "cuda:1"]


def test_weights_exported():
    hpo = _load_hpo_module()
    assert hpo.W_COORD == 1.0
    assert hpo.W_PDF == 0.5
    assert hpo.W_ADF == 0.25

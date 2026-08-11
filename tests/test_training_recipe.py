from __future__ import annotations

from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "run_dapo.sh").read_text(encoding="utf-8")


def test_gspo_policy_objective_is_configured() -> None:
    assert "--advantage-estimator gspo" in SCRIPT
    assert "--eps-clip 0.0003" in SCRIPT
    assert "--eps-clip-high 0.0004" in SCRIPT
    assert "--advantage-estimator grpo" not in SCRIPT


def test_gspo_uses_sequence_level_reduction_without_token_tis() -> None:
    assert "--calculate-per-token-loss" not in SCRIPT
    assert "--use-tis" not in SCRIPT
    assert "--custom-tis-function-path" not in SCRIPT


def test_formal_gspo_uses_four_updates_and_keeps_r3() -> None:
    assert 'NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-4}"' in SCRIPT
    assert '--num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT"' in SCRIPT
    assert 'ENABLE_R3="${ENABLE_R3:-1}"' in SCRIPT
    assert "--use-rollout-routing-replay" in SCRIPT


def test_smoke_runs_four_steps_with_one_intermediate_eval_and_final_save() -> None:
    assert 'TOTAL_STEPS="${TOTAL_STEPS:-4}"' in SCRIPT
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-4}"' in SCRIPT
    assert 'EVAL_INTERVAL="${EVAL_INTERVAL:-3}"' in SCRIPT
    assert 'EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-0}"' in SCRIPT
    assert 'EVAL_N="${EVAL_N:-2}"' in SCRIPT
    assert "--skip-eval-before-train" in SCRIPT

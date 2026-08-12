"""The synthetic-negative builder's own guards.

`test_synthetic_negatives.py` covers the constructions. This covers the script
that turns them into training rows, and specifically the one failure the
constructions cannot see: an arm that reduces the baseline correlation while the
model learns something else entirely off a presence mask.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _builder():
    spec = importlib.util.spec_from_file_location(
        "_build_synthetic_negatives",
        Path(__file__).resolve().parents[1] / "scripts" / "build_synthetic_negatives.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _builder()


def frame(rate: float, n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({"dv_usable": [True] * int(rate * n) + [False] * (n - int(rate * n))})


def test_matching_dv_rates_pass_and_report_the_gap():
    assert builder._assert_no_dv_shortcut(frame(0.93), frame(0.93)) == pytest.approx(0.0)


def test_a_small_gap_is_tolerated():
    assert builder._assert_no_dv_shortcut(frame(0.88), frame(0.93)) < builder.MAX_DV_RATE_GAP


def test_fully_masked_negatives_against_measured_real_rows_raise():
    """The trap `--dv-policy mask` walks into. Every synthetic negative mask-off
    against 93% of real rows mask-on means 'mask off' IS the label: the baseline
    correlation collapses beautifully and the intervention caused none of it."""
    with pytest.raises(ValueError, match="separates the classes on its own"):
        builder._assert_no_dv_shortcut(frame(0.0), frame(0.93))


def test_the_message_names_the_flag_that_fixes_it():
    with pytest.raises(ValueError, match="--dv-policy inherit"):
        builder._assert_no_dv_shortcut(frame(0.0), frame(0.93))


def test_a_frame_without_the_mask_column_raises_rather_than_skipping_the_check():
    """Silently skipping is how a guard stops being one."""
    with pytest.raises(KeyError, match="dv_usable"):
        builder._assert_no_dv_shortcut(pd.DataFrame({"label": [0, 0]}), frame(0.93))


def test_synthetic_tic_ids_cannot_collide_with_a_real_one():
    """Numbered downwards from -1: a positive offset collides with a real TIC
    the day the catalogue grows past it."""
    assert builder.SYNTHETIC_TIC_BASE < 0
    assert all(builder.SYNTHETIC_TIC_BASE - i < 0 for i in range(10_000))

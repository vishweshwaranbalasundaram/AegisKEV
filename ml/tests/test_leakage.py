"""Leakage, right-censoring and chronological-split tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.features.advanced_features import (
    add_domain_interaction_features,
    compute_matured_causal_group_prior,
)
from ml.features.build_dataset import (
    apply_censoring,
    assert_no_cisa_columns,
    assert_split_monotonic,
    assign_target,
    chronological_split,
    exclude_already_kev,
    finalize_features,
)


# --------------------------------------------------------------------------- #
# Causal prior: labels are revealed only after the 180-day window matures
# --------------------------------------------------------------------------- #
def _prior_frame(groups, dates, labels):
    return pd.DataFrame(
        {
            "group": groups,
            "effective_anchor_date": pd.to_datetime(dates),
            "y_180d": labels,
        }
    )


def test_causal_prior_has_no_history_before_maturation():
    df = _prior_frame(
        ["g1", "g1", "g2", "g1"],
        ["2021-01-01", "2021-01-01", "2021-01-01", "2021-07-01"],
        [1.0, 0.0, 1.0, 0.0],
    )
    out = compute_matured_causal_group_prior(df, "group", "g", m=20.0)

    # Everything anchored 2021-01-01 matures 2021-06-30, so no label is visible.
    assert out.loc[0, "g_hist_count"] == 0
    assert out.loc[1, "g_hist_count"] == 0
    assert out.loc[2, "g_hist_count"] == 0
    assert out.loc[0, "g_kev_prior"] == 0.0
    assert out.loc[2, "g_kev_prior"] == 0.0

    # On 2021-07-01 only rows 0 and 1 (g1) are mature; global rate is 2/3.
    expected = (1 + 20.0 * (2.0 / 3.0)) / (2 + 20.0)
    assert out.loc[3, "g_hist_count"] == 2
    assert out.loc[3, "g_kev_prior"] == pytest.approx(expected)


def test_causal_prior_reveals_on_maturation_day_not_before():
    df = _prior_frame(
        ["g", "g", "g"],
        ["2021-01-01", "2021-06-29", "2021-06-30"],
        [1.0, 0.0, 0.0],
    )
    out = compute_matured_causal_group_prior(df, "group", "g", m=20.0)

    # First label matures exactly on 2021-06-30.
    assert out.loc[1, "g_hist_count"] == 0
    assert out.loc[2, "g_hist_count"] == 1


def test_causal_prior_batches_same_day_disclosures():
    df = _prior_frame(
        ["g", "g", "g", "g"],
        ["2020-01-01", "2021-07-01", "2021-07-01", "2022-01-01"],
        [0.0, 1.0, 0.0, 0.0],
    )
    out = compute_matured_causal_group_prior(df, "group", "g", m=20.0)

    # Same-day rows see the identical pre-day state and never each other.
    assert out.loc[1, "g_hist_count"] == out.loc[2, "g_hist_count"] == 1
    assert out.loc[1, "g_kev_prior"] == out.loc[2, "g_kev_prior"] == 0.0

    # A year later both same-day labels have matured (2021-12-28).
    expected = (1 + 20.0 * (1.0 / 3.0)) / (3 + 20.0)
    assert out.loc[3, "g_hist_count"] == 3
    assert out.loc[3, "g_kev_prior"] == pytest.approx(expected)


def test_causal_prior_future_label_never_reaches_earlier_rows():
    df = _prior_frame(
        ["g", "g"],
        ["2021-01-01", "2021-01-02"],
        [1.0, 0.0],
    )
    out = compute_matured_causal_group_prior(df, "group", "g", m=20.0)

    assert out["g_hist_count"].tolist() == [0, 0]
    assert out["g_kev_prior"].tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Target, exclusion and right-censoring
# --------------------------------------------------------------------------- #
def test_target_only_counts_kev_entries_inside_the_180_day_window():
    df = pd.DataFrame(
        {
            "effective_anchor_date": pd.to_datetime(["2022-01-01"] * 5),
            "date_added": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-06-30", "2022-07-01", None]
            ),
        }
    )
    out = assign_target(df)

    # Same-day, outside-window and never-in-KEV rows are all negatives.
    assert out["y_180d"].tolist() == [0, 1, 1, 0, 0]


def test_already_kev_at_anchor_is_excluded():
    df = pd.DataFrame(
        {
            "effective_anchor_date": pd.to_datetime(["2022-01-01"] * 3),
            "date_added": pd.to_datetime(["2021-12-31", "2022-01-01", None]),
        }
    )
    out = exclude_already_kev(df)

    assert len(out) == 1
    assert out["date_added"].isna().all()


def test_right_censoring_keeps_rows_with_a_full_observation_window():
    kev_cutoff = pd.Timestamp("2025-03-17")
    df = pd.DataFrame(
        {
            "effective_anchor_date": pd.to_datetime(
                ["2024-09-17", "2024-09-18", "2024-09-19"]
            )
        }
    )
    matured, unmatured, cutoff = apply_censoring(df, kev_cutoff)

    assert cutoff == pd.Timestamp("2024-09-18")
    assert len(matured) == 2
    assert len(unmatured) == 1
    assert unmatured.iloc[0]["effective_anchor_date"] == pd.Timestamp("2024-09-19")


# --------------------------------------------------------------------------- #
# Chronological splits
# --------------------------------------------------------------------------- #
def test_fixed_split_windows_are_monotonic_and_cover_every_row():
    dates = [
        "2023-01-01",
        "2024-06-30",
        "2024-07-01",
        "2024-11-30",
        "2024-12-01",
        "2025-03-16",
        "2025-03-17",
        "2025-06-01",
    ]
    df = pd.DataFrame(
        {"effective_anchor_date": pd.to_datetime(dates), "row": range(len(dates))}
    )
    splits, scheme = chronological_split(df)

    assert scheme == "fixed"
    assert splits["train"]["row"].tolist() == [0, 1]
    assert splits["val_hpo"]["row"].tolist() == [2, 3]
    assert splits["val_cal"]["row"].tolist() == [4, 5]
    assert splits["test"]["row"].tolist() == [6, 7]
    assert sum(len(frame) for frame in splits.values()) == len(df)
    assert_split_monotonic(splits)


def test_quantile_fallback_when_fixed_windows_are_empty():
    dates = pd.date_range("2025-01-01", periods=20, freq="14D")
    df = pd.DataFrame({"effective_anchor_date": dates})
    splits, scheme = chronological_split(df)

    assert scheme == "quantile"
    assert all(len(frame) > 0 for frame in splits.values())
    assert sum(len(frame) for frame in splits.values()) == len(df)
    assert_split_monotonic(splits)


def test_split_monotonic_assertion_catches_reversed_boundaries():
    df = pd.DataFrame(
        {
            "effective_anchor_date": pd.to_datetime(["2024-01-01", "2023-01-01"]),
            "y_180d": [0, 0],
        }
    )
    splits = {
        "train": df.iloc[[0]].copy(),  # 2024-01-01
        "val_hpo": df.iloc[[1]].copy(),  # 2023-01-01: reversed on purpose
        "val_cal": df.iloc[[1]].copy(),
        "test": df.iloc[[1]].copy(),
    }
    with pytest.raises(AssertionError, match="non-monotonic"):
        assert_split_monotonic(splits)


# --------------------------------------------------------------------------- #
# Leaky columns
# --------------------------------------------------------------------------- #
def test_cisa_columns_are_rejected():
    with pytest.raises(AssertionError, match="CISA"):
        assert_no_cisa_columns(pd.DataFrame({"cisaExploitAdd": ["2021-12-10"]}))
    assert_no_cisa_columns(pd.DataFrame({"cvss_base_score": [9.8]}))


def test_finalize_features_drops_date_added_and_cisa_fields():
    df = pd.DataFrame(
        {
            "cisaActionDue": ["2021-12-24"],
            "date_added": [pd.Timestamp("2021-12-10")],
            "cvss_base_score": [10.0],
        }
    )
    out = finalize_features(df)

    assert list(out.columns) == ["cvss_base_score"]


# --------------------------------------------------------------------------- #
# Domain interactions
# --------------------------------------------------------------------------- #
def test_domain_interactions_are_computed_without_future_data():
    df = pd.DataFrame(
        {
            "anchor_date": pd.to_datetime(["2021-01-01"] * 3),
            "epss": [0.1, 0.5, 0.9],
            "cvss_base_score": [5.0, 9.0, 10.0],
            "has_exploit_tag": [True, False, True],
            "has_patch_tag": [False, False, True],
            "is_network_av": [True, True, False],
            "is_low_ac": [True, True, True],
            "is_priv_none": [True, False, True],
            "is_ui_none": [True, True, True],
            "cvss_exploitability_score": [3.9, 3.9, 0.0],
            "cvss_impact_score": [5.9, 0.0, 5.9],
        }
    )
    out = add_domain_interaction_features(df)

    assert out["epss_same_day_pct"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert out["exploit_without_patch_race"].tolist() == [1, 0, 0]
    assert out["zero_click_network_flaw"].tolist() == [1, 0, 0]
    assert out["epss_minus_cvss_norm"].iloc[0] == pytest.approx(0.1 - 0.5)
    assert out["epss_x_cvss"].iloc[2] == pytest.approx(0.9)
    assert out["exploit_to_impact_ratio"].iloc[0] == pytest.approx(3.9 / 5.9)
    assert pd.isna(out["exploit_to_impact_ratio"].iloc[1])  # zero impact guarded

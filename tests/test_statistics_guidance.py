"""STATISTICS.txt and the tank-by-minute table (Audit I, I9): guidance for
the group comparison, the primary measure, minute by minute as repeated
measures, and when a mixed model on the fish rows does and does not help."""
import numpy as np
import pandas as pd
import pytest

from fish_analyzer import results, results_export

from test_results import two_groups  # noqa: F401  (fixture)


@pytest.fixture
def exported(two_groups, tmp_path):  # noqa: F811
    return results_export.export_results_folder(two_groups, None, tmp_path)


@pytest.fixture
def guidance(exported):
    return (exported / "STATISTICS.txt").read_text(encoding="utf-8")


def test_the_minute_table_has_one_row_per_tank_per_minute(two_groups, exported):  # noqa: F811
    per_tank = pd.read_csv(exported / "results_by_minute_per_tank.csv")
    fish = results.minute_table(two_groups)

    assert list(per_tank.columns[:4]) == ["Group", "Session", "Tracking", "Minute"]
    assert not per_tank.duplicated(["Session", "Minute"]).any()
    expected = fish.groupby(["Session", "Minute"])["MedianSpeed"].mean()
    got = per_tank.set_index(["Session", "Minute"])["MedianSpeed"]
    assert np.allclose(got.sort_index(), expected.sort_index())


def test_it_names_one_primary_measure(guidance):
    assert "DistancePerTrackedMin" in guidance
    assert "primary" in guidance
    assert "Holm" in guidance


def test_it_treats_minute_by_minute_as_repeated_measures(guidance):
    assert "repeated measures" in guidance
    assert "results_by_minute_per_tank.csv" in guidance
    assert "Group x Minute" in guidance


def test_it_says_when_a_mixed_model_helps_and_when_it_fails(guidance):
    assert "GAMLj" in guidance
    assert "cluster = Session" in guidance
    assert "singular fit" in guidance
    assert "without identities" in guidance


def test_it_warns_when_a_group_has_one_tank(guidance):
    assert "treated has 1 tank" in guidance


def test_the_readme_sends_the_reader_to_it(exported):
    readme = (exported / "README.txt").read_text(encoding="utf-8")
    assert "STATISTICS.txt" in readme.split("results_per_fish.csv")[0]


def test_the_prism_readme_says_how_to_test_minute_by_minute(exported):
    prism = (exported / "prism" / "PRISM_README.txt").read_text(encoding="utf-8")
    assert "repeated-measures" in prism and "mixed-effects" in prism

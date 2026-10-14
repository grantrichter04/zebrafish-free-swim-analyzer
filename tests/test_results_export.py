"""The Results export: a folder with a CSV for every view on the tab."""
import numpy as np
import pandas as pd
import pytest

from fish_analyzer import results, results_export

from test_results import two_groups  # noqa: F401  (fixture)


@pytest.fixture
def exported(two_groups, tmp_path):  # noqa: F811
    return results_export.export_results_folder(two_groups, None, tmp_path)


def test_the_folder_holds_a_file_for_every_view(exported):
    assert sorted(p.name for p in exported.iterdir()) == sorted(results_export.FILES)
    assert exported.name.startswith("free_swim_results_")


def test_the_minute_file_is_the_minute_table(two_groups, exported):  # noqa: F811
    minutes = pd.read_csv(exported / "results_by_minute.csv")
    expected = results.minute_table(two_groups)
    assert len(minutes) == len(expected)
    assert np.allclose(minutes["MedianSpeed"], expected["MedianSpeed"])
    assert set(minutes["Tracking"]) == {"identities"}


def test_the_speed_histogram_adds_up_to_every_tracked_frame(two_groups, exported):  # noqa: F811
    histogram = pd.read_csv(exported / "speed_histogram.csv")
    for sample in results.speed_samples(two_groups):
        rows = histogram[(histogram["Session"] == sample.session)
                         & (histogram["Fish"].astype(str) == sample.fish)]
        assert rows["Frames"].sum() == len(sample.speeds)
        assert rows["Frames_pct"].sum() == pytest.approx(100.0)
        cumulative = rows["Frames_pct"].cumsum().to_numpy()
        below = rows["SpeedTo"].to_numpy()
        # The cumulative curve read from the bins matches the speeds.
        assert np.allclose(cumulative, [np.mean(sample.speeds < b) * 100 for b in below])


def test_the_density_file_is_each_sessions_share_of_time(exported):
    density = pd.read_csv(exported / "position_density.csv")
    assert density.groupby("Session")["Time_pct"].sum().to_numpy() == pytest.approx(100.0)


def test_bin_widths_are_round_numbers():
    assert results_export.nice_bin_width(6.0) == 0.1
    assert results_export.nice_bin_width(4.3) == 0.05
    assert results_export.nice_bin_width(30.0) == 0.5


def test_the_prism_comparison_table_is_one_column_per_group_of_tank_means(
        two_groups, exported):  # noqa: F811
    prism = exported / "prism"
    table = pd.read_csv(prism / "group_comparison_MedianSpeed.csv")
    means = results.session_means(results.results_table(two_groups), ["MedianSpeed"])

    assert list(table.columns) == ["control", "treated"]
    assert table["control"].dropna().to_list() == pytest.approx(
        means.loc[means["Group"] == "control", "MedianSpeed"].to_list(), rel=1e-4)
    assert table["treated"].isna().sum() == 1, "the one treated tank, then a blank"
    rows = pd.read_csv(prism / "sessions_in_each_row.csv")
    assert rows["control"].to_list() == ["control_1", "control_2"]


def test_the_prism_minute_table_gives_every_group_the_same_number_of_columns(
        exported):
    table = pd.read_csv(exported / "prism" / "by_minute_MedianSpeed.csv")

    assert list(table.columns) == ["Minute", "control: control_1",
                                   "control: control_2", "treated: treated_1",
                                   "treated: (none)"]
    assert table["treated: (none)"].isna().all()
    assert table["Minute"].to_list() == list(range(1, len(table) + 1))
    assert (exported / "prism" / "PRISM_README.txt").is_file()

"""METHODS.txt: a draft methods paragraph and references, filled in with the
export's own versions, settings, scale and n (Audit I, I6)."""
import pytest

from fish_analyzer import provenance, results_export

from test_results import two_groups  # noqa: F401  (fixture)
from test_tracking_settings import two_sessions  # noqa: F401  (fixture)


@pytest.fixture
def methods(two_sessions, tmp_path):  # noqa: F811
    folder = results_export.export_results_folder(
        two_sessions, {"early": "control", "plain": "treated"}, tmp_path)
    return (folder / "METHODS.txt").read_text(encoding="utf-8")


def test_it_names_the_software_and_versions(methods):
    assert "idtracker.ai 6.0.14" in methods
    assert f"Free Swim Analyzer {provenance.analyzer_version()}" in methods


def test_it_states_the_analysis_settings_used(methods):
    assert "running mean over 0.1 s" in methods
    assert "tracked in less than 80% of frames" in methods
    assert "faster than 25 BL/s" in methods
    assert "slower than 0.5 BL/s for at least 1 s" in methods
    assert "faster than 1 BL/s" in methods
    assert "15% of the tank's shorter side" in methods


def test_it_states_the_scale(methods):
    assert "body lengths" in methods
    assert "40 pixels" in methods


def test_it_says_what_happened_to_tracking(methods):
    assert "finished early for 1 of 2 sessions" in methods
    assert "1 of 2 sessions" in methods and "validator" in methods


def test_it_states_the_unit_of_analysis_and_n(methods):
    assert "the tank was the experimental unit" in methods
    assert "control 1, treated 1" in methods


def test_it_lists_the_references_with_dois(methods):
    for doi in ("10.7554/eLife.107602", "10.1038/s41592-018-0295-5",
                "10.1083/jcb.202001064", "10.21105/joss.03202"):
        assert doi in methods


def test_sessions_on_different_scales_are_not_described_as_one_scale(
        two_groups, tmp_path):  # noqa: F811
    """control_1 is at 40 px per BL and control_2 at 52."""
    folder = results_export.export_results_folder(two_groups, None, tmp_path)
    text = (folder / "METHODS.txt").read_text(encoding="utf-8")
    assert "for every session" not in text
    assert "different scales (40, 52 pixels per BL)" in text


def test_it_gives_the_recording_length(methods):
    assert "recordings of 0.33 min" in methods


def test_without_session_folders_it_does_not_invent_tracking_details(
        two_groups, tmp_path):  # noqa: F811
    folder = results_export.export_results_folder(two_groups, None, tmp_path)
    text = (folder / "METHODS.txt").read_text(encoding="utf-8")
    assert "[idtracker.ai version]" in text
    assert "finished early" not in text

"""scripts/verify_on_session.py, the real-data re-check the README points
developers to, must keep running as measures are renamed. It crashed on
'speed_p99' after top speed became the 95th percentile (Audit I, I5)."""
import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location(
        "verify_on_session", REPO / "scripts" / "verify_on_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_runs_to_the_end_on_a_session(synthetic_npy, tmp_path, capsys):
    session = tmp_path / "session_synthetic"
    (session / "trajectories").mkdir(parents=True)
    shutil.copy(synthetic_npy, session / "trajectories" / "trajectories.npy")

    assert _script().main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "B17" in out and "withdrawn columns absent" in out

# Extras

Standalone scripts that are not part of the Free Swim Analyzer app. Nothing in
`fish_analyzer/` imports them, the app has no button for them, and the test
suite does not cover them.

| Script | What it does |
|---|---|
| `fish_posture_analyzer.py` | Extracts a midline/skeleton from idtracker.ai's per-fish image crops |
| `head_detection/` | Head-versus-tail detection and turn analysis, with validation videos |

They need more than the app does:

```bash
pip install -e ".[standalone]"
```

`head_detection/` also needs idtracker.ai, which `install.bat` installs.

They are kept because head direction is what the withdrawn turning metrics
would need in order to come back; see
[docs/audit/AUDIT_B_CORRECTNESS.md](../docs/audit/AUDIT_B_CORRECTNESS.md).

"""Running one data source must not overwrite another source's results.

A bare artifacts/ablation.json meant `--source jaad` clobbered the CARLA numbers the
write-up is built on. Only git had the original. The path is now derived from the
source, and this pins that.
"""
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_ablation.py"


def test_ablation_artifact_is_named_after_the_source():
    src = SCRIPT.read_text()
    assert 'f"ablation_{cfg.data.source.lower()}.json"' in src, (
        "the ablation artifact must be per-source; a bare ablation.json lets one "
        "source silently overwrite another"
    )


def test_the_bare_alias_is_only_written_for_carla():
    """The plain name is kept so existing readers work, but only carla may claim it."""
    src = SCRIPT.read_text()
    m = re.search(r'if cfg\.data\.source\.lower\(\) == "carla":\s*\n\s*\(art / "ablation\.json"\)', src)
    assert m, "the bare ablation.json alias must be guarded to the carla source"

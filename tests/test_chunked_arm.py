import json
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def test_arm_targets_the_chunk_end_not_schedule_horizon():
    result = subprocess.run(
        [sys.executable, "viz/arm.py", "--id", "test-chunk-dryrun",
         "--family", "test", "--goal", "test", "--dry-run", "--",
         "--steps", "16000", "--stop-at-step", "4000",
         "--preset", "pythia160_2k"],
        cwd=LAB, text=True, capture_output=True, check=True)
    spec = json.loads(result.stdout)["spec"]
    assert spec["target_steps"] == 4000
    assert spec["schedule_steps"] == 16000
    assert spec["chunk_steps"] == 4000 and spec["chunk_index"] == 1
    assert spec["log"] == "logs/test-chunk-dryrun_chunk01.log"


def test_chunked_arm_rejects_auto_chain_for_explicit_pause(tmp_path):
    result = subprocess.run(
        [sys.executable, "viz/arm.py", "--id", "test-chunk-dryrun",
         "--family", "test", "--goal", "test", "--dry-run",
         "--next-job", "train_job.next.json", "--", "--steps", "16000",
         "--stop-at-step", "4000", "--preset", "pythia160_2k"],
        cwd=LAB, text=True, capture_output=True)
    assert result.returncode != 0
    assert "pause" in result.stderr.lower()

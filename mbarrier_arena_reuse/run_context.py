"""Run the minimal reproducer in separate CUDA contexts on one idle GPU."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--repeats", type=int, default=300)
parser.add_argument("--hold-ms", type=int, default=10)
args = parser.parse_args()
root = Path(__file__).resolve().parent
subprocess.run(["make", "-s", "build/minimal"], cwd=root, check=True)


def start(invalidate):
    return subprocess.Popen(
        [str(root / "build/minimal"), str(args.device), str(invalidate),
         str(args.repeats), str(args.hold_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def collect(process):
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("timeout is not a reproduction") from None
    if process.returncode not in (0, 2):
        raise RuntimeError(f"CUDA/configuration failure: {stderr}")
    rows = [json.loads(line) for line in stdout.splitlines()]
    return dict(exit_code=process.returncode, result=rows[-1], samples=rows[:-1])


alone = collect(start(0))
print(json.dumps(dict(mode="alone", probe=alone)), flush=True)
if alone["exit_code"]:
    raise RuntimeError("exclusive control failed; check for other GPU contexts")
for trial in range(args.trials):
    for invalidate in (0, 1):
        # The contender uses the correct protocol and shares no address space
        # or allocation with the probe. It cannot directly overwrite the probe.
        peer = start(1)
        probe = start(invalidate)
        try:
            probe_result = collect(probe)
            peer_result = collect(peer)
        finally:
            for process in (probe, peer):
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        print(json.dumps(dict(mode="two_contexts", trial=trial,
                              probe=probe_result, peer=peer_result)), flush=True)
        expected = 2 if invalidate == 0 else 0
        if probe_result["exit_code"] != expected or peer_result["exit_code"] != 0:
            raise RuntimeError("expected old-fails/fixed-passes contrast was not observed")

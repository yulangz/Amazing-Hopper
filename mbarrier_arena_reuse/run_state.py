"""Vary the retired barrier state while a separate, correct context competes."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
root = Path(__file__).resolve().parent
subprocess.run(["make", "-s", "build/minimal", "build/state_probe"], cwd=root, check=True,
               stdout=subprocess.DEVNULL)


def collect(process):
    stdout, stderr = process.communicate(timeout=60)
    if process.returncode not in (0, 2):
        raise RuntimeError(f"CUDA/configuration failure: {stderr}")
    rows = [json.loads(line) for line in stdout.splitlines()]
    return dict(exit_code=process.returncode, result=rows[-1], samples=rows[:-1])


cases = [(False, 1, 1, 1), (True, 0, 1, 1),
         *[(True, 1, count, phases) for count in (1, 7) for phases in (1, 2)],
         (True, 2, 1, 1), (True, 2, 7, 2), (True, 3, 1, 1)]
for paired, mode, count, phases in cases:
    processes = []
    try:
        if paired:
            peer = subprocess.Popen(
                [str(root / "build/minimal"), str(args.device), "1", "300", "10"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(peer)
        probe = subprocess.Popen(
            [str(root / "build/state_probe"), str(args.device), str(mode),
             str(count), str(phases), "100", "20"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(probe)
        result = collect(probe)
        peer_result = collect(peer) if paired else None
        print(json.dumps(dict(paired=paired, probe=result, peer=peer_result)), flush=True)
        expected = 2 if paired and mode in (1, 3) else 0
        if result["exit_code"] != expected or (paired and peer_result["exit_code"] != 0):
            raise RuntimeError("expected corruption/control contrast was not observed")
        if result["result"]["corrupted_guards"]:
            raise RuntimeError("ordinary-data guards corrupted")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()

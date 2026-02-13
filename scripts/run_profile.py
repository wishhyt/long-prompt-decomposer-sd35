#!/usr/bin/env python

import argparse
import json
import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _load_profile(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    if "entry_module" not in profile:
        raise ValueError(f"Profile is missing 'entry_module': {profile_path}")
    default_args = profile.get("default_args", [])
    if not isinstance(default_args, list):
        raise ValueError(f"'default_args' must be a list in profile: {profile_path}")
    return profile


def run_profile(default_profile=None):
    parser = argparse.ArgumentParser(description="Run an LPD script via JSON profile.")
    parser.add_argument("--profile", default=default_profile, help="Path to a JSON profile file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command only.")
    args, passthrough = parser.parse_known_args()

    if not args.profile:
        parser.error("--profile is required")

    profile_path = args.profile
    if not os.path.isabs(profile_path):
        profile_path = os.path.join(ROOT_DIR, profile_path)

    profile = _load_profile(profile_path)
    entry_module = profile["entry_module"]
    default_args = profile.get("default_args", [])

    cmd = [sys.executable, "-m", entry_module, *default_args, *passthrough]

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    completed = subprocess.run(cmd, cwd=ROOT_DIR)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run_profile())

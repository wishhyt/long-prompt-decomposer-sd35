#!/usr/bin/env python
# coding=utf-8

import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from scripts.run_profile import run_profile


def __getattr__(name):
    if name in {"TextDecomposer", "parse_args", "lpd_main"}:
        from scripts.train import lpd_ella_ct5 as _impl

        if name == "lpd_main":
            return _impl.main
        return getattr(_impl, name)
    raise AttributeError(name)


def main():
    return run_profile(default_profile="configs/profiles/data_test.json")


if __name__ == "__main__":
    raise SystemExit(main())

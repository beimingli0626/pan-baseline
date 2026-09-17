"""python -m baselines {train,eval,eval_accum} [flags]: the PAN command, with the baseline
archs registered (importing the package, which `-m` does first, registers them)."""
import importlib
import sys

COMMANDS = ("train", "eval", "eval_accum")

if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    sys.exit(f"usage: python -m baselines {{{','.join(COMMANDS)}}} [flags]")
importlib.import_module(f"pan.planner.{sys.argv.pop(1)}").main()

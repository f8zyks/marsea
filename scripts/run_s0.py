"""S0 (Sec. 12.3): the routing gate.  Routed to the whole-stack sweep of marsea/sweep.py (span-tolerant, sink excluded,
per-example fractions, replication statistic M(a) at (l*, h*)); the earlier single-head/single-token check is retired.
    python scripts/run_s0.py --set "data/ruler/E2_L8192_K8_V1_Q1_d0.5_s0,data/ruler/E2_L8192_K8_V4_Q1_d0.5_s0" --n 40 \
        [--ckpt runs/phaseA_seed0.pt] --update_detector runs/detector.json
"""
import pathlib, runpy, sys
sys.argv[0] = str(pathlib.Path(__file__).with_name("run_s0_sweep.py"))
runpy.run_path(sys.argv[0], run_name="__main__")

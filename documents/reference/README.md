# reference/ — cold-read implementation of spec v3 (2026-09-04)

Written by a second engineer from `marsea_implementation_spec.md` (v3) and paper Sec. 2 ALONE.
CPU PyTorch, `[B,H,n_q,n_k]`, GQA, ragged relations, straight-through relation head, MLP
exclusivity heads, six baselines.

- `test_cold.py`: every invariant of the v3 list, T0/T1/T13, gradient flow — 0 failures.
- `compare_harness.py`: matches `verify_all.py`'s Prop. A loop to 3e-15 (fp64).
- `ambiguities.md`: the 31 points where v3's text was insufficient; all resolved in spec v4+.

Before reuse, bring the code to the current spec (v4.6): B2/MESH step (straight-through, eps = 1,
end on the row scaling), head-input normalisation (§5.2), the detector (§6.7), no packing (§9),
`nu` not detached (§5.3), the v4.5 invariant list (§2). Treat it as a starting point and a test
oracle, not the deliverable.

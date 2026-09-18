"""
MarSea -- Multivariate Relations by Selective Exclusion in Attention.

Implementation of `records/marsea_implementation_spec.md` (v4.6) and paper Sec. 2.
Package layout follows the spec's build order (Sec. 14):

  primitives   Sec. 3   sorted_prefix_stat, sparsemax (custom autograd), entmax, proj_eq, proj_le
  relation     Sec. 4   the pairwise relation E, its straight-through gate, b0 calibration
  heads        Sec. 5.2 column_stats, row_summary, TauK, TauQ
  normalizer   Sec. 5   marsea_normalize (dense reference), Diagnostics, State
  invariants   Sec. 2   INV-1..INV-9 and SAN-1 (asserted under MARSEA_DEBUG=1)
  baselines    Sec. 8   B0 SoftmaxNorm, B1 SoftmaxOneNorm, B2 MESHNorm, B3 KeyOnlyTauNorm,
                        B4 RowEntmaxNorm, B5 MatchedSparsityNorm
  chunked      Sec. 6.6 exact chunked implementation (T11)
  causal       Sec. 7   hierarchical top-K, sealing, frozen-prefix decoding (Sec. 6.4)
  backbone     Sec. 6   the module swap on Qwen2 / Llama, vis construction, context object
  detector     Sec. 6.7 retrieval-head detector (Wu et al. 2024)
  fidelity     Sec. 12  T_j / K_i ground truth, intervals, hit rate, P/R, distractor split
  data/        Sec. 13  RULER, MuSiQue, HotpotQA, IHEval loaders and the 1:1:1 mix
  train        Sec. 9   Phase A / Phase B loop (records/marsea_training_procedure.md)
  evaluate     Sec. 12  teacher-forced and generation evaluation, E-suite
"""
from .primitives import (sorted_prefix_stat, sparsemax_masked, entmax_masked, proj_eq_masked,
                         proj_le_masked, implied_threshold, TOL_QUOTA, NEG_PAD)
from .relation import RelationHead, straight_through
from .heads import column_stats, row_summary, TauK, TauQ, feat, inv_softplus
from .normalizer import MarSeaNormalizer, Diagnostics, State, row_softmax, repeat_kv
from .baselines import (SoftmaxNorm, SoftmaxOneNorm, MESHNorm, KeyOnlyTauNorm, RowEntmaxNorm,
                        MatchedSparsityNorm, make_normalizer)
from .invariants import check_invariants, debug_enabled

__version__ = "0.1.0"
SPEC_VERSION = "4.9"        # D-31 (sink exclusion), D-27 (grids), D-9a (split T_j) are all implemented

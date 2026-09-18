# MarSea — package 2026-09-08

Read first: records/marsea_decisions_register.md (index of standing decisions) and
records/marsea_master_record.md (§§35–49 are the current design; the banner at its top says which
sections carry the standing decisions). The implementation contract is
records/marsea_implementation_spec.md (v4.6); the training loop is records/marsea_training_procedure.md.

  marsea_iclr.tex / .bbl / .pdf      ICLR 2027 draft (41 pp).  NOTE: there is no .bib; the .bbl is hand-maintained.
                                     Build: pdflatex x2.  Do NOT run bibtex (it empties the .bbl).
  tmlr/                              the fuller TMLR fallback draft (89 pp), same rule.
  verify_all.py                      numerical harness: 1,058,450 checks, 0 violations, 62 rows (both drafts' tables reconcile)
  v1_probe.py / .pdf, pr_curve.py    Fig. 2 (E1) and Fig. 3
  records/                           spec v4.6, training procedure, decisions register, master record, audits 4 and 5
  round2/                            engineer's pseudo-code from all sources + the 30-item contradiction list (all resolved)
  reference/                         cold-read implementation of spec v3 (PyTorch), its tests, harness comparison,
                                     ambiguity log; see its README for what to bring to v4 before reuse

# Scripts retained from the submitted version

Every experiment of the submitted version that could be repeated under matched
conditions was repeated for the revision, and only the re-run is reported. The
scripts here produced the four comparisons that Appendix F of the paper retains
because they could not be re-run under the revised protocol:

| Script | Retained result |
|---|---|
| `submitted_additional_medical_baselines.py` | Comparison with additional medical QA baselines on MedQuAD and the iCliniq-style dataset. The baseline values come from single runs under their own training budgets, prompt handling and reference processing, so the table is descriptive only. |
| `submitted_faiss_indexing_comparison.py` | FAISS indexing variant comparison. Produced by the LLaMA-3 retrieval-comparison harness, not by the LLaMA-3.1-8B-Instruct pipeline of the submitted open-ended experiments. Single run, seed 42. |
| `submitted_closedform_baseline_evaluation.py` | Closed-form baseline evaluation and training-loss analysis. |
| `submitted_verifier_variant_comparison.py` | Verifier variant comparison. |
| `submitted_openended_pipeline.py` | The submitted open-ended pipeline itself, kept so the differences the revision documents can be inspected against the code that produced the original numbers. |

These scripts are **not** part of the reproduction path for the main results.
Everything reported in Sections 4.1 to 4.6 of the paper comes from
`scripts/pipeline/` and `experiments/`, which supersede them. They address paths
in the layout of the original working tree and are kept for inspection rather
than for re-execution.

Directory defaults in these files were absolute paths in the original working
tree. They now read `SET_THIS_PATH`, so a run fails immediately and visibly
rather than silently writing to the wrong place.

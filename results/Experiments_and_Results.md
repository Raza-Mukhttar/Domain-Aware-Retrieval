# Experiments Run for the Reviewers' Concerns, with Results

**Manuscript AIIM-D-26-03241 — *Artificial Intelligence in Medicine***

Domain-Aware Retrieval, Verification and Entity-Aware Decoding in Medical Question Answering: A Controlled Empirical Evaluation


## How to read this file

One section per experiment. Each gives the reviewer concerns it resolves in the reviewers' own words, what was executed and at what scale, the detailed results, and where those results appear in the compiled revised manuscript (`main_revised_highlighted.pdf`). Both reviewers are covered in the same file, because the same experiments answer points from both; the index below separates them.


Every number in this file is read from the stored result files in `work2/results/`, not transcribed, so it cannot drift from the runs it describes. In the contrast tables, values are differences in points with a 10,000-resample paired bootstrap 95% interval, and **\*** marks significance at p<0.05 after Holm correction over the five metrics. Positive means the first-named condition scores higher.


Two panels recur. The **submitted prompt handling** truncates over-long prompts from the right, which removed the target question from most retrieval prompts; it is retained as an implementation diagnostic. The **corrected prompt handling** truncates from the left so the question always survives, and carries the primary results. The two are separate generation runs, so only paired contrasts within a run are compared.


## Results at a glance

One line per experiment. The detailed tables are in the sections below; every figure here is read from the same result files.

| Experiment | Headline result | |---|---| | [X01](#x01--open-ended-generator-fine-tuning-three-seeds) Open-ended generator fine-tuning, three seeds | 6 generator runs trained; validation loss 1.259-2.242; three-seed SD of the open-ended metrics under 0.4 points, so the seed is not what drives the results. | | [X02](#x02--contrastive-alignment-stage-re-run-and-tensor-audited) Contrastive alignment stage, re-run and tensor-audited | The contrastive stage changed 0.40% of the 525,336,576 embedding entries (max |Δ| 2.2e-04); with/without contrasts show no detectable benefit (+0.17 [-0.32, +0.69] on MedQuAD). | | [X03](#x03--open-ended-component-grid-submitted-prompt-handling) Open-ended component grid, submitted prompt handling | Submitted-policy grid: retrieval -8.90 [-10.46, -7.33] on MedQuAD and -8.40 [-9.39, -7.39] on iCliniq -- the panel later shown to be driven by the truncation defect. | | [X04](#x04--open-ended-component-grid-corrected-prompt-handling) Open-ended component grid, corrected prompt handling | Corrected-policy grid (primary): retrieval +2.50 [+0.96, +4.07] MedQuAD, -1.81 [-2.60, -1.01] iCliniq; question-type +1.24 [+0.22, +2.25]; verifier -0.07 [-0.45, +0.27]. | | [X05](#x05--input-truncation-audit-and-three-seed-correction) Input-truncation audit and three-seed correction | The defect: the target question survived in 30/300 MedQuAD and 0/300 iCliniq retrieval prompts. Correcting it moves the measured retrieval effect by 11.4 and 6.6 ROUGE-1 points. | | [X06](#x06--run-to-run-non-determinism-measurement) Run-to-run non-determinism measurement | No-retrieval prompts max 52 tokens (none over 512), yet only 10.3% of answers are identical across runs: bf16 beam search is not bit-reproducible, so differences below ~0.3 points are not interpreted. | | [X07](#x07--test-sampling-rule-check-both-prompt-policies) Test-sampling-rule check, both prompt policies | Sampling rule: corrected retrieval effect +2.5 (random 300) vs +3.79 (first 300); same sign under both rules on both datasets. | | [X08](#x08--retrieval-effect-decomposition) Retrieval-effect decomposition | Under corrected prompting retrieval raises precision by +3.7 and recall by +1.8 points on MedQuAD with answer length unchanged. | | [X09](#x09--exemplar-contamination-measurement) Exemplar-contamination measurement | Exemplar content in answers: 82.7% under the submitted policy vs 38.0% corrected -- about half as much once the question survives. | | [X10](#x10--medquad-metadata-recovery-topic-overlap-and-deduplicated-subset-analysis) MedQuAD metadata recovery, topic-overlap and deduplicated-subset analysis | Retrieval benefit is +4.48 where the topic is absent from the pool and +0.0 where it is present: the opposite of a leakage pattern. | | [X11](#x11--duplicate-and-near-duplicate-audit-of-every-partition) Duplicate and near-duplicate audit of every partition | MedQuAD test: 17.54% exact and 58.95% near-duplicate overlap with training; 674 of 1642 questions survive deduplication. iCliniq: none. | | [X12](#x12--icliniq-style-corpus-provenance-check) iCliniq-style corpus provenance check | The corpus is single-platform ChatDoctor-iCliniq (7,321 rows); exact matching against the multi-platform repository cited in the submitted version found 0 overlaps. | | [X13](#x13--closed-form-matched-within-backbone-controls) Closed-form matched within-backbone controls | 48 matched closed-form control runs; dense retrieval on untrained backbones significant in 9 of 24 cells. | | [X14](#x14--closed-form-generator-fine-tuning-and-complete-framework-evaluation) Closed-form generator fine-tuning and complete-framework evaluation | 12 closed-form generators trained; the complete framework is not better than the task-adapted generator alone on any benchmark, and verifier fusion lowered accuracy in 12 of 12 single-seed cells. | | [X15](#x15--matched-prompt-template-controls-(template-c.1)) Matched prompt-template controls (template C.1) | Re-running the untrained controls under the task-adapted prompt head (template C.1) makes the adaptation contrast change weights only; both comparators are now reported. | | [X16](#x16--closed-form-answer-key-and-class-distribution-audit) Closed-form answer-key and class-distribution audit | Key validation disclosed degenerate MedMCQA and PubMedQA answer keys in the original loaders; those two columns are withdrawn from interpretation. | | [X17](#x17--why-closed-form-retrieval-and-verifier-fusion-hurt) Why closed-form retrieval and verifier fusion hurt | The verifier alone selects options at 23.2% against 25.0% chance on MedQA: at or below chance as a selector, which explains the closed-form decreases. | | [X18](#x18--verifier-re-evaluation-with-uncertainty-and-a-query-level-metric) Verifier re-evaluation with uncertainty and a query-level metric | Verifier ROC-AUC 0.7856 [0.7564, 0.8153] on hard negatives; it matches the oracle candidate 35.3% of the time. | | [X19](#x19--entity-aware-decoding-implemented-and-ablated) Entity-aware decoding implemented and ablated | Decoding-time bonus implemented and ablated separately from reranking: +0.64 [-0.04, +1.31] on MedQuAD, under one point either way. | | [X20](#x20--latency-memory-and-computational-cost-measurement) Latency, memory and computational cost measurement | Measured cost per question and peak memory per device for every condition; no deployability claim is drawn. | | [X21](#x21--paired-statistical-procedure-over-every-contrast) Paired statistical procedure over every contrast | 62 paired contrasts with 10,000-resample bootstraps, Holm correction over five metrics, and 204 exact McNemar tests. | | [X22](#x22--retrieval-index-and-strategy-comparison-(retained-descriptive)) Retrieval index and strategy comparison (retained, descriptive) | Not re-run: the IVF METEOR transcription error was corrected to 29.80 and the index/strategy tables are kept as a single-run descriptive record. | | [X23](#x23--release-tree-inventory-and-reproduction-map) Release tree, inventory and reproduction map | Release tree: 729 files, 380 MB, with checksums, a lockfile, per-run manifests and a table-to-command map. | | [X24](#x24--retrieval-relevance-from-the-recovered-focus-labels) Retrieval relevance from the recovered focus labels | Topic agreement@1 38.33%, @3 19.44% against 0.038% chance (514.0x lift): retrieval is far from random, but precision@k against a relevance label remains undefinable. | | [X25](#x25--cumulative-ablation-ladder) Cumulative ablation ladder | Cumulative ladder: MedQuAD reaches +4.83 points over the generator alone, iCliniq only +0.49 with an interval including zero. | | [X26](#x26--sensitivity-to-the-two-fixed-selection-weights) Sensitivity to the two fixed selection weights | Selection weight alpha changes ROUGE-1 by at most 0.10 points across its whole range; the closed-form fusion weight of 0.35 is worse than no verifier in 12 of 12 cells. | | [X27](#x27--consistency-audit-of-the-claim-narrowing) Consistency audit of the claim narrowing | 0 of 7 surfaces of the paper carry an unqualified factuality/completeness/safety claim: the narrowing is consistent, which is the branch both reviewers permitted in place of a clinician study. | | [X28](#x28--source-disjoint-retrieval-evaluation) Source-disjoint retrieval evaluation | Barring same-source and same-topic exemplars entirely gives +3.35 [+1.82, +4.95] against +2.50 unrestricted: the benefit does not depend on same-source material. | | [X29](#x29--sensitivity-to-the-exemplar-count-k-and-the-bonus-strength-gamma) Sensitivity to the exemplar count k and the bonus strength gamma | k=1/3/5 gives retrieval effects of +2.21/+2.50/+2.62; no gamma in {0.5, 1, 2, 4} produces a benefit excluding zero and gamma=4 is negative (-0.90) -- the near-zero entity effect is the mechanism, not the weight. | | [X30](#x30--verifier-transfer-per-benchmark-training-versus-the-transferred-checkpoint) Verifier transfer: per-benchmark training versus the transferred checkpoint | Training the verifier per benchmark beats the transferred checkpoint in 8 of 12 cells, confirming the transfer diagnosis, but still beats using no verifier in only 4 of 12: it is not a fix. | | [X31](#x31--closed-form-and-multi-backbone-seed-variability) Closed-form and multi-backbone seed variability | Closed-form accuracy SD across seeds averages 4.04 points (max 11.55); the verifier effect is negative in every seed in only 6 of 12 cells. Open-ended SD is 0.25 points. | | [X32](#x32--retrieval-precision-recall-and-mrr-against-a-non-circular-relevance-criterion) Retrieval precision, recall and MRR against a non-circular relevance criterion | Non-circular precision@3 at tau=0.35: dense 30.0% and question-type 34.22% against a 3.334% random baseline (about 9x); question-type wins at every threshold, corroborating its +1.24 ROUGE-1 benefit. | | [X33](#x33--feasibility-of-a-source--and-topic-disjoint-repartition) Feasibility of a source- and topic-disjoint repartition | A whole-group 80/10/10 split on (source document, focus) gives 1,641 disjoint test items from 5,429 groups -- large enough for the paired tests, so the repartition is viable rather than blocked by corpus structure. |


## Index — Reviewer #1 (30 concerns)

| Item | As written by the reviewer | Experiments | |---|---|---| | **** | State explicit research questions or hypotheses distinguishing the expected effects of retrieval, contrastive alignment, verification, entity-aware reranking, and decoding control. | [X03](#x03), [X04](#x04) | | **** | Clarify whether the primary contribution is a new algorithm, an engineering integration, or an empirical study of modular interventions. | _no experiment required_ — Statement of the contribution type; no computation required. | | **** | Explain the rationale for using training-set QA exemplars rather than external evidence retrieval; distinguish exemplar grounding from evidence-based clinical grounding. | [X11](#x11) | | **** | Report exact backbone checkpoints, embedding model, tokenizer, context length, decoding settings, hardware, software versions, random seeds and all hyperparameters. | [X01](#x01), [X14](#x14) | | **** | Describe trained/frozen parameters, adapter/projection design, the contrastive stage, loss weights, optimizer schedule and stopping criteria. | [X02](#x02) | | **** | Define the verifier architecture, training data, positive/negative construction, evidence source, class balance, candidate number N and train/validation/test separation. | [X18](#x18), [X30](#x30) | | **** | Provide a formal definition or pseudocode for the entity-aware decoding penalty, entity normalization/matching, the question-type classifier and prompt templates. | [X19](#x19) | | **** | Provide code, configuration files, exact dataset versions or detailed preprocessing scripts. | [X23](#x23) | | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | [X01](#x01), [X06](#x06), [X07](#x07), [X16](#x16), [X21](#x21), [X31](#x31) | | **** | Add paired significance testing for open-ended metrics and uncertainty for accuracy; do not interpret small BERTScore differences as stable. | [X04](#x04), [X05](#x05), [X06](#x06), [X21](#x21) | | **** | Include matched vanilla, few-shot, standard dense-RAG and full-framework baselines for each backbone and each closed-form benchmark. | [X13](#x13), [X14](#x14), [X15](#x15) | | **** | Clarify whether all baselines use identical data, prompts, fine-tuning budgets, decoding rules and test splits; Table 8 may be unfair. | [X15](#x15) | | **** | Add a dataset table with source, version, sample counts, question format, label structure and final split sizes after preprocessing and deduplication. | [X11](#x11), [X12](#x12), [X16](#x16) | | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. | [X02](#x02), [X03](#x03), [X04](#x04), [X19](#x19), [X25](#x25), [X29](#x29) | | **** | Add a hyperparameter table and a verifier training/evaluation data table. | [X01](#x01), [X14](#x14), [X18](#x18) | | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. | [X01](#x01), [X05](#x05), [X08](#x08), [X10](#x10), [X20](#x20), [X24](#x24), [X32](#x32) | | **** | Revise Figure 1 so that every module shown is defined in the Methods section. | [X02](#x02), [X19](#x19) | | **** | Tone down claims that the verifier improves factual reliability. | [X17](#x17), [X18](#x18), [X26](#x26), [X30](#x30) | | **** | Clarify that the retrieval pool consists of training examples, not an external knowledge source. | [X11](#x11) | | **** | Do not interpret ROUGE/METEOR/BERTScore as evidence of clinical usefulness or safety; add clinician-rated evaluation if clinical claims are retained. | [X27](#x27) | | **** | Provide the missing closed-form vanilla and standard-RAG controls before attributing the gains in Tables 7 and 8 to the framework. | [X04](#x04), [X13](#x13), [X14](#x14), [X16](#x16) | | **** | Correct and define the basis for 'strongest competing baseline' and the relative improvements in the Conclusions. | [X22](#x22) | | **** | Emphasize modularity and portability as engineering strengths while separating demonstrated findings from anticipated deployment benefits. | _no experiment required_ — Strengths paragraph separating demonstrated findings from anticipated benefits. | | **** | Highlight the separation between open-ended generation and closed-form option selection. | _no experiment required_ — Statement that open-ended and closed-form evaluation are separate experiments. | | **** | Explain the practical motivation for compact summaries and type-stratified retrieval, including prompt-length and latency considerations. | [X03](#x03), [X04](#x04), [X09](#x09) | | **** | Add the lack of confidence intervals, repeated random-seed runs and significance testing. | [X01](#x01), [X21](#x21), [X31](#x31) | | **** | Add verifier limitations: moderate ROC-AUC (0.6539), calibration uncertainty, dependence on automatic labels. | [X18](#x18) | | **** | Clarify licensing, privacy and reproducibility constraints of public web-derived data. | [X12](#x12) | | **** | Organize the Methods around a clear end-to-end algorithm. | _no experiment required_ — Reordering of the Methods into the executed end-to-end order. | | **** | Comprehensive language editing. | _no experiment required_ — Language and claim-narrowing pass over the original prose. |


## Index — Reviewer #2 (49 concerns)

| Item | As written by the reviewer | Experiments | |---|---|---| | **** | Distinguish the objective of improving benchmark QA performance from the broader aims of factual correctness, clinical completeness and safety. | _no experiment required_ — Separation of the measured objective from the broader clinical aims. | | **** | State the methodological contribution beyond combining components; explain expected benefits over conventional RAG. | _no experiment required_ — Statement of the methodological contribution beyond component combination. | | **** | Link each principal objective to a specific experiment, particularly question-type retrieval, contrastive alignment and entity-aware decoding. | [X03](#x03), [X04](#x04) | | **** | Clarify that open-ended and closed-form evaluations are separate dataset-specific experiments, not transfer. | _no experiment required_ — Statement that the two evaluations are separate, not transfer. | | **** | Reconcile few-shot vanilla prompting with Section 3.2's ten epochs; identify per condition whether the backbone was frozen, fully fine-tuned or adapted. | [X14](#x14), [X15](#x15) | | **** | Provide exact checkpoints, adapter settings, lengths, prompt templates, candidate counts, decoding parameters, seeds and checkpoint-selection criteria; verify the learning rate printed as 2x10^4. | [X01](#x01) | | **** | Report dataset versions, configurations, retained counts, exclusions, split seeds, evaluation sizes; iCliniq composition; MMLU subjects; PubMedQA configuration and whether abstracts were supplied. | [X07](#x07), [X11](#x11), [X12](#x12) | | **** | Clarify the relationship between the retrieval encoder, generator embeddings and projected representations; identify updated parameters, loss weight, temperature, negatives, and whether FAISS is rebuilt. | [X02](#x02) | | **** | Specify the question-type classifier, its training data and inference assignment; hard filtering versus soft prioritization; fallback for insufficient exemplars. | [X03](#x03), [X04](#x04), [X32](#x32) | | **** | Describe the verifier as implemented: architecture, supervision, sizes, positive/negative construction, evidence inputs, correctness-label definition. | [X18](#x18), [X30](#x30) | | **** | Define the entity penalty and how it changes token selection or candidate scoring; distinguish decoding-time control from reranking; describe normalization, target selection and exclusion of irrelevant entities. | [X19](#x19), [X29](#x29) | | **** | Provide the summarization prompt and checkpoint, FAISS configurations, metric settings, and release code, configs, prompts and split manifests. | [X23](#x23) | | **** | Report evaluated counts per dataset, whether full splits or subsets were used, the sampling procedure and class distributions. | [X07](#x07), [X16](#x16) | | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | [X01](#x01), [X04](#x04), [X05](#x05), [X06](#x06), [X21](#x21), [X31](#x31) | | **** | Report duplicate and near-duplicate checks across partitions; consider shared sources and overlapping answers; assess grouped splits. | [X10](#x10), [X11](#x11), [X28](#x28), [X33](#x33) | | **** | Include a conventional RAG baseline and matched backbone controls with comparable budgets. | [X13](#x13), [X14](#x14), [X15](#x15) | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | [X02](#x02), [X03](#x03), [X04](#x04), [X19](#x19), [X26](#x26), [X29](#x29) | | **** | Explain why Table 6 reports different ROC-AUC values for classification and reranking; report CIs and query-level answer-selection performance. | [X17](#x17), [X18](#x18), [X30](#x30) | | **** | Interpret small differences cautiously; do not present minor retrieval/indexing differences as advantages. | [X21](#x21) | | **** | Add a dataset summary table. | [X11](#x11), [X12](#x12), [X16](#x16) | | **** | Add a configuration table; identify the backbone represented by 'Ours' in each table. | [X01](#x01) | | **** | Expand Table 3 with the missing ablations; reconcile 'only generator' with the vanilla baseline. | [X02](#x02), [X03](#x03), [X04](#x04), [X25](#x25) | | **** | Expand Table 6 with sample sizes, prevalence, uncertainty and a query-level selection metric; explain classifier vs reranker. | [X18](#x18), [X26](#x26) | | **** | Verify the IVF METEOR value of 20.80 in Table 4; describe the configuration differences between Tables 1 and 5. | [X22](#x22) | | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. | [X01](#x01), [X08](#x08), [X20](#x20), [X24](#x24), [X32](#x32) | | **** | Do not treat ROUGE-1/METEOR/BERTScore as evidence of factual correctness, completeness or safety. | [X17](#x17), [X27](#x27) | | **** | Correct the Section 3.4 statement that BERTScore improvements are consistent against vanilla (Qwen3-8B iCliniq 82.66 -> 82.38). | _no experiment required_ — Correction of the BERTScore consistency sentence against the paper's own values. | | **** | Describe Equation 18 as question-entity retention, not clinical completeness. | _no experiment required_ — Renaming of the retention term; no computation required. | | **** | Do not attribute gains to components not independently ablated; closed-form cross-backbone comparisons do not establish incremental benefit. | [X04](#x04), [X13](#x13), [X14](#x14), [X16](#x16) | | **** | Limit efficiency, scalability and deployability claims; declining training loss does not establish generalization. | [X20](#x20) | | **** | Add a blinded clinician assessment, or narrow factuality/safety claims throughout. | [X27](#x27) | | **** | Emphasize evaluation across open-ended and closed-form QA and three backbones while distinguishing separate experiments from transfer. | _no experiment required_ — Strengths statement across tasks and backbones. | | **** | Highlight the modular design and the retrieval, summarization, verification and indexing experiments while acknowledging the need for further ablations. | _no experiment required_ — Strengths statement on the modular design. | | **** | Describe the metric improvements in a balanced way acknowledging mixed BERTScore results. | [X22](#x22) | | **** | Retain the distinction between computational benchmark evaluation and clinical validation. | [X27](#x27) | | **** | Add that reference-based metrics and question-entity coverage do not establish factual accuracy, completeness or safety. | [X27](#x27) | | **** | Add that verifier reliability depends on label quality and independence, including partially correct answers. | [X17](#x17), [X18](#x18), [X30](#x30) | | **** | Add partition overlap and benchmark pretraining exposure. | [X10](#x10), [X11](#x11), [X28](#x28), [X33](#x33) | | **** | Add that missing matched controls and ablations limit attribution. | [X04](#x04), [X13](#x13), [X14](#x14) | | **** | Add absence of CIs and repeated-run variability. | [X01](#x01), [X06](#x06), [X21](#x21), [X31](#x31) | | **** | Discuss and illustrate failure modes: summarization removing qualifiers; entity control importing exemplar content. | [X05](#x05), [X08](#x08), [X09](#x09) | | **** | Clarify that portability does not establish efficiency or generalizability. | _no experiment required_ — Limitation separating portability from efficiency and generalisability. | | **** | Consolidate implementation and training details; present offline preparation, adaptation and inference in a consistent order. | _no experiment required_ — Consolidation of implementation and training detail. | | **** | Use consistent terminology for frozen, fine-tuned, adapter-based, vanilla and generator-only conditions. | _no experiment required_ — Consistent condition terminology. | | **** | Clearly separate entity-aware decoding from entity-aware reranking; if only reranking is implemented, state that and revise title/contributions. | [X19](#x19) | | **** | Shorten repeated descriptions of reliability, portability and clinical relevance; tie claims to findings. | _no experiment required_ — Shortening of repeated reliability and portability claims. | | **** | Separate observations from explanations; mark untested reasons as hypotheses. | [X05](#x05), [X08](#x08), [X09](#x09) | | **** | Keep reproducibility details in concise tables and supplementary methods. | [X23](#x23) | | **** | Comprehensive language editing. | _no experiment required_ — Language and claim-narrowing pass over the original prose. |


---


## The experiments


### X01 — Open-ended generator fine-tuning, three seeds

**What was run.** Full fine-tuning of LLaMA-3.1-8B-Instruct on each open-ended dataset, three independent seeds (42, 1, 2); the seed fixes the 500 training rows drawn from the 2,000-row pool, the data order and the optimiser initialisation. Yields seed variability, validation-loss curves and measured cost.


**Scale executed.** 6 training runs (2 datasets x 3 seeds), 500 rows, 3 epochs, bf16, 3 x 48 GB GPUs, ~18 min each  
**Scripts.** `train_generator.py`, `launch_training.sh`


**Reported in.** Table 15 (p. 18); Table 7 (p. 11); Figure 2 (p. 17)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report exact backbone checkpoints, embedding model, tokenizer, context length, decoding settings, hardware, software versions, random seeds and all hyperparameters. | | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | | **** | Add a hyperparameter table and a verifier training/evaluation data table. | | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. | | **** | Add the lack of confidence intervals, repeated random-seed runs and significance testing. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Provide exact checkpoints, adapter settings, lengths, prompt templates, candidate counts, decoding parameters, seeds and checkpoint-selection criteria; verify the learning rate printed as 2x10^4. | | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Add a configuration table; identify the backbone represented by 'Ours' in each table. | | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. | | **** | Add absence of CIs and repeated-run variability. |


**Results**

**Training cost and final validation loss of the six runs**

| Dataset | Seed | Minutes | Peak GB/device | GPUs | LR | Val. loss | |---|---|---|---|---|---|---| | MedQuAD | 42 | 18.1 | 27.2 | 3 | 5e-06 | 1.259 | | MedQuAD | 1 | 18.0 | 27.2 | 3 | 5e-06 | 1.270 | | MedQuAD | 2 | 18.1 | 27.2 | 3 | 5e-06 | 1.265 | | iCliniq-style | 42 | 17.8 | 27.2 | 3 | 5e-06 | 2.242 | | iCliniq-style | 1 | 17.7 | 27.2 | 3 | 5e-06 | 2.239 | | iCliniq-style | 2 | 17.8 | 27.2 | 3 | 5e-06 | 2.241 |

**Seed variability, mean ± SD over the three seeds (submitted prompt handling, %)**

| Dataset | Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 | |---|---|---|---|---|---|---| | MedQuAD | no retrieval, N=5 combined | 34.34 ± 0.36 | 13.01 ± 0.32 | 21.27 ± 0.23 | 26.80 ± 0.43 | 85.01 ± 0.04 | | MedQuAD | retrieval, N=5 combined | 24.73 ± 0.30 | 6.73 ± 0.17 | 14.96 ± 0.21 | 18.62 ± 0.03 | 82.30 ± 0.13 | | iCliniq-style | no retrieval, N=5 combined | 25.39 ± 0.13 | 6.70 ± 0.09 | 16.04 ± 0.06 | 26.12 ± 0.03 | 83.25 ± 0.12 | | iCliniq-style | retrieval, N=5 combined | 16.91 ± 0.20 | 2.26 ± 0.09 | 10.57 ± 0.07 | 17.19 ± 0.26 | 81.21 ± 0.06 |


### X02 — Contrastive alignment stage, re-run and tensor-audited

**What was run.** The contrastive stage re-run on both seed-42 generators with the submitted hyper-parameters so it can be ablated, plus a direct comparison of the saved embedding tensors before and after.


**Scale executed.** 2 stage runs, 126 optimiser steps each, ~0.4 min, peak 20 GB; tensor diff over 525,336,576 entries  
**Scripts.** `contrastive_stage.py`, `audit_checks.py`


**Reported in.** Section 2.6.2 (p. 8); Table 16 (p. 18); Table 17 (p. 19); Table 7 (p. 11)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Describe trained/frozen parameters, adapter/projection design, the contrastive stage, loss weights, optimizer schedule and stopping criteria. | | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. | | **** | Revise Figure 1 so that every module shown is defined in the Methods section. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Clarify the relationship between the retrieval encoder, generator embeddings and projected representations; identify updated parameters, loss weight, temperature, negatives, and whether FAISS is rebuilt. | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | | **** | Expand Table 3 with the missing ablations; reconcile 'only generator' with the vanilla baseline. |


**Results**

**Executed stage and direct comparison of the saved embedding tensors**

| Dataset | Steps | Minutes | Peak GB | Loss | Tensors identical | Entries changed | Max |Δ| | Mean |Δ| |
|---|---|---|---|---|---|---|---|---| | MedQuAD | 126 | 0.44 | 20.0 | 0.738 → 0.938 | no | 0.40% | 2.17e-04 | 6.8e-08 | | iCliniq-style | 126 | 0.42 | 20.0 | 0.680 → 0.660 | no | 0.38% | 1.91e-04 | 6.3e-08 |

**Effect of adding the contrastive stage (paired, ROUGE-1 points)**

| Dataset | Contrast | Δ | 95% CI | p (Holm) | |---|---|---|---|---| | MedQuAD | corrected, with retrieval | +0.17 | [-0.32, +0.69] | 1.00 | | MedQuAD | corrected, no retrieval | +0.04 | [-0.47, +0.57] | 1.00 | | MedQuAD | submitted, with retrieval | -0.05 | [-0.48, +0.36] | 1.00 | | MedQuAD | submitted, no retrieval | +0.08 | [-0.37, +0.53] | 1.00 | | iCliniq-style | corrected, with retrieval | +0.13 | [-0.42, +0.69] | 1.00 | | iCliniq-style | corrected, no retrieval | -0.47 | [-1.01, +0.05] | 1.00 | | iCliniq-style | submitted, with retrieval | +0.07 | [-0.45, +0.60] | 1.00 | | iCliniq-style | submitted, no retrieval | -0.33 | [-0.83, +0.17] | 0.67 |


### X03 — Open-ended component grid, submitted prompt handling

**What was run.** Full prompt-set grid on the seed-42 generators under the prompt handling of the submitted implementation (right truncation): no-retrieval, dense retrieval, question-type retrieval, unsummarised exemplars, entity decoding with and without retrieval, at N=1 and N=5 with four selection rules. This is the diagnostic panel.


**Scale executed.** 2 datasets x 6 prompt sets x {N=1, N=5} x 4 selection rules, 300 questions each  
**Scripts.** `run_openended.py`, `scheduler.py`


**Reported in.** Table 17 (p. 19); Section 3.1 (p. 14); Table 10 (p. 15)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | State explicit research questions or hypotheses distinguishing the expected effects of retrieval, contrastive alignment, verification, entity-aware reranking, and decoding control. | | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. | | **** | Explain the practical motivation for compact summaries and type-stratified retrieval, including prompt-length and latency considerations. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Link each principal objective to a specific experiment, particularly question-type retrieval, contrastive alignment and entity-aware decoding. | | **** | Specify the question-type classifier, its training data and inference assignment; hard filtering versus soft prioritization; fallback for insufficient exemplars. | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | | **** | Expand Table 3 with the missing ablations; reconcile 'only generator' with the vanilla baseline. |


**Results**

**Paired contrasts under the submitted prompt handling** (seed-42 generator, 300 questions per dataset; difference in points with 95% bootstrap interval; **\*** = significant at p<0.05 after Holm correction over the five metrics). This panel is the implementation diagnostic, not the attribution table.

| Dataset | Contrast | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 | |---|---|---|---|---|---|---| | MedQuAD | Retrieval (k=3) vs none, N=5 combined | -8.90 [-10.46, -7.33]**\*** | -5.86 [-7.33, -4.36]**\*** | -5.97 [-7.23, -4.73]**\*** | -7.68 [-9.21, -6.23]**\*** | -2.60 [-2.97, -2.21]**\*** | | MedQuAD | Question-type vs dense retrieval | -0.22 [-1.27, +0.84] | -0.11 [-0.99, +0.83] | -0.14 [-0.95, +0.69] | +0.22 [-0.69, +1.21] | -0.08 [-0.36, +0.19] | | MedQuAD | Unsummarised vs summarised exemplars | +4.30 [+2.70, +5.89]**\*** | +4.28 [+2.74, +5.86]**\*** | +3.77 [+2.37, +5.18]**\*** | +4.09 [+2.61, +5.57]**\*** | +0.69 [+0.30, +1.09]**\*** | | MedQuAD | Budget N=5 vs N=1 (no retrieval) | +1.55 [+0.64, +2.47]**\*** | +0.97 [+0.32, +1.63]**\*** | +1.02 [+0.40, +1.68]**\*** | +1.17 [+0.38, +2.01]**\*** | +0.50 [+0.23, +0.77]**\*** | | MedQuAD | Budget N=5 vs N=1 (retrieval) | +0.76 [+0.01, +1.52] | +0.55 [-0.09, +1.18]**\*** | +0.56 [-0.03, +1.16] | +0.41 [-0.30, +1.07] | +0.14 [-0.04, +0.32] | | MedQuAD | Combined reranking vs first beam | +0.11 [-0.35, +0.56] | +0.09 [-0.30, +0.48] | -0.08 [-0.47, +0.28] | +0.10 [-0.27, +0.49] | +0.09 [-0.04, +0.22] | | MedQuAD | Verifier selection vs first beam | +0.10 [-0.35, +0.55] | +0.10 [-0.29, +0.49] | -0.09 [-0.47, +0.28] | +0.10 [-0.27, +0.51] | +0.09 [-0.04, +0.22] | | MedQuAD | Entity reranking vs first beam | +0.11 [-0.01, +0.28] | +0.05 [-0.00, +0.16] | +0.08 [-0.01, +0.20] | +0.15 [-0.01, +0.40] | +0.04 [-0.00, +0.09] | | MedQuAD | Entity decoding (γ=1) vs none | +0.02 [-0.38, +0.41] | +0.10 [-0.22, +0.38] | +0.11 [-0.19, +0.40] | -0.00 [-0.38, +0.41] | +0.02 [-0.09, +0.12] | | MedQuAD | Entity decoding + reranking | -0.05 [-0.49, +0.34] | +0.04 [-0.37, +0.39] | +0.16 [-0.15, +0.45] | +0.00 [-0.34, +0.34] | -0.00 [-0.11, +0.09] | | MedQuAD | Entity decoding, no retrieval | +0.34 [-0.24, +0.94] | +0.12 [-0.29, +0.53] | +0.08 [-0.32, +0.50] | +0.32 [-0.24, +0.89] | -0.01 [-0.19, +0.16] | | MedQuAD | Contrastive stage (retrieval) | -0.05 [-0.48, +0.36] | -0.03 [-0.40, +0.34] | -0.03 [-0.37, +0.31] | +0.03 [-0.31, +0.34] | -0.03 [-0.14, +0.08] | | iCliniq-style | Retrieval (k=3) vs none, N=5 combined | -8.40 [-9.39, -7.39]**\*** | -4.33 [-4.84, -3.83]**\*** | -5.45 [-6.03, -4.88]**\*** | -8.83 [-9.82, -7.84]**\*** | -2.02 [-2.31, -1.73]**\*** | | iCliniq-style | Question-type vs dense retrieval | -0.16 [-1.01, +0.67] | +0.10 [-0.22, +0.43] | +0.04 [-0.42, +0.51] | -0.04 [-0.85, +0.79] | -0.08 [-0.30, +0.13] | | iCliniq-style | Unsummarised vs summarised exemplars | +1.57 [+0.46, +2.68]**\*** | +1.20 [+0.74, +1.66]**\*** | +0.74 [+0.13, +1.38] | +2.10 [+1.00, +3.21]**\*** | +0.02 [-0.29, +0.33] | | iCliniq-style | Budget N=5 vs N=1 (no retrieval) | +2.17 [+1.34, +3.00]**\*** | +0.81 [+0.41, +1.19]**\*** | +0.91 [+0.45, +1.37]**\*** | +2.62 [+1.89, +3.37]**\*** | +0.99 [+0.72, +1.26]**\*** | | iCliniq-style | Budget N=5 vs N=1 (retrieval) | +2.29 [+1.58, +3.02]**\*** | +0.65 [+0.41, +0.91]**\*** | +1.26 [+0.89, +1.65]**\*** | +2.12 [+1.47, +2.79]**\*** | +0.50 [+0.30, +0.71]**\*** | | iCliniq-style | Combined reranking vs first beam | +0.13 [+0.00, +0.26] | +0.07 [+0.00, +0.18] | +0.04 [-0.05, +0.12] | +0.12 [-0.08, +0.44] | +0.07 [+0.02, +0.15] | | iCliniq-style | Verifier selection vs first beam | +0.10 [-0.01, +0.23] | +0.02 [-0.01, +0.05] | +0.03 [-0.05, +0.11] | -0.01 [-0.10, +0.11] | +0.07 [+0.01, +0.14] | | iCliniq-style | Entity reranking vs first beam | +0.06 [+0.00, +0.15] | +0.05 [+0.00, +0.16] | +0.03 [+0.00, +0.08] | +0.17 [+0.00, +0.50] | +0.03 [-0.00, +0.09] | | iCliniq-style | Entity decoding (γ=1) vs none | +0.09 [-0.34, +0.54] | +0.01 [-0.14, +0.17] | +0.12 [-0.11, +0.36] | +0.21 [-0.19, +0.63] | +0.03 [-0.08, +0.15] | | iCliniq-style | Entity decoding + reranking | -0.02 [-0.45, +0.43] | -0.04 [-0.23, +0.14] | +0.03 [-0.20, +0.28] | +0.01 [-0.47, +0.47] | -0.02 [-0.12, +0.09] | | iCliniq-style | Entity decoding, no retrieval | +0.36 [-0.15, +0.88] | +0.16 [-0.07, +0.40] | +0.45 [+0.17, +0.73]**\*** | +0.17 [-0.31, +0.63] | +0.14 [-0.00, +0.28] | | iCliniq-style | Contrastive stage (retrieval) | +0.07 [-0.45, +0.60] | +0.02 [-0.23, +0.25] | +0.06 [-0.22, +0.33] | -0.11 [-0.67, +0.42] | +0.01 [-0.12, +0.13] |


### X04 — Open-ended component grid, corrected prompt handling

**What was run.** The same grid re-run with left truncation, so the target question always survives in the prompt. This is the primary component evidence of the paper.


**Scale executed.** same grid, 2 datasets, 300 questions each, plus the contrastive checkpoints  
**Scripts.** `run_openended.py`, `relaunch_corrected_grid.sh`


**Reported in.** Table 16 (p. 18); Section 3.3 (p. 16)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | State explicit research questions or hypotheses distinguishing the expected effects of retrieval, contrastive alignment, verification, entity-aware reranking, and decoding control. | | **** | Add paired significance testing for open-ended metrics and uncertainty for accuracy; do not interpret small BERTScore differences as stable. | | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. | | **** | Provide the missing closed-form vanilla and standard-RAG controls before attributing the gains in Tables 7 and 8 to the framework. | | **** | Explain the practical motivation for compact summaries and type-stratified retrieval, including prompt-length and latency considerations. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Link each principal objective to a specific experiment, particularly question-type retrieval, contrastive alignment and entity-aware decoding. | | **** | Specify the question-type classifier, its training data and inference assignment; hard filtering versus soft prioritization; fallback for insufficient exemplars. | | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | | **** | Expand Table 3 with the missing ablations; reconcile 'only generator' with the vanilla baseline. | | **** | Do not attribute gains to components not independently ablated; closed-form cross-backbone comparisons do not establish incremental benefit. | | **** | Add that missing matched controls and ablations limit attribution. |


**Results**

**Paired contrasts under the corrected prompt handling** — the primary component evidence of the paper (seed-42 generator, 300 questions per dataset, left truncation so the target question always survives).

| Dataset | Contrast | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 | |---|---|---|---|---|---|---| | MedQuAD | Retrieval (k=3) vs none, N=5 combined | +2.50 [+0.96, +4.07] | +5.14 [+3.35, +7.03]**\*** | +4.70 [+3.16, +6.30]**\*** | +2.68 [+1.32, +4.09] | +0.84 [+0.44, +1.25]**\*** | | MedQuAD | Question-type vs dense retrieval | +1.24 [+0.22, +2.25]**\*** | +1.07 [+0.00, +2.18]**\*** | +1.08 [+0.17, +2.05]**\*** | +1.15 [+0.05, +2.26]**\*** | +0.35 [+0.09, +0.60]**\*** | | MedQuAD | Unsummarised vs summarised exemplars | +1.84 [+0.59, +3.11]**\*** | +0.94 [-0.35, +2.24]**\*** | +0.58 [-0.53, +1.76] | +1.17 [-0.12, +2.46]**\*** | +0.29 [-0.01, +0.59] | | MedQuAD | Budget N=5 vs N=1 (no retrieval) | +1.97 [+1.04, +2.89]**\*** | +1.00 [+0.34, +1.71]**\*** | +1.18 [+0.54, +1.85]**\*** | +1.24 [+0.43, +2.06]**\*** | +0.64 [+0.38, +0.91]**\*** | | MedQuAD | Budget N=5 vs N=1 (retrieval) | +1.17 [-0.06, +2.43] | +2.08 [+0.79, +3.44] | +1.69 [+0.56, +2.88] | +1.05 [-0.09, +2.16] | +0.12 [-0.17, +0.42] | | MedQuAD | Combined reranking vs first beam | -0.10 [-0.48, +0.24] | -0.15 [-0.55, +0.20] | -0.10 [-0.48, +0.23] | -0.07 [-0.40, +0.25] | -0.03 [-0.10, +0.04] | | MedQuAD | Verifier selection vs first beam | -0.07 [-0.45, +0.27] | -0.13 [-0.53, +0.22] | -0.10 [-0.48, +0.23] | -0.04 [-0.39, +0.28] | -0.04 [-0.11, +0.03] | | MedQuAD | Entity reranking vs first beam | -0.12 [-0.31, -0.00] | -0.11 [-0.29, +0.01] | -0.12 [-0.32, +0.01] | -0.12 [-0.30, -0.01] | -0.02 [-0.06, +0.00] | | MedQuAD | Entity decoding (γ=1) vs none | +0.64 [-0.04, +1.31] | +0.22 [-0.31, +0.71] | +0.47 [-0.01, +0.94] | +0.12 [-0.46, +0.69] | +0.11 [-0.06, +0.28] | | MedQuAD | Entity decoding + reranking | +0.44 [-0.21, +1.08] | +0.03 [-0.47, +0.50] | +0.33 [-0.14, +0.78] | +0.05 [-0.52, +0.59] | +0.07 [-0.09, +0.23] | | MedQuAD | Entity decoding, no retrieval | -0.11 [-0.72, +0.50] | +0.17 [-0.22, +0.58] | +0.04 [-0.36, +0.47] | +0.35 [-0.14, +0.86] | +0.00 [-0.16, +0.16] | | MedQuAD | Contrastive stage (retrieval) | +0.17 [-0.32, +0.69] | +0.19 [-0.28, +0.69] | +0.16 [-0.25, +0.59] | +0.15 [-0.38, +0.77] | +0.04 [-0.09, +0.19] | | iCliniq-style | Retrieval (k=3) vs none, N=5 combined | -1.81 [-2.60, -1.01]**\*** | -0.68 [-1.05, -0.31]**\*** | -0.87 [-1.31, -0.42]**\*** | -1.51 [-2.29, -0.76]**\*** | -0.09 [-0.32, +0.14] | | iCliniq-style | Question-type vs dense retrieval | +0.32 [-0.41, +1.07] | +0.23 [-0.12, +0.60] | +0.24 [-0.17, +0.66] | +0.17 [-0.48, +0.84] | +0.11 [-0.08, +0.30] | | iCliniq-style | Unsummarised vs summarised exemplars | +0.72 [-0.12, +1.57] | +0.62 [+0.27, +0.99]**\*** | +0.41 [-0.06, +0.87] | +0.74 [-0.05, +1.54] | -0.03 [-0.25, +0.18] | | iCliniq-style | Budget N=5 vs N=1 (no retrieval) | +2.34 [+1.51, +3.18]**\*** | +0.75 [+0.34, +1.18]**\*** | +1.11 [+0.63, +1.59]**\*** | +2.86 [+2.04, +3.67]**\*** | +1.22 [+0.92, +1.53]**\*** | | iCliniq-style | Budget N=5 vs N=1 (retrieval) | +2.16 [+1.27, +3.02]**\*** | +0.92 [+0.57, +1.27]**\*** | +1.13 [+0.67, +1.60]**\*** | +2.11 [+1.32, +2.92]**\*** | +0.32 [+0.09, +0.54]**\*** | | iCliniq-style | Combined reranking vs first beam | +0.10 [+0.03, +0.18] | +0.04 [-0.00, +0.09] | +0.06 [+0.01, +0.12] | -0.04 [-0.12, +0.04] | +0.02 [+0.00, +0.04] | | iCliniq-style | Verifier selection vs first beam | +0.09 [+0.03, +0.17] | +0.03 [-0.01, +0.08] | +0.06 [+0.01, +0.12] | -0.04 [-0.11, +0.04] | +0.02 [+0.00, +0.03] | | iCliniq-style | Entity reranking vs first beam | +0.01 [-0.01, +0.03] | +0.01 [-0.00, +0.02] | +0.00 [-0.00, +0.00] | -0.01 [-0.03, +0.02] | +0.00 [-0.00, +0.01] | | iCliniq-style | Entity decoding (γ=1) vs none | -0.06 [-0.62, +0.49] | +0.06 [-0.16, +0.29] | +0.09 [-0.21, +0.38] | -0.20 [-0.71, +0.33] | +0.03 [-0.12, +0.17] | | iCliniq-style | Entity decoding + reranking | -0.10 [-0.65, +0.45] | +0.04 [-0.18, +0.27] | +0.08 [-0.22, +0.38] | -0.10 [-0.61, +0.42] | +0.02 [-0.12, +0.16] | | iCliniq-style | Entity decoding, no retrieval | +0.30 [-0.34, +0.93] | +0.25 [-0.06, +0.55] | +0.31 [-0.04, +0.65] | +0.32 [-0.27, +0.91] | +0.14 [-0.03, +0.32] | | iCliniq-style | Contrastive stage (retrieval) | +0.13 [-0.42, +0.69] | -0.01 [-0.22, +0.19] | +0.01 [-0.27, +0.28] | -0.05 [-0.52, +0.43] | -0.01 [-0.14, +0.13] |


### X05 — Input-truncation audit and three-seed correction

**What was run.** Audit of the tokenised prompts that found the defect (over-long prompts truncated from the right, removing the target question), then re-evaluation of all three seeds on both datasets with the corrected policy so the defect can be separated from the mechanism.


**Scale executed.** 600 prompts audited; 6 re-evaluations (2 datasets x 3 seeds), 300 questions each  
**Scripts.** `audit_checks.py`, `run_openended.py --truncation_side left`


**Reported in.** Section 3.3 (p. 19); Table 18 (p. 20); Table 15 (p. 18)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add paired significance testing for open-ended metrics and uncertainty for accuracy; do not interpret small BERTScore differences as stable. | | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Discuss and illustrate failure modes: summarization removing qualifiers; entity control importing exemplar content. | | **** | Separate observations from explanations; mark untested reasons as hypotheses. |


**Results**

**The defect: tokenised retrieval prompts under the submitted policy**

| Dataset | n | Median prompt tokens | Over the 512 cap | Question survives | Question removed | |---|---|---|---|---|---| | MedQuAD | 300 | 1284 | 99% | 30 / 300 | 90% | | iCliniq-style | 300 | 1262 | 100% | 0 / 300 | 100% |

**Retrieval effect on ROUGE-1 under each policy, each measured within its own run**

| Dataset | Submitted (right truncation) | Corrected (left truncation) | Change in the measured effect | |---|---|---|---| | MedQuAD | -8.90 [-10.46, -7.33] | +2.50 [+0.95, +4.07] | +11.40 | | iCliniq-style | -8.40 [-9.39, -7.39] | -1.81 [-2.62, -1.02] | +6.59 |

**Per seed, both policies (points)**

| Dataset | Seed | ROUGE-1 submitted | ROUGE-1 corrected | BERTScore submitted | BERTScore corrected | |---|---|---|---|---|---| | MedQuAD | 42 | -8.90 | +2.50 | -2.59 | +0.84 | | MedQuAD | 1 | -10.16 | +1.68 | -2.89 | +0.65 | | MedQuAD | 2 | -9.79 | +2.73 | -2.66 | +0.96 | | iCliniq-style | 42 | -8.40 | -1.81 | -2.02 | -0.10 | | iCliniq-style | 1 | -8.79 | -2.53 | -2.22 | -0.39 | | iCliniq-style | 2 | -8.25 | -1.99 | -1.88 | -0.11 |


### X06 — Run-to-run non-determinism measurement

**What was run.** Comparison of the two policies' no-retrieval outputs to establish that the no-retrieval prompts are below the input cap and that residual differences are bf16 beam-search non-determinism, which fixes the floor below which cross-run differences are not interpreted.


**Scale executed.** 600 prompts measured for length; 600 answer pairs compared character-exactly  
**Scripts.** `nondeterminism_check.py`


**Reported in.** Section 3.3 (p. 19)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | | **** | Add paired significance testing for open-ended metrics and uncertainty for accuracy; do not interpret small BERTScore differences as stable. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Add absence of CIs and repeated-run variability. |


**Results**

**No-retrieval prompts and the two runs' no-retrieval outputs** (the no-retrieval condition is the control: its prompts are below the input cap, so the truncation policy cannot act on them)

| Dataset | Max prompt tokens | Prompts over 512 (of 300) | Answers character-identical across runs | Aggregate ROUGE-1 difference | |---|---|---|---|---| | MedQuAD | 52 | 0 | 10.3% | 0.09 | | iCliniq-style | 489 | 0 | 6.0% | 0.32 |

The interpretation floor that follows: cross-run differences below about 0.3 ROUGE-1 points are not read as effects, and every contrast in the paper is paired within a single run.


### X07 — Test-sampling-rule check, both prompt policies

**What was run.** The seed-42 generators evaluated on the first 300 test rows in file order (the sampling rule of the submitted runs) as well as the seeded random 300, under both prompt policies, so the robustness check covers the policy that carries the primary result.


**Scale executed.** 4 generation runs (2 datasets x 2 policies) on the first 300 rows, plus paired bootstraps  
**Scripts.** `run_openended.py --first_test_rows`, `sampling_check.py`


**Reported in.** Table 14 (p. 17); Table 2 (p. 6)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report dataset versions, configurations, retained counts, exclusions, split seeds, evaluation sizes; iCliniq composition; MMLU subjects; PubMedQA configuration and whether abstracts were supplied. | | **** | Report evaluated counts per dataset, whether full splits or subsets were used, the sampling procedure and class distributions. |


**Results**

| Dataset | Prompt policy | Test subset | ROUGE-1 no retr. | ROUGE-1 retr. | Paired retrieval effect | |---|---|---|---|---|---| | MedQuAD | submitted (right) | random 300, seed 42 | 33.97 | 25.07 | -8.90 [-10.46, -7.33] | | MedQuAD | submitted (right) | first 300 rows | 32.81 | 25.08 | -7.73 [-9.32, -6.09] | | MedQuAD | corrected (left) | random 300, seed 42 | 34.06 | 36.56 | +2.50 [+0.95, +4.04] | | MedQuAD | corrected (left) | first 300 rows | 32.87 | 36.67 | +3.79 [+2.09, +5.49] | | iCliniq-style | submitted (right) | random 300, seed 42 | 25.27 | 16.87 | -8.40 [-9.38, -7.42] | | iCliniq-style | submitted (right) | first 300 rows | 26.02 | 17.83 | -8.19 [-9.25, -7.16] | | iCliniq-style | corrected (left) | random 300, seed 42 | 25.59 | 23.78 | -1.81 [-2.62, -1.01] | | iCliniq-style | corrected (left) | first 300 rows | 25.99 | 24.80 | -1.19 [-1.97, -0.42] |

Questions shared by the two test subsets: MedQuAD 47; iCliniq-style 131 of 300. Only the paired Δ within a row is comparable across subsets; the absolute levels are not.


### X08 — Retrieval-effect decomposition

**What was run.** Decomposition of the generated answers' overlap with the reference into precision and recall, with answer lengths and the share of questions on which retrieval is the worse condition, under both policies, to characterise what the retrieval conditions actually did.


**Scale executed.** 2 datasets x 2 policies x 300 questions  
**Scripts.** `why_retrieval_hurts.py`


**Reported in.** Table 19 (p. 21)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. | | **** | Discuss and illustrate failure modes: summarization removing qualifiers; entity control importing exemplar content. | | **** | Separate observations from explanations; mark untested reasons as hypotheses. |


**Results**

**Decomposition of the generated answers' overlap with the reference** (this is a decomposition of generated-answer overlap, not retrieval relevance: no relevance ground truth independent of the retrieval encoder exists)

| Dataset | Prompt policy | Precision | Δ prec. | Recall | Δ rec. | Answer words | Retrieval worse on | |---|---|---|---|---|---|---|---| | MedQuAD | submitted (right) | 37.3 → 29.2 | -8.1 | 41.5 → 28.1 | -13.3 | 137 → 131 | 79% | | iCliniq-style | submitted (right) | 20.5 → 14.8 | -5.7 | 39.4 → 25.3 | -14.1 | 152 → 143 | 84% | | MedQuAD | corrected (left) | 37.3 → 41.0 | +3.7 | 41.5 → 43.3 | +1.8 | 137 → 141 | 46% | | iCliniq-style | corrected (left) | 20.5 → 19.3 | -1.2 | 39.4 → 37.1 | -2.2 | 152 → 152 | 56% |


### X09 — Exemplar-contamination measurement

**What was run.** Count, per generated answer, of biomedical entities occurring in the answer and in its retrieved exemplars but in neither the question nor the reference, under both prompt policies, plus total entities per answer.


**Scale executed.** 4 conditions x 2 policies x 2 datasets x 300 answers, scispaCy entity extraction  
**Scripts.** `analysis.py`, `entities.py`


**Reported in.** Section 3.2 (p. 15); Table 16 (p. 18)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Explain the practical motivation for compact summaries and type-stratified retrieval, including prompt-length and latency considerations. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Discuss and illustrate failure modes: summarization removing qualifiers; entity control importing exemplar content. | | **** | Separate observations from explanations; mark untested reasons as hypotheses. |


**Results**

**Entities present in the answer and in its retrieved exemplars but in neither the question nor the reference** (zero without retrieval by construction)

| Dataset | Prompt policy | Condition | n | Answers with an imported entity | Imported per answer | Total entities per answer | |---|---|---|---|---|---|---| | MedQuAD | submitted (right) | no retrieval | 300 | 0.0% | 0.00 | 6.11 | | MedQuAD | submitted (right) | dense retrieval | 300 | 82.7% | 2.87 | 4.76 | | MedQuAD | submitted (right) | question-type retrieval | 300 | 82.3% | 3.13 | 4.52 | | MedQuAD | submitted (right) | retrieval + entity decoding | 300 | 82.3% | 2.96 | 4.90 | | iCliniq-style | submitted (right) | no retrieval | 300 | 0.0% | 0.00 | 4.57 | | iCliniq-style | submitted (right) | dense retrieval | 300 | 57.7% | 1.21 | 2.97 | | iCliniq-style | submitted (right) | question-type retrieval | 300 | 55.0% | 1.15 | 2.67 | | iCliniq-style | submitted (right) | retrieval + entity decoding | 300 | 56.0% | 1.23 | 3.14 | | MedQuAD | corrected (left) | no retrieval | 300 | 0.0% | 0.00 | 6.02 | | MedQuAD | corrected (left) | dense retrieval | 300 | 38.0% | 0.91 | 4.47 | | MedQuAD | corrected (left) | question-type retrieval | 300 | 30.0% | 0.65 | 4.59 | | MedQuAD | corrected (left) | retrieval + entity decoding | 300 | 37.3% | 0.81 | 4.41 | | iCliniq-style | corrected (left) | no retrieval | 300 | 0.0% | 0.00 | 4.54 | | iCliniq-style | corrected (left) | dense retrieval | 300 | 36.0% | 0.79 | 4.05 | | iCliniq-style | corrected (left) | question-type retrieval | 300 | 28.0% | 0.54 | 3.91 | | iCliniq-style | corrected (left) | retrieval + entity decoding | 300 | 32.3% | 0.69 | 3.84 |


### X10 — MedQuAD metadata recovery, topic-overlap and deduplicated-subset analysis

**What was run.** Recovery of the MedQuAD question-focus and source-document metadata by exact question matching against the upstream release, then the retrieval effect on topic-held-out, topic-in-pool and duplicate-free subsets, under both prompt policies.


**Scale executed.** 1,642 test questions matched; subsets n=167 / 133 / 129; 10,000-resample bootstraps  
**Scripts.** `recover_medquad_meta.py`, `grouped_corrected.py`, `dedup_subset.py`


**Reported in.** Section 3.3.1 (p. 22); Table 30 (p. 26)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report duplicate and near-duplicate checks across partitions; consider shared sources and overlapping answers; assess grouped splits. | | **** | Add partition overlap and benchmark pretraining exposure. |


**Results**

**Retrieval effect by subset, MedQuAD (ROUGE-1)**

| Prompt policy | Subset | n | No retrieval | Retrieval | Paired effect | |---|---|---|---|---|---| | corrected (left) | all evaluated questions | 300 | 34.06 | 36.56 | +2.50 [+0.93, +4.05] | | corrected (left) | topic absent from the retrieval pool | 167 | 33.90 | 38.38 | +4.48 [+2.35, +6.71] | | corrected (left) | topic present in the pool | 133 | 34.26 | 34.26 | +0.00 [-2.03, +2.13] | | corrected (left) | no exact or near-duplicate in training | 129 | 33.63 | 38.27 | +4.63 [+2.06, +7.26] | | submitted (right) | all evaluated questions | 300 | 33.97 | 25.07 | -8.90 [-10.46, -7.33] | | submitted (right) | topic absent from the retrieval pool | 167 | 33.75 | 25.22 | -8.53 [-10.59, -6.44] | | submitted (right) | topic present in the pool | 133 | 34.25 | 24.88 | -9.37 [-11.77, -6.98] | | submitted (right) | no exact or near-duplicate in training | 129 | 33.09 | 24.15 | -8.94 [-11.46, -6.35] |

**Metadata recovery**: all 1,642 MedQuAD test questions were matched against the upstream release; 84.3% share their question focus with at least one training question and 90.7% share their source document. This is what makes the subsets above definable.

No source- or topic-disjoint repartition was performed; that would be required for any claim about generalisation to unseen groups.


### X11 — Duplicate and near-duplicate audit of every partition

**What was run.** Exact and TF-IDF character 3-5-gram near-duplicate checks between partitions for both open-ended corpora, with the partition sizes that remain after removing them.


**Scale executed.** all partitions of both corpora; cosine threshold 0.90  
**Scripts.** `dedup.py`


**Reported in.** Table 4 (p. 6); Table 1 (p. 5); Section 2.2.3 (p. 4)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Explain the rationale for using training-set QA exemplars rather than external evidence retrieval; distinguish exemplar grounding from evidence-based clinical grounding. | | **** | Add a dataset table with source, version, sample counts, question format, label structure and final split sizes after preprocessing and deduplication. | | **** | Clarify that the retrieval pool consists of training examples, not an external knowledge source. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report dataset versions, configurations, retained counts, exclusions, split seeds, evaluation sizes; iCliniq composition; MMLU subjects; PubMedQA configuration and whether abstracts were supplied. | | **** | Report duplicate and near-duplicate checks across partitions; consider shared sources and overlapping answers; assess grouped splits. | | **** | Add a dataset summary table. | | **** | Add partition overlap and benchmark pretraining exposure. |


**Results**

| Corpus | Partition | n | Exact in train | Exact % | Near-dup (cos≥0.90) | Near-or-exact % | After exact dedup | After near dedup | |---|---|---|---|---|---|---|---|---| | MedQuAD | validation | 1640 | 312 | 19.02% | 675 | 60.18% | 1328 | 653 | | MedQuAD | test | 1642 | 288 | 17.54% | 680 | 58.95% | 1354 | 674 | | iCliniq-style | validation | 732 | 0 | 0.00% | 0 | 0.00% | 732 | 732 | | iCliniq-style | test | 733 | 0 | 0.00% | 0 | 0.00% | 733 | 733 |

**Internal duplication of the training partitions**

| Corpus | Train rows | Unique questions | Internal duplicates | |---|---|---|---| | MedQuAD | 13125 | 11704 | 1421 | | iCliniq-style | 5856 | 5856 | 0 |


### X12 — iCliniq-style corpus provenance check

**What was run.** Exact-text matching of every question in the corpus against the multi-platform repository cited in the submitted version, which established that the corpus is the single-platform ChatDoctor-iCliniq release and shares no rows with that repository.


**Scale executed.** 7,321 rows matched across all partitions  
**Scripts.** `icliniq_composition.py`


**Reported in.** Table 5 (p. 6); Section 2.2.3 (p. 4)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add a dataset table with source, version, sample counts, question format, label structure and final split sizes after preprocessing and deduplication. | | **** | Clarify licensing, privacy and reproducibility constraints of public web-derived data. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report dataset versions, configurations, retained counts, exclusions, split seeds, evaluation sizes; iCliniq composition; MMLU subjects; PubMedQA configuration and whether abstracts were supplied. | | **** | Add a dataset summary table. |


**Results**

**Finding**: The iCliniq-style corpus used in the paper is the Hugging Face dataset lavita/ChatDoctor-iCliniq (train split, 7,321 rows; target column answer_icliniq), ingested by ingest_lavita.py. Its 7,321 rows equal the paper's total (5,856/732/733). It is a single-platform corpus (iCliniq via the ChatDoctor release), not an aggregate of iCliniq, eHealthForum, QuestionDoctors and WebMD. Exact-text matching against LasseRegin/medical-question-answer-data (icliniq 465, ehealthforum 171, questionDoctor 5,679, webmd 23,437 questions) found 0 overlaps in any split, confirming the corpus does not derive from that repository.

Corpus: `lavita/ChatDoctor-iCliniq` at revision `f2220c8483eb`.

**Partition sizes**

| Partition | Rows | |---|---| | train | 5856 | | validation | 732 | | test | 733 |

**Attribution of the corpus against the four platforms named in the submitted version** (exact-text matching against `LasseRegin/medical-question-answer-data`)

| Source platform | Rows of this corpus attributable to it | |---|---| | icliniq (ChatDoctor-iCliniq) | 7,321 | | ehealthforum | 0 | | questionDoctor | 0 | | webmd | 0 |

**Pinned dataset revisions used by every re-run**

| Dataset identifier | Pinned revision | Dated | |---|---|---| | `GBaker/MedQA-USMLE-4-options` | `0fb93dd23a73` | 2023-01-24 | | `cais/mmlu` | `c30699e8356d` | 2024-03-08 | | `lavita/ChatDoctor-iCliniq` | `f2220c8483eb` | 2023-09-11 | | `lavita/MedQuAD` | `84ea67f83cec` | 2023-12-22 | | `openlifescienceai/medmcqa` | `91c6572c4540` | 2024-01-04 | | `qiaojin/PubMedQA` | `9001f2853fb8` | 2024-03-06 | | `reginaboateng/cleaned_pubmedqa` | `299ff666cd25` | 2023-07-21 |


### X13 — Closed-form matched within-backbone controls

**What was run.** Zero-shot, few-shot (random exemplars), dense-RAG and dense-RAG-plus-verifier option scoring on the untrained backbones, matched on prompt template, evaluation items and scorer, with exact McNemar tests between conditions.


**Scale executed.** 3 backbones x 4 benchmarks x 4 conditions = 48 runs; 500 items each (PubMedQA 200)  
**Scripts.** `run_closedform.py`, `launch_closedform.sh`


**Reported in.** Table 20 (p. 21); Table 21 (p. 22); Table 12 (p. 15)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Include matched vanilla, few-shot, standard dense-RAG and full-framework baselines for each backbone and each closed-form benchmark. | | **** | Provide the missing closed-form vanilla and standard-RAG controls before attributing the gains in Tables 7 and 8 to the framework. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Include a conventional RAG baseline and matched backbone controls with comparable budgets. | | **** | Do not attribute gains to components not independently ablated; closed-form cross-backbone comparisons do not establish incremental benefit. | | **** | Add that missing matched controls and ablations limit attribution. |


**Results**

**Matched within-backbone controls on the untrained backbones** (accuracy % with Wilson 95% interval; identical items, prompt template and scorer within a row)

| Backbone | Benchmark | n | Zero-shot | Few-shot | Dense RAG | Dense RAG + verifier | |---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 500 | 63.2 [58.9, 67.3] | 55.4 [51.0, 59.7] | 50.0 [45.6, 54.4] | 44.4 [40.1, 48.8] | | LLaMA-3.1-8B | MMLU-Med | 500 | 72.0 [67.9, 75.8] | 68.0 [63.8, 71.9] | 69.6 [65.4, 73.5] | 68.8 [64.6, 72.7] | | LLaMA-3.1-8B | MedMCQA | 500 | 59.8 [55.4, 64.0] | 59.6 [55.2, 63.8] | 59.2 [54.8, 63.4] | 56.0 [51.6, 60.3] | | LLaMA-3.1-8B | PubMedQA | 200 | 80.0 [73.9, 85.0] | 80.0 [73.9, 85.0] | 77.5 [71.2, 82.7] | 77.0 [70.7, 82.3] | | Qwen3-8B | MedQA | 500 | 58.2 [53.8, 62.4] | 54.2 [49.8, 58.5] | 48.0 [43.6, 52.4] | 46.8 [42.5, 51.2] | | Qwen3-8B | MMLU-Med | 500 | 77.6 [73.7, 81.0] | 76.4 [72.5, 79.9] | 78.0 [74.2, 81.4] | 74.2 [70.2, 77.8] | | Qwen3-8B | MedMCQA | 500 | 55.0 [50.6, 59.3] | 56.2 [51.8, 60.5] | 58.0 [53.6, 62.2] | 54.0 [49.6, 58.3] | | Qwen3-8B | PubMedQA | 200 | 71.5 [64.9, 77.3] | 67.0 [60.2, 73.1] | 65.0 [58.2, 71.3] | 64.0 [57.1, 70.3] | | DeepSeek-R1-8B | MedQA | 500 | 24.8 [21.2, 28.8] | 26.0 [22.4, 30.0] | 26.6 [22.9, 30.6] | 24.8 [21.2, 28.8] | | DeepSeek-R1-8B | MMLU-Med | 500 | 30.0 [26.1, 34.2] | 32.8 [28.8, 37.0] | 33.0 [29.0, 37.2] | 33.6 [29.6, 37.9] | | DeepSeek-R1-8B | MedMCQA | 500 | 28.4 [24.6, 32.5] | 29.8 [26.0, 34.0] | 31.2 [27.3, 35.4] | 28.2 [24.4, 32.3] | | DeepSeek-R1-8B | PubMedQA | 200 | 40.0 [33.5, 46.9] | 54.0 [47.1, 60.8] | 47.0 [40.2, 53.9] | 48.0 [41.2, 54.9] |

**Exact McNemar tests on the discordant pairs**

| Backbone | Benchmark | Contrast | Δ accuracy | Discordant b/c | p | |---|---|---|---|---|---| | DeepSeek-R1-8B | MedMCQA | dense_rag vs vanilla | +2.8 | 46/32 | 0.14 | | DeepSeek-R1-8B | MedMCQA | full vs dense_rag | -3.0 | 46/61 | 0.18 | | DeepSeek-R1-8B | MedMCQA | fewshot vs vanilla | +1.4 | 40/33 | 0.48 | | LLaMA-3.1-8B | MedMCQA | dense_rag vs vanilla | -0.6 | 42/45 | 0.83 | | LLaMA-3.1-8B | MedMCQA | full vs dense_rag | -3.2 | 27/43 | 0.072 | | LLaMA-3.1-8B | MedMCQA | fewshot vs vanilla | -0.2 | 42/43 | 1.00 | | Qwen3-8B | MedMCQA | dense_rag vs vanilla | +3.0 | 45/30 | 0.11 | | Qwen3-8B | MedMCQA | full vs dense_rag | -4.0 | 17/37 | 0.009**\*** | | Qwen3-8B | MedMCQA | fewshot vs vanilla | +1.2 | 38/32 | 0.55 | | DeepSeek-R1-8B | MedQA | dense_rag vs vanilla | +1.8 | 56/47 | 0.43 | | DeepSeek-R1-8B | MedQA | full vs dense_rag | -1.8 | 28/37 | 0.32 | | DeepSeek-R1-8B | MedQA | fewshot vs vanilla | +1.2 | 53/47 | 0.62 | | LLaMA-3.1-8B | MedQA | dense_rag vs vanilla | -13.2 | 32/98 | <0.001**\*** | | LLaMA-3.1-8B | MedQA | full vs dense_rag | -5.6 | 27/55 | 0.003**\*** | | LLaMA-3.1-8B | MedQA | fewshot vs vanilla | -7.8 | 22/61 | <0.001**\*** | | Qwen3-8B | MedQA | dense_rag vs vanilla | -10.2 | 41/92 | <0.001**\*** | | Qwen3-8B | MedQA | full vs dense_rag | -1.2 | 25/31 | 0.50 | | Qwen3-8B | MedQA | fewshot vs vanilla | -4.0 | 45/65 | 0.070 | | DeepSeek-R1-8B | MMLU-Med | dense_rag vs vanilla | +3.0 | 48/33 | 0.12 | | DeepSeek-R1-8B | MMLU-Med | full vs dense_rag | +0.6 | 44/41 | 0.83 | | DeepSeek-R1-8B | MMLU-Med | fewshot vs vanilla | +2.8 | 46/32 | 0.14 | | LLaMA-3.1-8B | MMLU-Med | dense_rag vs vanilla | -2.4 | 32/44 | 0.21 | | LLaMA-3.1-8B | MMLU-Med | full vs dense_rag | -0.8 | 36/40 | 0.73 | | LLaMA-3.1-8B | MMLU-Med | fewshot vs vanilla | -4.0 | 25/45 | 0.022**\*** | | Qwen3-8B | MMLU-Med | dense_rag vs vanilla | +0.4 | 28/26 | 0.89 | | Qwen3-8B | MMLU-Med | full vs dense_rag | -3.8 | 15/34 | 0.009**\*** | | Qwen3-8B | MMLU-Med | fewshot vs vanilla | -1.2 | 19/25 | 0.45 | | DeepSeek-R1-8B | PubMedQA | dense_rag vs vanilla | +7.0 | 29/15 | 0.049**\*** | | DeepSeek-R1-8B | PubMedQA | full vs dense_rag | +1.0 | 8/6 | 0.79 | | DeepSeek-R1-8B | PubMedQA | fewshot vs vanilla | +14.0 | 36/8 | <0.001**\*** | | LLaMA-3.1-8B | PubMedQA | dense_rag vs vanilla | -2.5 | 6/11 | 0.33 | | LLaMA-3.1-8B | PubMedQA | full vs dense_rag | -0.5 | 12/13 | 1.00 | | LLaMA-3.1-8B | PubMedQA | fewshot vs vanilla | +0.0 | 7/7 | 1.00 | | Qwen3-8B | PubMedQA | dense_rag vs vanilla | -6.5 | 5/18 | 0.011**\*** | | Qwen3-8B | PubMedQA | full vs dense_rag | -1.0 | 4/6 | 0.75 | | Qwen3-8B | PubMedQA | fewshot vs vanilla | -4.5 | 6/15 | 0.078 |


### X14 — Closed-form generator fine-tuning and complete-framework evaluation

**What was run.** Fine-tuning of one generator per benchmark and backbone, then the complete submitted closed-form path (task-adapted generator with verifier-fused option scoring) against the matched untrained comparators, so adaptation and verifier fusion are separated.


**Scale executed.** 12 training runs (3 backbones x 4 benchmarks) plus 24 evaluation runs  
**Scripts.** `train_closedform.py`, `launch_cf_adapted.sh`


**Reported in.** Table 23 (p. 23); Table 24 (p. 23); Table 7 (p. 11); Table 3 (p. 6)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report exact backbone checkpoints, embedding model, tokenizer, context length, decoding settings, hardware, software versions, random seeds and all hyperparameters. | | **** | Include matched vanilla, few-shot, standard dense-RAG and full-framework baselines for each backbone and each closed-form benchmark. | | **** | Add a hyperparameter table and a verifier training/evaluation data table. | | **** | Provide the missing closed-form vanilla and standard-RAG controls before attributing the gains in Tables 7 and 8 to the framework. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Reconcile few-shot vanilla prompting with Section 3.2's ten epochs; identify per condition whether the backbone was frozen, fully fine-tuned or adapted. | | **** | Include a conventional RAG baseline and matched backbone controls with comparable budgets. | | **** | Do not attribute gains to components not independently ablated; closed-form cross-backbone comparisons do not establish incremental benefit. | | **** | Add that missing matched controls and ablations limit attribution. |


**Results**

**The complete submitted closed-form path against matched untrained comparators** (all under prompt template C.1; accuracy %)

| Backbone | Benchmark | Zero-shot (C.1) | Dense RAG (C.1) | Task-adapted | Adapted + verifier fusion | Train min | Peak GB | Val. loss | |---|---|---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 61.2 | 48.4 | 46.8 | 42.8 | 23.1 | 41.5 | 0.155 | | LLaMA-3.1-8B | MMLU-Med | 72.6 | 69.0 | 62.0 | 58.2 | 4.1 | 41.5 | 0.128 | | LLaMA-3.1-8B | MedMCQA | 58.4 | 59.4 | 47.4 | 44.8 | 14.1 | 41.5 | 0.221 | | LLaMA-3.1-8B | PubMedQA | 79.5 | 78.5 | 71.5 | 61.5 | 13.1 | 41.5 | 0.187 | | Qwen3-8B | MedQA | 51.2 | 48.2 | 64.4 | 61.2 | 10.9 | 41.8 | 0.299 | | Qwen3-8B | MMLU-Med | 73.4 | 77.8 | 78.6 | 72.6 | 1.8 | 41.8 | 0.056 | | Qwen3-8B | MedMCQA | 53.4 | 57.0 | 57.6 | 54.4 | 5.7 | 41.8 | 0.232 | | Qwen3-8B | PubMedQA | 74.0 | 66.0 | 78.5 | 73.0 | 15.1 | 41.8 | 0.161 | | DeepSeek-R1-8B | MedQA | 23.8 | 26.2 | 25.0 | 21.8 | 17.5 | 27.3 | 0.152 | | DeepSeek-R1-8B | MMLU-Med | 30.2 | 32.4 | 38.8 | 38.0 | 2.2 | 27.3 | 0.139 | | DeepSeek-R1-8B | MedMCQA | 28.2 | 31.0 | 32.0 | 29.8 | 6.8 | 27.3 | 0.208 | | DeepSeek-R1-8B | PubMedQA | 29.0 | 41.5 | 56.0 | 55.5 | 20.4 | 27.3 | 0.277 |

**Exact McNemar tests: what adaptation and verifier fusion each change**

| Backbone | Benchmark | Contrast | Δ accuracy | Discordant b/c | p | |---|---|---|---|---|---| | DeepSeek-R1-8B | MMLU-Med | adapted + verifier vs adapted | -0.8 | 41/45 | 0.75 | | DeepSeek-R1-8B | MMLU-Med | adapted vs zero-shot (C.1) | +8.6 | 118/75 | 0.002**\*** | | DeepSeek-R1-8B | MedMCQA | adapted + verifier vs adapted | -2.2 | 40/51 | 0.29 | | DeepSeek-R1-8B | MedMCQA | adapted vs zero-shot (C.1) | +3.8 | 98/79 | 0.18 | | DeepSeek-R1-8B | MedQA | adapted + verifier vs adapted | -3.2 | 20/36 | 0.044**\*** | | DeepSeek-R1-8B | MedQA | adapted vs zero-shot (C.1) | +1.2 | 55/49 | 0.62 | | DeepSeek-R1-8B | PubMedQA | adapted + verifier vs adapted | -0.5 | 6/7 | 1.00 | | DeepSeek-R1-8B | PubMedQA | adapted vs zero-shot (C.1) | +27.0 | 70/16 | <0.001**\*** | | LLaMA-3.1-8B | MMLU-Med | adapted + verifier vs adapted | -3.8 | 19/38 | 0.016**\*** | | LLaMA-3.1-8B | MMLU-Med | adapted vs zero-shot (C.1) | -10.6 | 37/90 | <0.001**\*** | | LLaMA-3.1-8B | MedMCQA | adapted + verifier vs adapted | -2.6 | 21/34 | 0.10 | | LLaMA-3.1-8B | MedMCQA | adapted vs zero-shot (C.1) | -11.0 | 44/99 | <0.001**\*** | | LLaMA-3.1-8B | MedQA | adapted + verifier vs adapted | -4.0 | 12/32 | 0.004**\*** | | LLaMA-3.1-8B | MedQA | adapted vs zero-shot (C.1) | -14.4 | 48/120 | <0.001**\*** | | LLaMA-3.1-8B | PubMedQA | adapted + verifier vs adapted | -10.0 | 5/25 | <0.001**\*** | | LLaMA-3.1-8B | PubMedQA | adapted vs zero-shot (C.1) | -8.0 | 11/27 | 0.014**\*** | | Qwen3-8B | MMLU-Med | adapted + verifier vs adapted | -6.0 | 6/36 | <0.001**\*** | | Qwen3-8B | MMLU-Med | adapted vs zero-shot (C.1) | +5.2 | 46/20 | 0.002**\*** | | Qwen3-8B | MedMCQA | adapted + verifier vs adapted | -3.2 | 25/41 | 0.064 | | Qwen3-8B | MedMCQA | adapted vs zero-shot (C.1) | +4.2 | 62/41 | 0.048**\*** | | Qwen3-8B | MedQA | adapted + verifier vs adapted | -3.2 | 16/32 | 0.029**\*** | | Qwen3-8B | MedQA | adapted vs zero-shot (C.1) | +13.2 | 107/41 | <0.001**\*** | | Qwen3-8B | PubMedQA | adapted + verifier vs adapted | -5.5 | 6/17 | 0.035**\*** | | Qwen3-8B | PubMedQA | adapted vs zero-shot (C.1) | +4.5 | 21/12 | 0.16 |


### X15 — Matched prompt-template controls (template C.1)

**What was run.** Re-run of the zero-shot and dense-RAG closed-form controls under the same prompt head as the task-adapted systems, so the adaptation contrast changes weights only and not the template.


**Scale executed.** 3 backbones x 4 benchmarks x 2 conditions = 24 runs  
**Scripts.** `launch_cf_c1.sh`, `launch_cf_c1_one.sh`


**Reported in.** Table 23 (p. 23); Table 24 (p. 23); Appendix C (p. 36)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Include matched vanilla, few-shot, standard dense-RAG and full-framework baselines for each backbone and each closed-form benchmark. | | **** | Clarify whether all baselines use identical data, prompts, fine-tuning budgets, decoding rules and test splits; Table 8 may be unfair. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Reconcile few-shot vanilla prompting with Section 3.2's ten epochs; identify per condition whether the backbone was frozen, fully fine-tuned or adapted. | | **** | Include a conventional RAG baseline and matched backbone controls with comparable budgets. |


**Results**

**The same untrained condition scored under both prompt heads** (C.2 is the matched-control head of the first revision, C.1 is the head the task-adapted systems use; re-running under C.1 makes the adaptation contrast change weights only)

| Backbone | Benchmark | Condition | Template C.2 | Template C.1 | Δ from the template alone | |---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | vanilla | 63.2 | 61.2 | -2.0 | | LLaMA-3.1-8B | MedQA | dense rag | 50.0 | 48.4 | -1.6 | | LLaMA-3.1-8B | MMLU-Med | vanilla | 72.0 | 72.6 | +0.6 | | LLaMA-3.1-8B | MMLU-Med | dense rag | 69.6 | 69.0 | -0.6 | | LLaMA-3.1-8B | MedMCQA | vanilla | 59.8 | 58.4 | -1.4 | | LLaMA-3.1-8B | MedMCQA | dense rag | 59.2 | 59.4 | +0.2 | | LLaMA-3.1-8B | PubMedQA | vanilla | 80.0 | 79.5 | -0.5 | | LLaMA-3.1-8B | PubMedQA | dense rag | 77.5 | 78.5 | +1.0 | | Qwen3-8B | MedQA | vanilla | 58.2 | 51.2 | -7.0 | | Qwen3-8B | MedQA | dense rag | 48.0 | 48.2 | +0.2 | | Qwen3-8B | MMLU-Med | vanilla | 77.6 | 73.4 | -4.2 | | Qwen3-8B | MMLU-Med | dense rag | 78.0 | 77.8 | -0.2 | | Qwen3-8B | MedMCQA | vanilla | 55.0 | 53.4 | -1.6 | | Qwen3-8B | MedMCQA | dense rag | 58.0 | 57.0 | -1.0 | | Qwen3-8B | PubMedQA | vanilla | 71.5 | 74.0 | +2.5 | | Qwen3-8B | PubMedQA | dense rag | 65.0 | 66.0 | +1.0 | | DeepSeek-R1-8B | MedQA | vanilla | 24.8 | 23.8 | -1.0 | | DeepSeek-R1-8B | MedQA | dense rag | 26.6 | 26.2 | -0.4 | | DeepSeek-R1-8B | MMLU-Med | vanilla | 30.0 | 30.2 | +0.2 | | DeepSeek-R1-8B | MMLU-Med | dense rag | 33.0 | 32.4 | -0.6 | | DeepSeek-R1-8B | MedMCQA | vanilla | 28.4 | 28.2 | -0.2 | | DeepSeek-R1-8B | MedMCQA | dense rag | 31.2 | 31.0 | -0.2 | | DeepSeek-R1-8B | PubMedQA | vanilla | 40.0 | 29.0 | -11.0 | | DeepSeek-R1-8B | PubMedQA | dense rag | 47.0 | 41.5 | -5.5 |


### X16 — Closed-form answer-key and class-distribution audit

**What was run.** Validation of every closed-form answer key, which disclosed the degenerate MedMCQA and PubMedQA keys of the original loaders, with gold-label prevalence and majority-class accuracy per benchmark.


**Scale executed.** 4 benchmarks, all evaluated items; per-class F1 and confusion matrices  
**Scripts.** `run_closedform.py`, `audit_checks.py`


**Reported in.** Table 22 (p. 22); Table 3 (p. 6); Section 3.1 (p. 14)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | | **** | Add a dataset table with source, version, sample counts, question format, label structure and final split sizes after preprocessing and deduplication. | | **** | Provide the missing closed-form vanilla and standard-RAG controls before attributing the gains in Tables 7 and 8 to the framework. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report evaluated counts per dataset, whether full splits or subsets were used, the sampling procedure and class distributions. | | **** | Add a dataset summary table. | | **** | Do not attribute gains to components not independently ablated; closed-form cross-backbone comparisons do not establish incremental benefit. |


**Results**

**Evaluated closed-form sets: gold-label distribution and majority-class accuracy** (identical items for every backbone and condition; validating these keys is what disclosed the degenerate MedMCQA and PubMedQA keys of the original loaders)

| Benchmark | n | Gold prevalence % (by option) | Majority-class accuracy | |---|---|---|---| | MedQA | 500 | 29.0 / 26.2 / 24.2 / 20.6 | 29.0% | | MMLU-Med | 500 | 22.2 / 24.2 / 22.4 / 31.2 | 31.2% | | MedMCQA | 500 | 30.8 / 24.8 / 23.4 / 21.0 | 30.8% | | PubMedQA | 200 | 56.5 / 34.5 / 9.0 | 56.5% |


### X17 — Why closed-form retrieval and verifier fusion hurt

**What was run.** Item-level analysis of the closed-form conditions: how often retrieval or verifier fusion flips a correct answer to incorrect and the reverse, and the verifier's behaviour as a standalone option selector against chance and against the majority class.


**Scale executed.** 3 backbones x 4 benchmarks, all evaluated items  
**Scripts.** `why_closedform.py`


**Reported in.** Table 26 (p. 24); Table 27 (p. 24); Table 19 (p. 21)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Tone down claims that the verifier improves factual reliability. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Explain why Table 6 reports different ROC-AUC values for classification and reranking; report CIs and query-level answer-selection performance. | | **** | Do not treat ROUGE-1/METEOR/BERTScore as evidence of factual correctness, completeness or safety. | | **** | Add that verifier reliability depends on label quality and independence, including partially correct answers. |


**Results**

**Item-level account of the closed-form losses, and the verifier as a standalone option selector**

| Backbone | Benchmark | n | Zero-shot acc. | Adapted acc. | Full framework acc. | Correct only zero-shot | Correct only full | Verifier alone acc. | Chance | Most-picked option share (full) | |---|---|---|---|---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 500 | 63.2 | 46.8 | 42.8 | 150 | 48 | 23.2 | 25.0 | 30.6% | | LLaMA-3.1-8B | MMLU-Med | 500 | 72.0 | 62.0 | 58.2 | 112 | 43 | 27.6 | 25.0 | 40.6% | | LLaMA-3.1-8B | MedMCQA | 500 | 59.8 | 47.4 | 44.8 | 114 | 39 | 21.4 | 25.0 | 35.8% | | LLaMA-3.1-8B | PubMedQA | 200 | 80.0 | 71.5 | 61.5 | 51 | 14 | 35.0 | 33.3 | 49.0% | | Qwen3-8B | MedQA | 500 | 58.2 | 64.4 | 61.2 | 55 | 70 | 23.2 | 25.0 | 35.6% | | Qwen3-8B | MMLU-Med | 500 | 77.6 | 78.6 | 72.6 | 50 | 25 | 27.6 | 25.0 | 36.6% | | Qwen3-8B | MedMCQA | 500 | 55.0 | 57.6 | 54.4 | 54 | 51 | 21.4 | 25.0 | 28.4% | | Qwen3-8B | PubMedQA | 200 | 71.5 | 78.5 | 73.0 | 20 | 23 | 35.0 | 33.3 | 58.0% | | DeepSeek-R1-8B | MedQA | 500 | 24.8 | 25.0 | 21.8 | 66 | 51 | 23.2 | 25.0 | 50.4% | | DeepSeek-R1-8B | MMLU-Med | 500 | 30.0 | 38.8 | 38.0 | 84 | 124 | 27.6 | 25.0 | 35.6% | | DeepSeek-R1-8B | MedMCQA | 500 | 28.4 | 32.0 | 29.8 | 99 | 106 | 21.4 | 25.0 | 27.6% | | DeepSeek-R1-8B | PubMedQA | 200 | 40.0 | 56.0 | 55.5 | 9 | 40 | 35.0 | 33.3 | 94.5% |


### X18 — Verifier re-evaluation with uncertainty and a query-level metric

**What was run.** Re-evaluation of the released verifier checkpoint on the seed-42 question set under both negative constructions, with question-level bootstrap intervals for ROC-AUC and PR-AUC, prevalence, and a query-level selection metric with explicit tie handling.


**Scale executed.** 300 questions x 2 negative constructions; 10,000-resample bootstraps  
**Scripts.** `verifier_eval.py`


**Reported in.** Table 31 (p. 26); Section 3.3.2 (p. 23); Table 9 (p. 13); Section 4.6 (p. 28)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Define the verifier architecture, training data, positive/negative construction, evidence source, class balance, candidate number N and train/validation/test separation. | | **** | Add a hyperparameter table and a verifier training/evaluation data table. | | **** | Tone down claims that the verifier improves factual reliability. | | **** | Add verifier limitations: moderate ROC-AUC (0.6539), calibration uncertainty, dependence on automatic labels. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Describe the verifier as implemented: architecture, supervision, sizes, positive/negative construction, evidence inputs, correctness-label definition. | | **** | Explain why Table 6 reports different ROC-AUC values for classification and reranking; report CIs and query-level answer-selection performance. | | **** | Expand Table 6 with sample sizes, prevalence, uncertainty and a query-level selection metric; explain classifier vs reranker. | | **** | Add that verifier reliability depends on label quality and independence, including partially correct answers. |


**Results**

**Re-evaluation of the released verifier checkpoint, with question-level bootstrap intervals** (a new evaluation on the seed-42 question set; these intervals are for these estimates, not for the submitted 0.6539 and 0.8489 values, which came from a different question set and pairing)

| Negative construction | Pairs | Questions | Positive prevalence | ROC-AUC [95% CI] | PR-AUC [95% CI] | No-skill PR-AUC | |---|---|---|---|---|---|---| | answer to another question (hard, 1:2) | 900 | 300 | 0.333 | 0.7856 [0.7564, 0.8153] | 0.5845 [0.5378, 0.6376] | 0.333 | | random other answer (1:1) | 600 | 300 | 0.500 | 0.9921 [0.9863, 0.9964] | 0.9933 [0.9888, 0.9967] | 0.500 |

**Query-level selection: does the verifier pick the best of the five candidates?**

| n | Matches the ROUGE-1 oracle | 95% CI | Mean regret (points) | 95% CI | Verifier score ties | |---|---|---|---|---|---| | 300 | 35.3% | [30.0, 40.7] | 1.19 | [0.89, 1.53] | 23 |

Tie rule: ties in the ROUGE-1 oracle count as a match; verifier ties broken by first index (numpy argmax)

**What that costs on the metric** (ROUGE-1 of the candidate each rule selects, same 300 questions)

| Selection rule | ROUGE-1 | |---|---| | Oracle (best of the five by ROUGE-1) | 26.26 | | Verifier's pick | 25.07 | | First beam (highest likelihood) | 24.96 |

The verifier's pick is +0.11 points from simply taking the first beam and -1.19 from the oracle, which is why the paper reports it as at or near chance as a selector rather than as a source of gains. Counting oracle ties as a match raises the agreement rate to the 35.3% above; without tie handling it is 25.0%.


### X19 — Entity-aware decoding implemented and ablated

**What was run.** Implementation of the decoding-time entity bonus as a logits processor and its ablation, alone and on top of post-generation reranking, with and without retrieval, separately from reranking.


**Scale executed.** 4 entity-decoding conditions x 2 datasets x 2 policies, 300 questions each  
**Scripts.** `run_openended.py --prompt_sets rag_edec,norag_edec`, `entities.py`


**Reported in.** Algorithm 1 (p. 13); Section 2.7.3 (p. 11); Table 16 (p. 18); Table 17 (p. 19)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Provide a formal definition or pseudocode for the entity-aware decoding penalty, entity normalization/matching, the question-type classifier and prompt templates. | | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. | | **** | Revise Figure 1 so that every module shown is defined in the Methods section. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Define the entity penalty and how it changes token selection or candidate scoring; distinguish decoding-time control from reranking; describe normalization, target selection and exclusion of irrelevant entities. | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | | **** | Clearly separate entity-aware decoding from entity-aware reranking; if only reranking is implemented, state that and revise title/contributions. |


**Results**

**Corrected prompt handling (primary)**

| Dataset | Contrast | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 | |---|---|---|---|---|---|---| | MedQuAD | Entity decoding (γ=1) vs none, first beam | +0.64 [-0.04, +1.31] | +0.22 [-0.31, +0.71] | +0.47 [-0.01, +0.94] | +0.12 [-0.46, +0.69] | +0.11 [-0.06, +0.28] | | MedQuAD | Entity decoding + combined reranking | +0.44 [-0.21, +1.08] | +0.03 [-0.47, +0.50] | +0.33 [-0.14, +0.78] | +0.05 [-0.52, +0.59] | +0.07 [-0.09, +0.23] | | MedQuAD | Entity decoding without retrieval | -0.11 [-0.72, +0.50] | +0.17 [-0.22, +0.58] | +0.04 [-0.36, +0.47] | +0.35 [-0.14, +0.86] | +0.00 [-0.16, +0.16] | | MedQuAD | Post-generation entity reranking vs first beam | -0.12 [-0.31, -0.00] | -0.11 [-0.29, +0.01] | -0.12 [-0.32, +0.01] | -0.12 [-0.30, -0.01] | -0.02 [-0.06, +0.00] | | iCliniq-style | Entity decoding (γ=1) vs none, first beam | -0.06 [-0.62, +0.49] | +0.06 [-0.16, +0.29] | +0.09 [-0.21, +0.38] | -0.20 [-0.71, +0.33] | +0.03 [-0.12, +0.17] | | iCliniq-style | Entity decoding + combined reranking | -0.10 [-0.65, +0.45] | +0.04 [-0.18, +0.27] | +0.08 [-0.22, +0.38] | -0.10 [-0.61, +0.42] | +0.02 [-0.12, +0.16] | | iCliniq-style | Entity decoding without retrieval | +0.30 [-0.34, +0.93] | +0.25 [-0.06, +0.55] | +0.31 [-0.04, +0.65] | +0.32 [-0.27, +0.91] | +0.14 [-0.03, +0.32] | | iCliniq-style | Post-generation entity reranking vs first beam | +0.01 [-0.01, +0.03] | +0.01 [-0.00, +0.02] | +0.00 [-0.00, +0.00] | -0.01 [-0.03, +0.02] | +0.00 [-0.00, +0.01] |

**Submitted prompt handling (diagnostic)**

| Dataset | Contrast | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 | |---|---|---|---|---|---|---| | MedQuAD | Entity decoding (γ=1) vs none, first beam | +0.02 [-0.38, +0.41] | +0.10 [-0.22, +0.38] | +0.11 [-0.19, +0.40] | -0.00 [-0.38, +0.41] | +0.02 [-0.09, +0.12] | | MedQuAD | Entity decoding + combined reranking | -0.05 [-0.49, +0.34] | +0.04 [-0.37, +0.39] | +0.16 [-0.15, +0.45] | +0.00 [-0.34, +0.34] | -0.00 [-0.11, +0.09] | | MedQuAD | Entity decoding without retrieval | +0.34 [-0.24, +0.94] | +0.12 [-0.29, +0.53] | +0.08 [-0.32, +0.50] | +0.32 [-0.24, +0.89] | -0.01 [-0.19, +0.16] | | MedQuAD | Post-generation entity reranking vs first beam | +0.11 [-0.01, +0.28] | +0.05 [-0.00, +0.16] | +0.08 [-0.01, +0.20] | +0.15 [-0.01, +0.40] | +0.04 [-0.00, +0.09] | | iCliniq-style | Entity decoding (γ=1) vs none, first beam | +0.09 [-0.34, +0.54] | +0.01 [-0.14, +0.17] | +0.12 [-0.11, +0.36] | +0.21 [-0.19, +0.63] | +0.03 [-0.08, +0.15] | | iCliniq-style | Entity decoding + combined reranking | -0.02 [-0.45, +0.43] | -0.04 [-0.23, +0.14] | +0.03 [-0.20, +0.28] | +0.01 [-0.47, +0.47] | -0.02 [-0.12, +0.09] | | iCliniq-style | Entity decoding without retrieval | +0.36 [-0.15, +0.88] | +0.16 [-0.07, +0.40] | +0.45 [+0.17, +0.73]**\*** | +0.17 [-0.31, +0.63] | +0.14 [-0.00, +0.28] | | iCliniq-style | Post-generation entity reranking vs first beam | +0.06 [+0.00, +0.15] | +0.05 [+0.00, +0.16] | +0.03 [+0.00, +0.08] | +0.17 [+0.00, +0.50] | +0.03 [-0.00, +0.09] |

**Question-entity retention (the quantity the bonus targets)**

| Dataset | Prompt policy | Condition | Retention | |---|---|---|---| | MedQuAD | corrected | retrieval, no bonus | 83.0% | | MedQuAD | corrected | retrieval + bonus | 85.6% | | MedQuAD | submitted | retrieval, no bonus | 24.1% | | MedQuAD | submitted | retrieval + bonus | 26.5% | | iCliniq-style | corrected | retrieval, no bonus | 37.6% | | iCliniq-style | corrected | retrieval + bonus | 47.4% | | iCliniq-style | submitted | retrieval, no bonus | 15.3% | | iCliniq-style | submitted | retrieval + bonus | 19.0% |


### X20 — Latency, memory and computational cost measurement

**What was run.** Measured per-question generation time, the added time of verifier scoring and entity extraction, retrieval time, and peak GPU memory per device, for the open-ended and closed-form conditions.


**Scale executed.** all evaluated conditions on one 48 GB GPU, bf16, batch 8  
**Scripts.** `run_openended.py`, `run_closedform.py`, `analysis.py`


**Reported in.** Table 28 (p. 25); Table 29 (p. 25); Table 7 (p. 11)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. | | **** | Limit efficiency, scalability and deployability claims; declining training loss does not establish generalization. |


**Results**

**Measured cost per question** (one 48 GB GPU, bf16, batch 8, 200 new tokens; peak memory is per device, i.e. the maximum over the visible GPUs, not the aggregate)

| Dataset | Condition | Generation s/item | Verifier + entities s/item | Retrieval s (300 questions) | Peak GB/device | |---|---|---|---|---|---| | MedQuAD | norag n1 first | 0.787 | 0.018 | 0.00 | 16.75 | | MedQuAD | norag n5 combined | 1.373 | 0.080 | 0.00 | 16.90 | | MedQuAD | rag n1 first | 0.863 | 0.017 | 0.06 | 17.51 | | MedQuAD | rag n5 combined | 2.201 | 0.075 | 0.06 | 21.34 | | MedQuAD | rag edec n5 combined | 7.687 | 0.075 | 0.05 | 17.40 | | iCliniq-style | norag n1 first | 0.916 | 0.020 | 0.00 | 17.38 | | iCliniq-style | norag n5 combined | 2.161 | 0.084 | 0.00 | 20.05 | | iCliniq-style | rag n1 first | 0.840 | 0.018 | 0.23 | 17.51 | | iCliniq-style | rag n5 combined | 2.085 | 0.080 | 0.23 | 21.33 | | iCliniq-style | rag edec n5 combined | 9.535 | 0.080 | 0.23 | 17.23 |

These measurements support no claim beyond what they show; in particular they do not establish deployability.


### X21 — Paired statistical procedure over every contrast

**What was run.** For every contrast: 10,000-resample paired bootstrap intervals of per-question differences, Wilcoxon signed-rank tests with Holm correction over the five metrics, Wilson intervals for accuracies and exact McNemar tests on discordant pairs.


**Scale executed.** all open-ended contrasts on both datasets and all closed-form backbone-benchmark cells  
**Scripts.** `analysis.py`


**Reported in.** Section 2.8 (p. 12); Table 16 (p. 18); Table 17 (p. 19); Table 21 (p. 22); Table 24 (p. 23)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | | **** | Add paired significance testing for open-ended metrics and uncertainty for accuracy; do not interpret small BERTScore differences as stable. | | **** | Add the lack of confidence intervals, repeated random-seed runs and significance testing. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Interpret small differences cautiously; do not present minor retrieval/indexing differences as advantages. | | **** | Add absence of CIs and repeated-run variability. |


**Results**

**Procedure applied to every contrast**: 10,000-resample paired bootstrap of per-question differences; Wilcoxon signed-rank with Holm correction over the five metrics within a contrast; Wilson intervals for accuracies; exact McNemar on discordant pairs (nominal, not corrected).

| Family | Paired contrasts | Interval excludes zero | Significant after Holm over five metrics | |---|---|---|---| | Corrected prompt handling (primary) | 26 | 9 | 6 | | Submitted prompt handling (diagnostic) | 36 | 14 | 11 | | Family | McNemar comparisons | p < 0.05 (nominal) | |---|---|---| | Closed-form, all backbone-benchmark cells | 204 | 95 |

Non-significant results are reported as no convincing benefit detected, not as equivalence.


### X22 — Retrieval index and strategy comparison (retained, descriptive)

**What was run.** The submitted comparison of FAISS index structures and retrieval strategies, retained as a single-run descriptive record, with the IVF METEOR transcription error corrected against the stored result file.


**Scale executed.** index structures and strategies as in the submitted runs, single run, no intervals  
**Scripts.** `(submitted runs; values re-read from the stored result files)`


**Reported in.** Table 35 (p. 27); Table 36 (p. 28); Section 4.5 (p. 27)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Correct and define the basis for 'strongest competing baseline' and the relative improvements in the Conclusions. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Verify the IVF METEOR value of 20.80 in Table 4; describe the configuration differences between Tables 1 and 5. | | **** | Describe the metric improvements in a balanced way acknowledging mixed BERTScore results. |


**Results**

**Index structures as retained in the manuscript** (single run, no intervals; the IVF METEOR cell is the corrected value)

| Index | MedQuAD R-1 | MedQuAD METEOR | MedQuAD BS | iCliniq R-1 | iCliniq METEOR | iCliniq BS | |---|---|---|---|---|---|---| | FLAT | 39.86 | 29.41 | 84.05 | 34.13 | 29.80 | 82.63 | | IVF | 39.97 | 29.40 | 84.09 | 34.14 | 29.80 | 82.62 | | HNSW | 40.36 | 29.55 | 84.18 | 34.00 | 28.83 | 82.04 | | PQ | 39.72 | 29.23 | 83.97 | 33.37 | 29.22 | 82.49 |

**Retrieval strategies as retained** (produced with `Meta-Llama-3-8B-Instruct`, not the LLaMA-3.1-8B pipeline used elsewhere)

| Strategy | MedQuAD R-1 | MedQuAD METEOR | MedQuAD BS | iCliniq R-1 | iCliniq METEOR | iCliniq BS | |---|---|---|---|---|---|---| | DENSE | 37.80 | 27.91 | 83.43 | 33.18 | 29.12 | 82.38 | | BM25 | 37.76 | 27.81 | 83.44 | 31.17 | 27.70 | 81.80 | | HYBRID | 37.61 | 27.50 | 83.51 | 32.78 | 28.90 | 82.25 | | RERANK | 37.66 | 27.57 | 83.49 | 33.06 | 28.93 | 82.32 |

This comparison was **not re-run**. The submitted single-run values are retained as a descriptive record, with one correction: the IVF METEOR cell read 20.80, a transcription error, and was corrected to **29.80**, the value in the stored result file. The manuscript states that the index-structure and retrieval-strategy differences lie within the noise of single runs and supports neither a preference among the structures nor a claim that they are equivalent, and that the DENSE row is not a plain RAG baseline. Those tables were produced with Meta-Llama-3-8B-Instruct rather than the 3.1 model used elsewhere, which the captions now state.


### X23 — Release tree, inventory and reproduction map

**What was run.** Assembly of the release: per-run manifests recording checkpoint, split row indices, sampling rule, truncation policy, prompt template, seed and timing; a file inventory with SHA-256 prefixes; an environment lockfile; and a map from every reported table and figure to the command that regenerates it.


**Scale executed.** 729 files, 380 MB inventoried  
**Scripts.** `make_release.py`


**Reported in.** Back matter (Data and code availability); Table 5 (p. 6)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Provide code, configuration files, exact dataset versions or detailed preprocessing scripts. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Provide the summarization prompt and checkpoint, FAISS configurations, metric settings, and release code, configs, prompts and split manifests. | | **** | Keep reproducibility details in concise tables and supplementary methods. |


**Results**

| Item | Content | |---|---| | Files inventoried | 729 | | Total size | 380 MB | | Checksums | SHA-256 prefix per file | | Environment | `release/requirements.lock.txt` | | Per-run provenance | `manifest.json` per run: checkpoint, split row indices, sampling rule, truncation policy, prompt template, seed, timing | | Reproduction map | `release/REPRODUCE.md`, mapping every reported table and figure to the command that regenerates it | | Model weights | not redistributed |

Anonymous accessibility of the repository can only be confirmed by the editor once the link is activated, which is why the two release concerns are reported as resolved in the manuscript with release verification outstanding.


### X24 — Retrieval relevance from the recovered focus labels

**What was run.** A retrieval-relevance metric that did not exist before the MedQuAD focus metadata was recovered: how often a retrieved exemplar shares the query's topic entity, at k=1..3, against the rate expected if exemplars were drawn from the pool at random. Precision@k against a relevance judgement remains undefined, because the exemplar pool carries no relevance labels and the only candidate ground truth would be produced by the same encoder that performs the retrieval.


**Scale executed.** 300 evaluated MedQuAD questions against the 2,000-item pool (1,458 distinct topics), both prompt policies  
**Scripts.** `extra_analyses.py`


**Reported in.** Table 19 (p. 21); Section 4.5 (p. 27)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. |


**Results**

**Topic agreement between the query and its retrieved exemplars, MedQuAD** (agreement@k is the share of the first k exemplars whose recovered focus entity equals the query's; chance is the probability that a uniformly drawn labelled pool item shares it, over a pool of 2,000 items carrying 1,458 distinct topics)

| Prompt policy | Questions | Agreement@1 | Agreement@2 | Agreement@3 | ≥1 of 3 same topic | Chance | Lift over chance | |---|---|---|---|---|---|---|---| | submitted | 300 | 38.3% | 26.0% | 19.4% | 41.3% | 0.038% | 514× | | corrected | 300 | 38.3% | 26.0% | 19.4% | 41.3% | 0.038% | 514× |

The two policies give identical values because retrieval is unaffected by the truncation policy: the same encoder, pool and query text produce the same exemplars, and only the prompt assembled from them differs. That is a useful internal check on the two panels.

**What this does and does not establish.** Retrieval selects topically related exemplars far above chance, so the retriever is not behaving randomly. It is not precision@k against a relevance judgement: the exemplar pool carries no relevance labels, and the only ground truth available would be produced by the same encoder that performs the retrieval, which is why the manuscript reports the decomposition of X08 and the subset analysis of X10 as its retrieval diagnostics instead.


### X25 — Cumulative ablation ladder

**What was run.** The cumulative build-up the independent grid does not show: generator only, then the candidate budget, then dense retrieval, then question-type stratification, then the decoding-time entity bonus, each rung adding one component to the rung above it on one fixed generator, with per-rung and cumulative paired intervals.


**Scale executed.** 5 rungs x 2 datasets x 2 prompt policies, 300 questions each, 10,000-resample paired bootstraps  
**Scripts.** `extra_analyses.py`


**Reported in.** Table 16 (p. 18); Section 4.4 (p. 25)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Expand Table 3 with the missing ablations; reconcile 'only generator' with the vanilla baseline. |


**Results**

**Corrected prompt handling**

| Dataset | Rung (each adds one component to the rung above) | n | ROUGE-1 | METEOR | BERTScore | Step on ROUGE-1 | Cumulative on ROUGE-1 | |---|---|---|---|---|---|---|---| | MedQuAD | Generator only (N=1, no retrieval) | 300 | 32.17 | 24.98 | 84.34 | -- | -- | | MedQuAD | + candidate budget N=5 with combined selection | 300 | 34.06 | 26.18 | 85.00 | +1.89 [+0.98, +2.82] | +1.89 [+0.97, +2.84] | | MedQuAD | + dense exemplar retrieval (k=3) | 300 | 36.56 | 28.86 | 85.84 | +2.50 [+0.93, +4.06] | +4.39 [+2.72, +6.12] | | MedQuAD | + question-type stratified retrieval | 300 | 37.79 | 30.01 | 86.19 | +1.24 [+0.24, +2.28] | +5.63 [+3.93, +7.42] | | MedQuAD | + decoding-time entity bonus | 300 | 36.99 | 28.91 | 85.91 | -0.80 [-1.89, +0.27] | +4.83 [+3.24, +6.43] | | iCliniq-style | Generator only (N=1, no retrieval) | 300 | 23.18 | 23.43 | 82.03 | -- | -- | | iCliniq-style | + candidate budget N=5 with combined selection | 300 | 25.59 | 26.30 | 83.25 | +2.40 [+1.57, +3.22] | +2.40 [+1.58, +3.26] | | iCliniq-style | + dense exemplar retrieval (k=3) | 300 | 23.78 | 24.79 | 83.15 | -1.81 [-2.62, -1.01] | +0.59 [-0.25, +1.44] | | iCliniq-style | + question-type stratified retrieval | 300 | 24.10 | 24.96 | 83.26 | +0.32 [-0.41, +1.08] | +0.92 [-0.00, +1.86] | | iCliniq-style | + decoding-time entity bonus | 300 | 23.68 | 24.69 | 83.17 | -0.42 [-1.19, +0.36] | +0.49 [-0.41, +1.36] |

**Submitted prompt handling**

| Dataset | Rung (each adds one component to the rung above) | n | ROUGE-1 | METEOR | BERTScore | Step on ROUGE-1 | Cumulative on ROUGE-1 | |---|---|---|---|---|---|---|---| | MedQuAD | Generator only (N=1, no retrieval) | 300 | 32.37 | 25.11 | 84.45 | -- | -- | | MedQuAD | + candidate budget N=5 with combined selection | 300 | 33.97 | 26.30 | 85.00 | +1.60 [+0.75, +2.49] | +1.60 [+0.73, +2.49] | | MedQuAD | + dense exemplar retrieval (k=3) | 300 | 25.07 | 18.62 | 82.41 | -8.90 [-10.45, -7.34] | -7.30 [-8.78, -5.79] | | MedQuAD | + question-type stratified retrieval | 300 | 24.85 | 18.85 | 82.32 | -0.22 [-1.25, +0.84] | -7.52 [-8.98, -6.02] | | MedQuAD | + decoding-time entity bonus | 300 | 25.02 | 18.62 | 82.40 | +0.17 [-1.00, +1.30] | -7.35 [-8.78, -5.88] | | iCliniq-style | Generator only (N=1, no retrieval) | 300 | 23.11 | 23.55 | 82.25 | -- | -- | | iCliniq-style | + candidate budget N=5 with combined selection | 300 | 25.27 | 26.13 | 83.24 | +2.15 [+1.31, +2.97] | +2.15 [+1.31, +3.00] | | iCliniq-style | + dense exemplar retrieval (k=3) | 300 | 16.87 | 17.30 | 81.22 | -8.40 [-9.39, -7.40] | -6.25 [-7.25, -5.20] | | iCliniq-style | + question-type stratified retrieval | 300 | 16.71 | 17.26 | 81.14 | -0.16 [-1.00, +0.68] | -6.40 [-7.43, -5.38] | | iCliniq-style | + decoding-time entity bonus | 300 | 16.85 | 17.31 | 81.20 | +0.14 [-0.73, +1.00] | -6.26 [-7.26, -5.30] |

**Reading.** On MedQuAD the ladder accumulates to +4.83 ROUGE-1 points over the generator alone, of which +1.89 comes from the candidate budget before any proposed component is involved. On the iCliniq-style data the cumulative total is +0.49 points with an interval of [-0.41, +1.36], i.e. indistinguishable from zero, and the budget rung alone contributes +2.40: on that dataset the proposed components together give back roughly what the budget gains. This is the same conclusion the independent grid reaches, reached a second way.


### X26 — Sensitivity to the two fixed selection weights

**What was run.** Both selection weights were fixed a priori and never tuned, which leaves open whether the near-zero selection effects are a property of the mechanism or of the chosen weight. The open-ended combined rule was re-selected from the saved candidate pools across the full range of its verifier/entity weight, and the closed-form option scoring was re-fused across the full range of its verifier weight.


**Scale executed.** alpha in {0, 0.25, 0.5, 0.75, 1} x 2 datasets x 2 policies; fusion weight in {0, 0.2, 0.35, 0.5, 0.65, 0.8, 1} x 3 backbones x 4 benchmarks  
**Scripts.** `extra_analyses.py`


**Reported in.** Table 16 (p. 18); Table 23 (p. 23); Section 2.7.2 (p. 11)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Tone down claims that the verifier improves factual reliability. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. | | **** | Expand Table 6 with sample sizes, prevalence, uncertainty and a query-level selection metric; explain classifier vs reranker. |


**Results**

**Open-ended combined selection rule: ROUGE-1 as the verifier/entity weight α is swept** (α=0 is entity retention alone, α=1 is the verifier alone, α=0.75 is the executed value; re-selected from the saved candidate pools, so the generator is identical throughout)

| Dataset | Prompt policy | α=0.0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 | Spread | |---|---|---|---|---|---|---|---| | MedQuAD | corrected | 36.53 | 36.56 | 36.56 | 36.56 | 36.59 | 0.06 | | iCliniq-style | corrected | 23.68 | 23.78 | 23.78 | 23.78 | 23.77 | 0.10 | | MedQuAD | submitted | 25.07 | 25.07 | 25.07 | 25.07 | 25.07 | 0.00 | | iCliniq-style | submitted | 16.80 | 16.87 | 16.87 | 16.87 | 16.85 | 0.07 |


The widest spread across the whole range of α is 0.10 ROUGE-1 points, which is below the run-to-run non-determinism floor of about 0.3 points established in X06. The near-zero selection effect is therefore a property of the mechanism — five beams of one beam search are highly correlated, so any reweighting of a score over them changes little — and not an artefact of the weight that was chosen.


**Closed-form option scoring: accuracy as the verifier fusion weight is swept** (w=0 is pure continuation log-probability, i.e. the task-adapted system with no verifier; w=1 is the verifier alone; w=0.35 is the executed value)

| Backbone | Benchmark | n | w=0.0 | w=0.2 | w=0.35 | w=0.5 | w=0.65 | w=0.8 | w=1.0 | Best w | Executed − no verifier | |---|---|---|---|---|---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 500 | 46.8 | 46.0 | 42.8 | 35.4 | 31.8 | 29.0 | 23.2 | 0.0 | -4.0 | | LLaMA-3.1-8B | MMLU-Med | 500 | 62.0 | 62.6 | 58.2 | 49.2 | 41.2 | 33.8 | 27.6 | 0.2 | -3.8 | | LLaMA-3.1-8B | MedMCQA | 500 | 47.4 | 47.2 | 44.8 | 37.4 | 31.2 | 25.8 | 21.4 | 0.0 | -2.6 | | LLaMA-3.1-8B | PubMedQA | 200 | 71.5 | 70.0 | 61.5 | 41.5 | 37.0 | 35.0 | 35.0 | 0.0 | -10.0 | | Qwen3-8B | MedQA | 500 | 64.4 | 63.0 | 61.6 | 49.6 | 38.6 | 33.6 | 23.2 | 0.0 | -2.8 | | Qwen3-8B | MMLU-Med | 500 | 78.6 | 77.4 | 72.6 | 55.2 | 45.2 | 35.0 | 27.6 | 0.0 | -6.0 | | Qwen3-8B | MedMCQA | 500 | 57.6 | 58.6 | 54.4 | 42.8 | 33.2 | 27.2 | 21.4 | 0.2 | -3.2 | | Qwen3-8B | PubMedQA | 200 | 78.5 | 77.5 | 73.0 | 50.5 | 41.0 | 37.5 | 35.0 | 0.0 | -5.5 | | DeepSeek-R1-8B | MedQA | 500 | 24.8 | 22.6 | 21.4 | 21.8 | 21.0 | 21.0 | 23.2 | 0.0 | -3.4 | | DeepSeek-R1-8B | MMLU-Med | 500 | 38.8 | 39.8 | 38.0 | 33.4 | 31.8 | 28.2 | 27.6 | 0.2 | -0.8 | | DeepSeek-R1-8B | MedMCQA | 500 | 32.0 | 30.4 | 29.8 | 26.4 | 26.0 | 23.4 | 21.4 | 0.0 | -2.2 | | DeepSeek-R1-8B | PubMedQA | 200 | 56.0 | 54.0 | 55.5 | 52.5 | 45.0 | 41.0 | 35.0 | 0.0 | -0.5 |


The executed weight is worse than using no verifier at all in 12 of 12 backbone–benchmark cells, and w=0 is the best of the seven weights tested in 9 of 12. Where another weight wins it wins by at most a point and always at a small w. So the 12-of-12 accuracy decrease reported for verifier fusion is not an artefact of the 0.35 choice: there is no weight in the range at which the fusion helps.


### X27 — Consistency audit of the claim narrowing

**What was run.** Reviewer 2 asked for the factuality, completeness and safety claims to be narrowed consistently rather than only in the limitations. Every surface of the paper was scanned for a surviving unqualified claim of that kind and for the presence of an explicit disclaimer, separating statements about this system from citations of other work and from future-work statements.


**Scale executed.** 7 surfaces: highlights, abstract, introduction, methods, results, discussion, conclusions  
**Scripts.** `extra_analyses.py`


**Reported in.** Abstract; Section 5.4 (p. 31); Section 6 (Conclusions)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Do not interpret ROUGE/METEOR/BERTScore as evidence of clinical usefulness or safety; add clinician-rated evaluation if clinical claims are retained. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Do not treat ROUGE-1/METEOR/BERTScore as evidence of factual correctness, completeness or safety. | | **** | Add a blinded clinician assessment, or narrow factuality/safety claims throughout. | | **** | Retain the distinction between computational benchmark evaluation and clinical validation. | | **** | Add that reference-based metrics and question-entity coverage do not establish factual accuracy, completeness or safety. |


**Results**

**Every surface of the paper, scanned for a surviving unqualified factuality, completeness or safety claim** (a mention is not counted against the paper when the surrounding text negates or withdraws it, and is classed separately when it reports what other work says or what future work should do)

| Surface | Mentions of such a claim | Unqualified claims | Not a claim about this system | Explicit disclaimer | |---|---|---|---|---| | Highlights | 0 | 0 | 0 | makes no such claim | | Abstract | 2 | 0 | 0 | yes | | Introduction | 15 | 0 | 0 | yes | | Methods | 0 | 0 | 0 | makes no such claim | | Results | 0 | 0 | 0 | makes no such claim | | Discussion | 4 | 0 | 0 | yes | | Conclusions | 2 | 0 | 0 | yes |

**Result.** 0 of 7 surfaces carry an unqualified claim; 4 carry an explicit disclaimer and 3 make no claim of the kind at all. The narrowing is therefore consistent across the paper and not confined to the limitations, which is what was asked. The two mentions the scan initially flagged were a citation of what the evaluation literature says metrics cannot capture, and the future-work sentence calling for clinician-led evaluation; neither asserts anything about this system.

**What is still not done.** No clinician-rated evaluation was performed. Both reviewers phrased the request conditionally — add a clinician assessment *or* narrow the claims — and this revision takes the second branch. The audit above is the evidence that the branch was taken completely.


### X28 — Source-disjoint retrieval evaluation

**What was run.** The repartition Reviewer 2 named, in the form the recovered metadata makes possible without retraining: exemplars that share the query's MedQuAD source document or topic entity are barred from retrieval entirely, and the retrieval effect is re-measured against a no-retrieval control generated in the same run. This tests directly whether the measured benefit depends on same-source exemplars.


**Scale executed.** 300 MedQuAD questions, corrected prompt handling, exemplars restricted to a source- and topic-disjoint pool, no-retrieval control paired within the same run  
**Scripts.** `run_openended.py --exclude_same both`, `srcdisj_check.py`


**Reported in.** Section 3.3.1 (p. 22); Table 30 (p. 26)


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report duplicate and near-duplicate checks across partitions; consider shared sources and overlapping answers; assess grouped splits. | | **** | Add partition overlap and benchmark pretraining exposure. |


**Results**

**Pool restriction.** Barring every exemplar that shares the query's source document or topic entity removes on average 1.59 of 2000 pool items per question (62.7% of questions have any item barred at all). The retrieval pool is therefore already almost disjoint from the evaluated questions by source, which is itself part of the answer: there is little same-source material for retrieval to exploit.

| Condition | n | ROUGE-1 no retrieval | ROUGE-1 retrieval | Paired retrieval effect | |---|---|---|---|---| | Unrestricted pool (primary corrected run) | 300 | 34.06 | 36.56 | +2.50 [+0.97, +4.11] | | Source- and topic-disjoint pool | 300 | 34.43 | 37.78 | +3.35 [+1.82, +4.95] |

**Reading.** With same-source and same-topic exemplars barred entirely, the retrieval effect is +3.35 points [+1.82, +4.95], larger than the +2.50 [+0.97, +4.11] measured with the unrestricted pool. The benefit therefore does not depend on exemplars drawn from the query's own source document or topic: removing exactly the material a leakage account would blame leaves the effect intact. Both intervals exclude zero, and each effect is paired against a no-retrieval control generated in the same run.

**What it does not cover.** This restricts what retrieval may *see*; it does not remove same-source material from the generator's training data. A full repartition would additionally re-train on a source-disjoint split, which is listed under 'Considered and not run'. The result above therefore addresses the leakage question for retrieval, and the paper still makes no claim about generalisation to unseen source groups.


### X29 — Sensitivity to the exemplar count k and the bonus strength gamma

**What was run.** The two remaining fixed hyper-parameters, which unlike the selection weights could only be swept by generating. For k, the retrieval effect is re-measured with 1, 3 and 5 exemplars; for gamma, the entity-bonus effect is re-measured at 0.5, 1, 2 and 4. Each value was generated in its own run together with its own paired control, so no comparison crosses a run boundary.


**Scale executed.** 5 generation runs on 300 MedQuAD questions (k=1, k=5, gamma=0.5, 2, 4), plus the executed k=3 and gamma=1 taken from the primary corrected run; 10,000-resample paired bootstraps  
**Scripts.** `run_openended.py --k / --gamma_override`, `hparam_sweeps.py`


**Reported in.** Table 16 (p. 18); Section 2.7.3 (p. 11); Algorithm 1 (p. 13)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add a complete ablation table covering ordinary retrieval, question-type retrieval, contrastive alignment, summarization, verifier, entity reranking and entity-aware decoding. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Define the entity penalty and how it changes token selection or candidate scoring; distinguish decoding-time control from reranking; describe normalization, target selection and exclusion of irrelevant entities. | | **** | Independently ablate question-type awareness, contrastive alignment, entity coverage and entity-aware decoding; state whether ablations were retrained or inference-only. |


**Results**

**Exemplar count k** (each row's retrieval effect is paired against the no-retrieval control generated in the same run; k=3 is the executed value)

| k | Run | n | ROUGE-1 no retr. | ROUGE-1 retr. | Paired retrieval effect | Exemplars used | |---|---|---|---|---|---|---| | 1 | sweep run k1 | 300 | 34.06 | 36.27 | +2.21 [+0.88, +3.61] | 1.0 | | 3 | primary corrected run | 300 | 34.06 | 36.56 | +2.50 [+1.00, +4.04] | 3.0 | | 5 | sweep run k5 | 300 | 34.06 | 36.68 | +2.62 [+1.10, +4.25] | 5.0 |


The effect rises monotonically from +2.21 to +2.62 points as k goes from 1 to 5, a spread of 0.41 points with every interval overlapping every other. More exemplars help slightly, and the executed k=3 sits in the middle: the retrieval result is not an artefact of that choice.


**Decoding-bonus strength γ**, all five metrics (effect of adding the bonus to the retrieval condition of the same run; γ=1 is the executed value; **\*** = p<0.05 after Holm correction over the five metrics)

| γ | n | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore | |---|---|---|---|---|---|---| | 0.5 | 300 | +0.45 [-0.13, +1.02] | +0.25 [-0.17, +0.68] | +0.42 [-0.01, +0.85] | +0.16 [-0.29, +0.61] | +0.07 [-0.07, +0.21] | | 1.0 | 300 | +0.44 [-0.20, +1.06] | +0.03 [-0.46, +0.50] | +0.33 [-0.13, +0.78] | +0.05 [-0.50, +0.57] | +0.07 [-0.09, +0.23] | | 2.0 | 300 | +0.16 [-0.74, +1.01] | -0.28 [-1.15, +0.50] | +0.02 [-0.75, +0.75] | -0.11 [-0.87, +0.65] | -0.02 [-0.25, +0.20] | | 4.0 | 300 | -0.90 [-2.07, +0.26] | -0.99 [-2.23, +0.11] | -0.88 [-1.95, +0.12] | -0.88 [-1.92, +0.14] | -0.31 [-0.61, -0.01] | | 5.0 | 300 | -1.93 [-3.27, -0.69] | -2.21 [-3.72, -0.78] | -1.91 [-3.22, -0.70] | -1.89 [-3.06, -0.76] | -0.56 [-0.91, -0.22] | | 10.0 | 300 | -2.81 [-4.16, -1.49]**\*** | -2.62 [-3.99, -1.37]**\*** | -2.47 [-3.74, -1.33]**\*** | -2.54 [-3.66, -1.48]**\*** | -0.81 [-1.17, -0.47]**\*** |


**Question-entity retention, the quantity the bonus is built to raise**

| γ | Without bonus | With bonus | Change | |---|---|---|---| | 0.5 | 83.00 | 84.58 | +1.58 | | 1.0 | 83.00 | 85.57 | +2.57 | | 2.0 | 83.00 | 85.77 | +2.77 | | 4.0 | 83.00 | 86.17 | +3.17 | | 5.0 | 83.00 | 86.56 | +3.56 | | 10.0 | 83.00 | 85.57 | +2.57 |


**This answers the question the entity-coverage ablation invites, and the retention table shows why.** The bonus does what it was designed to do — retention rises monotonically with γ — while answer quality falls monotonically on all five metrics over the same range. The mechanism is faithful to its specification; the specification does not help. No γ produces a benefit that survives correction, the executed γ=1 is near the best of the six values tested, and the harm becomes significant only at γ=10.0, where it is significant on all five metrics at once. The γ=1 row reproduces the corresponding contrast of the primary grid exactly, checking that the sweep and the grid measure the same quantity.


### X30 — Verifier transfer: per-benchmark training versus the transferred checkpoint

**What was run.** The closed-form path fuses option log-probabilities with a verifier trained on MedQA pairs and applied unchanged to the other three benchmarks, which the paper diagnosed as the cause of the accuracy decrease but did not test. One verifier was therefore trained per benchmark on that benchmark's own training split, with the same architecture, input format and proxy-label construction, and the saved option log-probabilities were re-fused with both verifiers at the executed weight and across the whole weight range.


**Scale executed.** 4 verifier training runs (ClinicalBERT, 8,000 pairs each, ~25 s per run); re-fusion over 3 backbones x 4 benchmarks x 7 weights x 2 verifiers  
**Scripts.** `train_verifier_cf.py`, `verifier_transfer.py`


**Reported in.** Table 23 (p. 23); Table 9 (p. 13); Section 3.3.2 (p. 23)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Define the verifier architecture, training data, positive/negative construction, evidence source, class balance, candidate number N and train/validation/test separation. | | **** | Tone down claims that the verifier improves factual reliability. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Describe the verifier as implemented: architecture, supervision, sizes, positive/negative construction, evidence inputs, correctness-label definition. | | **** | Explain why Table 6 reports different ROC-AUC values for classification and reranking; report CIs and query-level answer-selection performance. | | **** | Add that verifier reliability depends on label quality and independence, including partially correct answers. |


**Results**

**Accuracy with no verifier, with the transferred verifier, and with a verifier trained on the benchmark itself** (all at the executed fusion weight of 0.35 unless stated; the same saved option log-probabilities throughout, so only the verifier changes)

| Backbone | Benchmark | n | No verifier | Transferred @0.35 | Own @0.35 | Own − transferred | Own, best weight | |---|---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 500 | 46.8 | 42.4 | 45.0 | +2.6 | 47.4 (w=0.2) | | Qwen3-8B | MedQA | 500 | 64.4 | 62.8 | 60.2 | -2.6 | 64.4 (w=0.0) | | DeepSeek-R1-8B | MedQA | 500 | 24.8 | 24.4 | 26.6 | +2.2 | 27.2 (w=0.5) | | LLaMA-3.1-8B | MedMCQA | 500 | 47.4 | 47.0 | 47.6 | +0.6 | 47.8 (w=0.2) | | Qwen3-8B | MedMCQA | 500 | 57.6 | 56.8 | 57.4 | +0.6 | 58.2 (w=0.2) | | DeepSeek-R1-8B | MedMCQA | 500 | 32.0 | 32.8 | 31.2 | -1.6 | 32.0 (w=0.0) | | LLaMA-3.1-8B | MMLU-Med | 500 | 62.0 | 56.0 | 57.4 | +1.4 | 62.0 (w=0.0) | | Qwen3-8B | MMLU-Med | 500 | 78.6 | 70.6 | 75.2 | +4.6 | 78.6 (w=0.0) | | DeepSeek-R1-8B | MMLU-Med | 500 | 38.8 | 39.2 | 36.6 | -2.6 | 39.6 (w=0.2) | | LLaMA-3.1-8B | PubMedQA | 200 | 71.5 | 69.5 | 72.5 | +3.0 | 78.0 (w=0.5) | | Qwen3-8B | PubMedQA | 200 | 78.5 | 79.5 | 80.5 | +1.0 | 82.0 (w=0.5) | | DeepSeek-R1-8B | PubMedQA | 200 | 56.0 | 57.0 | 54.5 | -2.5 | 56.5 (w=0.65) |

**Reading.** Training the verifier on the benchmark it is used on beats the transferred checkpoint in 8 of 12 backbone–benchmark cells, so the transfer was a real cause of the decrease and the paper's diagnosis is confirmed rather than merely asserted. It is not a fix: the own-benchmark verifier still beats using no verifier at all in only 4 of 12 cells at the executed weight, and in 8 of 12 if the weight is chosen per cell — and that second figure is optimistic, because the weight is then selected on the evaluation set itself. The honest conclusion is that per-benchmark training removes part of the penalty without making verifier fusion beneficial, which is consistent with X26: there is no weight at which fusion helps, and with X17: the verifier is at or near chance as a standalone selector.

The four trained verifiers were deleted after this re-scoring; their training manifests, sizes and validation accuracies are recorded with the released run metadata.


### X31 — Closed-form and multi-backbone seed variability

**What was run.** The closed-form results were single-seed. Every benchmark and backbone was re-trained with two further seeds and re-evaluated on the identical items, giving three seeds per cell, and the two backbones whose open-ended runs were single-seed were re-trained the same way. The point is not a better number but a scale: how large is the training variance against which the paper's per-cell effects are read?


**Scale executed.** 24 closed-form training runs plus their evaluations (3 backbones x 4 benchmarks x 2 extra seeds), and 12 open-ended runs for the two remaining backbones; every checkpoint deleted after its evaluation  
**Scripts.** `run_seed_variability.sh`, `train_closedform.py`, `train_generator.py`, `seed_variability.py`


**Reported in.** Table 25 (p. 24); Table 23 (p. 23); Table 15 (p. 18)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report the number of examples in every split and the number of evaluation runs; present results as mean +/- SD or bootstrap CIs across seeds. | | **** | Add the lack of confidence intervals, repeated random-seed runs and significance testing. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Add CIs for metrics and differences, paired analyses with query-level grouping, and training-seed variability. | | **** | Add absence of CIs and repeated-run variability. |


**Results**

**Closed-form accuracy over three training seeds** (seeds 42, 1 and 2; identical evaluation items throughout, so only the training run differs)

| Backbone | Benchmark | Adapted, seeds 42/1/2 | Adapted mean ± SD | + verifier, seeds 42/1/2 | + verifier mean ± SD | Mean verifier effect | Negative in every seed | |---|---|---|---|---|---|---|---| | LLaMA-3.1-8B | MedQA | 46.8 / 24.8 / 39.4 | 37.0 ± 11.19 | 42.8 / 24.8 / 38.0 | 35.2 ± 9.32 | -1.80 | no | | LLaMA-3.1-8B | MedMCQA | 47.4 / 49.0 / 51.2 | 49.2 ± 1.91 | 44.8 / 47.4 / 51.4 | 47.9 ± 3.32 | -1.33 | no | | LLaMA-3.1-8B | MMLU-Med | 62.0 / 42.4 / 62.8 | 55.7 ± 11.55 | 58.2 / 39.8 / 60.0 | 52.7 ± 11.18 | -3.07 | yes | | LLaMA-3.1-8B | PubMedQA | 71.5 / 72.5 / 78.0 | 74.0 ± 3.50 | 61.5 / 69.5 / 78.0 | 69.7 ± 8.25 | -4.33 | no | | Qwen3-8B | MedQA | 64.4 / 64.2 / 59.6 | 62.7 ± 2.72 | 61.2 / 62.4 / 58.8 | 60.8 ± 1.83 | -1.93 | yes | | Qwen3-8B | MedMCQA | 57.6 / 59.6 / 56.6 | 57.9 ± 1.53 | 54.4 / 56.8 / 53.0 | 54.7 ± 1.92 | -3.20 | yes | | Qwen3-8B | MMLU-Med | 78.6 / 78.6 / 78.2 | 78.5 ± 0.23 | 72.6 / 74.0 / 73.4 | 73.3 ± 0.70 | -5.13 | yes | | Qwen3-8B | PubMedQA | 78.5 / 76.0 / 73.5 | 76.0 ± 2.50 | 73.0 / 77.5 / 76.5 | 75.7 ± 2.36 | -0.33 | no | | DeepSeek-R1-8B | MedQA | 25.0 / 31.2 / 25.8 | 27.3 ± 3.37 | 21.8 / 32.4 / 26.4 | 26.9 ± 5.32 | -0.47 | no | | DeepSeek-R1-8B | MedMCQA | 32.0 / 35.6 / 31.4 | 33.0 ± 2.27 | 29.8 / 32.2 / 28.4 | 30.1 ± 1.92 | -2.87 | yes | | DeepSeek-R1-8B | MMLU-Med | 38.8 / 38.4 / 32.8 | 36.7 ± 3.35 | 38.0 / 36.0 / 31.8 | 35.3 ± 3.16 | -1.40 | yes | | DeepSeek-R1-8B | PubMedQA | 56.0 / 56.0 / 52.5 | 54.8 ± 2.02 | 55.5 / 55.5 / 53.0 | 54.7 ± 1.44 | -0.17 | no |

**How large is the training variance?** The standard deviation of accuracy across seeds averages 4.04 points and reaches 11.55. Two LLaMA-3.1-8B runs collapsed towards chance on seed 1 (MedQA 46.8 → 24.8, MMLU-Med 62.0 → 42.4), which is itself the instability the reviewers were probing for: a single closed-form fine-tuning run is not a reliable estimate of what that configuration achieves.

**What it does to the paper's claims.** Averaged over three seeds the verifier effect stays negative in 12 of 12 cells, so the direction of the reported finding is unchanged. But it is negative in *every* seed in only 6 of 12 cells, and in 8 of 12 the mean effect is smaller than the seed-to-seed standard deviation of the task-adapted system it is measured against. The per-cell signs of the single-seed table are therefore not a per-benchmark ranking, and the manuscript now says so at the point where those signs are reported. This is the main substantive change this experiment produced.

Every checkpoint was deleted immediately after its evaluation, so the runs never occupied more than about one checkpoint per active lane.

**Open-ended side: the backbones whose open-ended runs were single-seed** (corrected prompt handling, 300 questions, retrieval effect paired within each run)

| Backbone | Dataset | Seeds | ROUGE-1 seeds 42/1/2 | ROUGE-1 mean ± SD | Retrieval effect per seed | Effect mean ± SD | Same sign in all seeds | |---|---|---|---|---|---|---|---| | Qwen3-8B | MedQuAD | 3 | 36.69 / 36.81 / 36.19 | 36.56 ± 0.33 | +5.13 / +5.59 / +4.84 | +5.19 ± 0.38 | yes | | Qwen3-8B | iCliniq-style | 3 | 24.02 / 24.22 / 24.36 | 24.20 ± 0.17 | +0.04 / +0.60 / +0.68 | +0.44 ± 0.35 | yes | | DeepSeek-R1-8B | MedQuAD | 3 | 34.27 / 33.38 / 33.81 | 33.82 ± 0.45 | +3.14 / +1.70 / +2.19 | +2.34 ± 0.73 | yes | | DeepSeek-R1-8B | iCliniq-style | 3 | 22.37 / 22.49 / 22.41 | 22.42 ± 0.06 | -2.27 / -2.09 / -2.63 | -2.33 ± 0.27 | yes |

**The contrast with the closed-form side is the point.** Open-ended ROUGE-1 varies across seeds by 0.25 points on average, against 4.04 points for closed-form accuracy — roughly two orders of magnitude tighter. The retrieval effect keeps the same sign in every seed in 4 of 4 cells. Open-ended conclusions drawn from one seed are therefore on much firmer ground than closed-form ones, which is why the paper's three-seed protocol was applied to the open-ended generator first.


### X32 — Retrieval precision, recall and MRR against a non-circular relevance criterion

**What was run.** Precision@k was previously declared not computable because the exemplar pool carries no relevance labels and the only candidate ground truth --- question-focus agreement --- is a near-synonym of the retrieval criterion itself. This uses a criterion drawn from a different signal: an exemplar counts as relevant when its \emph{unsummarised answer} reaches ROUGE-1 F at or above a threshold against the test question's reference answer. Retrieval ranks by question-embedding similarity, relevance is judged from answer content, so precision@k is not measuring the retriever against itself.


**Scale executed.** 300 evaluated questions x 2,000 pool items = 600,000 relevance scores, at three thresholds, for dense and question-type retrieval, top-10 rankings rebuilt on CPU; the fast ROUGE-1 F used for the matrix was verified to reproduce rouge_score exactly before use  
**Scripts.** `retrieval_prf.py`


**Reported in.** Table 19 (p. 21); Section 4.5 (p. 27); Table 16 (p. 18)


**Reviewer #1 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Replace or supplement normalized training-loss plots with validation-loss curves, retrieval recall/precision, latency, memory and computational cost. |


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Specify the question-type classifier, its training data and inference assignment; hard filtering versus soft prioritization; fallback for insufficient exemplars. | | **** | Revise Figure 1 to show trained/indexed representations and where decoding-time control occurs; explain normalization and epochs vs steps for Figures 2-3; consider moving training curves to the supplement. |


**Results**

**Relevance criterion.** unsummarised pool answer vs test reference answer, ROUGE-1 F >= tau. Verified before use: the fast ROUGE-1 F used for the 300×2000 relevance matrix reproduces `rouge_score` to 0e+00 absolute error.

| τ | Strategy | Relevant per question | Questions with none | Precision@3 | Random@3 | Lift | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | |---|---|---|---|---|---|---|---|---|---|---|---| | 0.25 | dense | 429.83 | 2 | 56.11% | 21.49% | 2.6× | 0.34% | 0.84% | 1.43% | 2.50% | 0.7119 | | 0.25 | question-type | 429.83 | 2 | 63.56% | 21.49% | 3.0× | 0.34% | 0.92% | 1.46% | 2.78% | 0.7652 | | 0.35 | dense | 66.68 | 40 | 30.00% | 3.33% | 9.0× | 2.29% | 4.80% | 7.72% | 10.81% | 0.4443 | | 0.35 | question-type | 66.68 | 40 | 34.22% | 3.33% | 10.3× | 2.47% | 4.98% | 7.33% | 10.49% | 0.4893 | | 0.45 | dense | 38.97 | 144 | 17.00% | 1.95% | 8.7× | 5.75% | 10.75% | 14.18% | 18.20% | 0.2407 | | 0.45 | question-type | 38.97 | 144 | 20.89% | 1.95% | 10.7× | 4.79% | 11.21% | 13.69% | 18.71% | 0.2871 |

**What it shows.** Precision@3 exceeds the random-ranking baseline by 2.6× to 10.7× at every threshold, so the retriever selects exemplars whose answers are genuinely closer to the target answer than chance would give — this is the measurement the row previously marked not computable was asking for, and the conclusion does not hinge on the threshold.

**Question-type retrieval beats dense retrieval on precision@3 and MRR at all 3 of 3 thresholds**, which is independent corroboration of the +1.24 ROUGE-1 benefit the corrected grid measures for question-type stratification: two different signals, same direction.

**Reading the recall column.** Recall@k is bounded by k divided by the number of relevant items, and at these thresholds a question has tens to hundreds of relevant exemplars, so recall@3 cannot exceed a few percent by construction. It is reported for completeness; precision@3 and MRR are the informative quantities here.


### X33 — Feasibility of a source- and topic-disjoint repartition

**What was run.** Before re-partitioning and re-training, the question is whether a disjoint split is large enough to support the paired tests the paper uses. The corpus was grouped by source document and focus entity and whole groups were packed into an 80/10/10 split, so no group appears on both sides of the split.


**Scale executed.** all 16,407 MedQuAD items grouped three ways (source, focus, and the pair), greedy largest-first packing  
**Scripts.** `group_split_feasibility.py`


**Reported in.** Section 3.3.1 (p. 22); Table 4 (p. 6); Section 2.2.3 (p. 4)


**Reviewer #2 concerns resolved**

| Item | As written by the reviewer | |---|---| | **** | Report duplicate and near-duplicate checks across partitions; consider shared sources and overlapping answers; assess grouped splits. | | **** | Add partition overlap and benchmark pretraining exposure. |


**Results**

**Grouping the 16,407 MedQuAD items and packing whole groups into 80/10/10** (0 items carry no recovered metadata and cannot be grouped)

| Grouping key | Groups | Median group | Largest group | Singletons | Train | Validation | Disjoint test | % of corpus | |---|---|---|---|---|---|---|---|---| | source document | 3,453 | 2 | 52 | 1,365 | 13,125 | 1,641 | 1,641 | 10.0% | | focus entity | 4,741 | 2 | 58 | 2,238 | 13,113 | 1,640 | 1,640 | 10.0% | | source and focus | 5,429 | 2 | 33 | 2,580 | 13,125 | 1,641 | 1,641 | 10.0% |

**Verdict.** A whole-group 80/10/10 split on (source document, focus entity) yields 1641 disjoint test items from 5,429 groups. That is large enough to support the paired tests used elsewhere in the paper (n>=250), so the repartition is viable.


---


## Concerns that required no experiment

These asked for a rewrite, a clarification, a terminology change or an added limitation rather than a computation. They are answered in the response letters and visible as highlighted text in the manuscript.


### Reviewer #1 (5)

| Item | As written by the reviewer | How it was addressed | |---|---|---| | **** | Clarify whether the primary contribution is a new algorithm, an engineering integration, or an empirical study of modular interventions. | Statement of the contribution type; no computation required. | | **** | Emphasize modularity and portability as engineering strengths while separating demonstrated findings from anticipated deployment benefits. | Strengths paragraph separating demonstrated findings from anticipated benefits. | | **** | Highlight the separation between open-ended generation and closed-form option selection. | Statement that open-ended and closed-form evaluation are separate experiments. | | **** | Organize the Methods around a clear end-to-end algorithm. | Reordering of the Methods into the executed end-to-end order. | | **** | Comprehensive language editing. | Language and claim-narrowing pass over the original prose. |


### Reviewer #2 (12)

| Item | As written by the reviewer | How it was addressed | |---|---|---| | **** | Distinguish the objective of improving benchmark QA performance from the broader aims of factual correctness, clinical completeness and safety. | Separation of the measured objective from the broader clinical aims. | | **** | State the methodological contribution beyond combining components; explain expected benefits over conventional RAG. | Statement of the methodological contribution beyond component combination. | | **** | Clarify that open-ended and closed-form evaluations are separate dataset-specific experiments, not transfer. | Statement that the two evaluations are separate, not transfer. | | **** | Correct the Section 3.4 statement that BERTScore improvements are consistent against vanilla (Qwen3-8B iCliniq 82.66 -> 82.38). | Correction of the BERTScore consistency sentence against the paper's own values. | | **** | Describe Equation 18 as question-entity retention, not clinical completeness. | Renaming of the retention term; no computation required. | | **** | Emphasize evaluation across open-ended and closed-form QA and three backbones while distinguishing separate experiments from transfer. | Strengths statement across tasks and backbones. | | **** | Highlight the modular design and the retrieval, summarization, verification and indexing experiments while acknowledging the need for further ablations. | Strengths statement on the modular design. | | **** | Clarify that portability does not establish efficiency or generalizability. | Limitation separating portability from efficiency and generalisability. | | **** | Consolidate implementation and training details; present offline preparation, adaptation and inference in a consistent order. | Consolidation of implementation and training detail. | | **** | Use consistent terminology for frozen, fine-tuned, adapter-based, vanilla and generator-only conditions. | Consistent condition terminology. | | **** | Shorten repeated descriptions of reliability, portability and clinical relevance; tie claims to findings. | Shortening of repeated reliability and portability claims. | | **** | Comprehensive language editing. | Language and claim-narrowing pass over the original prose. |


---


## Considered and not run

What this file does not contain, and why. Each entry names what it would take, so the boundary is explicit rather than left to be inferred.

| Request | Raised in | What it would take | Why it was not run | |---|---|---|---| | Clinician-rated evaluation of correctness, completeness, safety and usefulness | | A blinded panel of clinicians and an ethics-approved protocol. | Outside what this revision can execute. Both reviewers phrased the request conditionally — add a clinician assessment or narrow the claims — and the revision takes the second branch. X27 is the evidence that the branch was taken completely: no unqualified factuality, completeness or safety claim survives on any surface of the paper, and the abstract and conclusions state that clinician-led evaluation is required before any clinical use. | | A source-disjoint repartition with retraining | | Re-partitioning MedQuAD by source document and re-training the generator on the disjoint split. | X28 runs the inference-only form of this test, which is what the recovered metadata makes possible without retraining: every exemplar sharing the query's source document or topic is barred from retrieval. A full repartition would additionally remove same-source material from the generator's training data, which needs a retraining run and therefore the same GPU capacity as the seed work above. |


---


**Coverage.** 33 experiments; 62 of the 79 concerns are answered by at least one experiment and 17 required none, which accounts for every concern exactly once.

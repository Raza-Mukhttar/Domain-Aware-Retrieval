# preprocessing/

How every dataset in `data/` was produced, and how to rebuild it. Run the checker first
— it validates the shipped data against what these scripts produce, offline, in seconds:

```bash
python preprocessing/verify_preprocessing.py
```

| Script | Produces |
|---|---|
| `ingest_open_ended.py` | `data/medquad/` and `data/icliniq/` from the Hugging Face releases |
| `make_answer_summaries.py` | `data/derived/train_summaries_<corpus>.json`, the retrieval exemplars |
| `extract_entity_vocabulary.py` | `data/derived/medical_entities.json` |
| `build_group_disjoint_split.py` | `data/medquad_group_disjoint/`, the source/topic-disjoint repartition |
| `prepare_closed_form.py` | the four multiple-choice benchmarks, with the answer key validated |
| `verify_preprocessing.py` | checks the shipped data against all of the above |

## Open-ended corpora

```bash
python ingest_open_ended.py --dataset lavita/MedQuAD --outdir ../data/medquad
python ingest_open_ended.py --dataset lavita/ChatDoctor-iCliniq \
    --answer_col answer_icliniq --outdir ../data/icliniq
```

One object per line, fields `{question, context, answer}`. A release that ships a single
split is divided 80/10/10 **in original row order**, not shuffled, so the boundaries
belong to the release rather than to a seed; rows with an empty question or answer are
dropped after the split is taken. That is why the sizes are not exactly 80/10/10:

| Corpus | train | validation | test |
|---|---:|---:|---:|
| MedQuAD | 13,125 | 1,640 | 1,642 |
| iCliniq-style | 5,856 | 732 | 733 |

iCliniq reproduces exactly from 7,321 raw rows. MedQuAD's sizes are consistent with
16,411 raw rows and four dropped for an empty field.

**`--answer_col` is not optional in practice.** Without it the preference order is
`answer_chatdoctor`, `answer_chatgpt`, `answer_icliniq`, so a model's answer would become
the reference ahead of the physician's. Pass `--answer_col answer_icliniq`.

## Answer summaries

DeepSeek-R1-Distill-Llama-8B, greedy, 200 new tokens, with the prompt of Figure C.3. The
summaries are retrieval exemplars only — every reported metric scores against the
original dataset answer. The `~100 tokens` instruction in the prompt was not obeyed: on
MedQuAD the median summary is 1.81× the length of the answer it replaced, so a "summary"
is usually longer than its source. The script reports that ratio so the discrepancy is
visible rather than assumed away.

## Entity vocabulary

scispaCy `en_ner_bc5cdr_md` over the training summaries, lower-cased, entries longer than
two characters. **64.4% of entries contain a space**, so the vocabulary has to be matched
as phrases; matching it token-by-token makes those entries unreachable, which is the
defect `medlm.entities` corrects by matching normalised n-grams.

Install the model first, as it is not on PyPI:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

## Closed-form benchmarks

Pulled from the Hub at run time rather than stored, so `prepare_closed_form.py` exists to
make that preprocessing inspectable:

```bash
python prepare_closed_form.py --check
```

Each loader fixes a specific way the answer key can be unusable, and `--check` fails if
any key has fewer than two classes or a majority class above 95%:

- **MedMCQA** is evaluated on **validation**, because the test split ships `cop = -1` for
  all 6,150 rows. A loader that clamps that to 0 makes every gold answer "A", and the
  resulting accuracies are just how often a model answered "A".
- **PubMedQA** takes yes/no/maybe from `final_decision` on `pqa_labeled`, with the
  abstract as context. Defaulting a missing field to "yes" makes every gold "yes".
- **MMLU-Med** concatenates six medical subjects in a fixed order; `dev` + `validation`
  form the 153-row pool and `test` is 1,089 rows.
- **MedQA** takes gold from `answer_idx` over four options.

Both defects above were present in the earlier baseline runs, which is why those columns
were reported as unavailable rather than as numbers.

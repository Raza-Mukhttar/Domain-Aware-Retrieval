# -*- coding: utf-8 -*-
"""Entity-aware decoding control and candidate generation, Eq. (14)."""
import torch
from transformers import LogitsProcessor, LogitsProcessorList


class EntityBonusProcessor(LogitsProcessor):
    """Soft entity-aware DECODING control.

    At every step, for each target entity (tokenised once), if the generated
    suffix matches a prefix of the entity's token sequence, the logit of the
    entity's next token receives +gamma; the first token of any target entity not
    yet emitted receives +gamma as well. This modifies token selection during beam
    search and is therefore a decoding-time mechanism, in contrast to the
    reranking score, which acts on completed candidates. gamma=0 disables it.
    """

    def __init__(self, entity_token_seqs, gamma, prompt_len):
        self.entities = [torch.tensor(e) for e in entity_token_seqs if len(e) > 0]
        self.gamma = float(gamma)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if self.gamma == 0 or not self.entities:
            return scores
        generated = input_ids[:, self.prompt_len:]
        for b in range(input_ids.shape[0]):
            g = generated[b]
            for e in self.entities:
                e = e.to(g.device)
                # entity already present in the generated text: no bonus
                if len(g) >= len(e) and any(torch.equal(g[i:i + len(e)], e)
                                            for i in range(len(g) - len(e) + 1)):
                    continue
                boosted = False
                for L in range(min(len(e) - 1, len(g)), 0, -1):   # longest matching prefix first
                    if torch.equal(g[-L:], e[:L]):
                        scores[b, e[L]] += self.gamma
                        boosted = True
                        break
                if not boosted:
                    scores[b, e[0]] += self.gamma
        return scores


@torch.no_grad()
def generate(model, tokenizer, prompts, n_cand, max_new, max_input, batch_size,
             entity_seqs=None, gamma=0.0):
    """Return, for each prompt, a list of n_cand cleaned candidate strings.

    entity_seqs holds per-prompt token-id sequences for decoding-time entity
    control; it is used only when gamma > 0 and forces batch_size 1 so the
    processor sees one prompt's targets.
    """
    out = []
    model.eval()
    if gamma > 0:
        batch_size = 1
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_input).to(model.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=True)
        if gamma > 0 and entity_seqs is not None:
            kw["logits_processor"] = LogitsProcessorList(
                [EntityBonusProcessor(entity_seqs[s], gamma, enc["input_ids"].shape[1])])
        if n_cand > 1:
            kw.update(num_beams=n_cand, num_return_sequences=n_cand,
                      do_sample=False, early_stopping=True)
        else:
            kw.update(num_beams=1, do_sample=False)
        gen = model.generate(**enc, **kw)
        prompt_len = enc["input_ids"].shape[1]
        # exact prompt removal by token slicing
        text = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append([t.strip() for t in text[i * n_cand:(i + 1) * n_cand]])
        del enc, gen
    return out


# Prompt sets: which retrieval variant assembles the prompt.
PROMPT_SETS = {
    "norag":        dict(use_retrieval=False, use_question_type=False, exemplar_field=None),
    "rag":          dict(use_retrieval=True,  use_question_type=False, exemplar_field="summary"),
    "rag_qtype":    dict(use_retrieval=True,  use_question_type=True,  exemplar_field="summary"),
    "rag_fullans":  dict(use_retrieval=True,  use_question_type=False, exemplar_field="answer"),
    "rag_edec":     dict(use_retrieval=True,  use_question_type=False, exemplar_field="summary", gamma=1.0),
    "norag_edec":   dict(use_retrieval=False, use_question_type=False, exemplar_field=None, gamma=1.0),
    "rag_qtype_fullans":      dict(use_retrieval=True, use_question_type=True, exemplar_field="answer"),
    "rag_qtype_fullans_edec": dict(use_retrieval=True, use_question_type=True, exemplar_field="answer", gamma=1.0),
}

# Selection rules, Eq. (18). All four are applied to the SAME candidate pool, so
# the ablations differ only in the scoring function, never in the generated text.
SELECTORS = {
    "first":    dict(alpha=None, use_verifier=False, use_entity=False),
    "verifier": dict(alpha=1.00, use_verifier=True,  use_entity=False),
    "entity":   dict(alpha=0.00, use_verifier=False, use_entity=True),
    "combined": dict(alpha=0.75, use_verifier=True,  use_entity=True),
}

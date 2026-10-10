# -*- coding: utf-8 -*-
"""Prompt assembly, Eq. (9) of the paper.

Prompt(q) = {(q_i1, s_i1), ..., (q_ik, s_ik), q}: the k retrieved exemplar
question-summary pairs followed by the target question. The instruction block is
reproduced verbatim from the executed runs; changing it changes every number.
"""
INSTRUCTION = ("You are a medical expert.\n"
               "Use clinical reasoning to answer concisely.\n"
               "Answer must be related to the question.\n")


def build_prompt(question, exemplars, exemplar_field):
    """exemplar_field selects what the exemplar contributes: "summary" for the
    machine summary, "answer" for the unsummarised training answer, None for the
    no-retrieval condition."""
    parts = [INSTRUCTION]
    for ex in exemplars:
        parts.append(f"Q: {ex['question']}\nA: {ex[exemplar_field]}\n---\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)

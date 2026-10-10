# -*- coding: utf-8 -*-
"""Tokeniser loading that is verified to decode correctly.

With transformers 5.17, DeepSeek-R1-Distill-Llama-8B resolves to the legacy slow `LlamaTokenizer`
even though its config asks for `LlamaTokenizerFast`, and that slow path drops every space on
decode: "What are the symptoms" comes back as "Whatarethesymptoms". Encoding is unaffected, so
training is unaffected, but any generated text decoded through it is destroyed.

`load_tokenizer` therefore round-trips a probe string and, if the spaces do not survive, rebuilds
the tokeniser directly from tokenizer.json as a PreTrainedTokenizerFast, which decodes correctly.
Every caller in this codebase goes through here so the defect cannot reappear silently.
"""
import os, glob
from transformers import AutoTokenizer, PreTrainedTokenizerFast
PROBE = "What are the symptoms of Rotor syndrome ?"
def decodes_cleanly(tok):
    try:
        ids = tok(PROBE)["input_ids"]
        out = tok.decode(ids, skip_special_tokens=True)
    except Exception:
        return False
    # the probe has six spaces; a decoder that drops them is broken for our purposes
    return out.count(" ") >= PROBE.count(" ") - 1
def _tokenizer_json(path):
    if os.path.isdir(path):
        p = os.path.join(path, "tokenizer.json")
        if os.path.exists(p): return p
    g = glob.glob(os.path.join(path, "**", "tokenizer.json"), recursive=True)
    return g[0] if g else None
def load_tokenizer(path, **kw):
    tok = AutoTokenizer.from_pretrained(path, **kw)
    if decodes_cleanly(tok):
        return tok
    tj = _tokenizer_json(path)
    if not tj:
        print(f"[tok_util] WARNING: {path} decodes without spaces and has no tokenizer.json to fall back to", flush=True)
        return tok
    fast = PreTrainedTokenizerFast(tokenizer_file=tj)
    for attr in ("bos_token", "eos_token", "unk_token", "pad_token"):
        v = getattr(tok, attr, None)
        if v is not None:
            try: setattr(fast, attr, v)
            except Exception: pass
    if getattr(fast, "pad_token", None) is None:
        fast.pad_token = fast.eos_token
    ok = decodes_cleanly(fast)
    print(f"[tok_util] {os.path.basename(str(path).rstrip('/'))}: {type(tok).__name__} dropped spaces on decode; "
          f"rebuilt from tokenizer.json as PreTrainedTokenizerFast (clean={ok})", flush=True)
    return fast if ok else tok

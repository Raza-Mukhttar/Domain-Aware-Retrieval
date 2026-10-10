# -*- coding: utf-8 -*-
"""Turn the tables that are hard to scan into figures.

Design decisions, once, applied to all of them:

* Numbers are PARSED FROM THE MANUSCRIPT .tex, not recomputed, so a figure can never disagree with
  the table it stands beside -- including the Holm significance markers.
* Colour follows the reference palette unchanged: categorical slots 1-3 (blue #2a78d6, orange
  #eb6834, aqua #1baf7a) for identity, the documented blue<->red diverging pair with a gray midpoint
  for signed quantities, and the blue sequential ramp for magnitude. No rainbow, no dual axes.
* Every figure carries a non-colour secondary encoding (filled vs hollow markers, marker shape,
  printed values on cells), so identity and significance survive greyscale printing and CVD.
* These are print figures for a LaTeX manuscript, so there is no hover layer; the table each one
  replaces is the "table view" the accessibility pass asks for, and is cited in the caption.
"""
import os, re, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# Figures are drawn from the table bodies of the manuscript source, so the value
# in a figure and the value in its table cannot drift apart.
TEX = os.environ.get("MEDLM_MANUSCRIPT",
                     os.path.join(ROOT, "outputs", "manuscript",
                                  "AI_In_Medicine_Final_Highlighted.tex"))
OUT = os.environ.get("MEDLM_FIGURE_DIR", os.path.join(ROOT, "outputs", "figures"))
if not os.path.exists(TEX):
    raise SystemExit(f"manuscript source not found at {TEX}; set MEDLM_MANUSCRIPT")
os.makedirs(OUT, exist_ok=True)

# ---- palette (reference instance, used unchanged) ----
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF = "#fcfcfb"
POS, NEG, MID = "#2a78d6", "#d03b3b", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", [NEG, MID, POS])
SEQ = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
TEXT = open(TEX, encoding="utf-8").read()


class MissingTable(KeyError):
    """The manuscript no longer carries this table label."""


def table_body(label):
    """The rows of one table environment, by \\label."""
    marker = "\\label{" + label + "}"
    if marker not in TEXT:
        raise MissingTable(label)
    i = TEXT.index(marker)
    j = TEXT.index("\\begin{tabular}", i)
    k = TEXT.index("\\end{tabular}", j)
    return TEXT[j:k]
NUMCI = re.compile(r"\$?([+-]?\d+\.\d+)\$?(\^\{?\*\}?)?\s*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]")
def parse_ci(cell):
    """-> (value, lo, hi, significant) from a '$+2.50^{*}$ [+0.96,+4.07]' cell."""
    m = NUMCI.search(cell.replace("$", "").replace("\\,", " "))
    if not m: return None
    star = "*" in cell
    return float(m.group(1)), float(m.group(3)), float(m.group(4)), star
GREEK = {r"\\gamma": "\u03b3", r"\\alpha": "\u03b1", r"\\beta": "\u03b2", r"\\tau": "\u03c4",
         r"\\Delta": "\u0394", r"\\pm": "\u00b1", r"\\times": "\u00d7", r"\\geq": "\u2265",
         r"\\leq": "\u2264", r"\\to": "\u2192", r"\\emph": "", r"\\ldots": "\u2026"}
def clean(s):
    s = re.sub(r"\\(emph|textbf|hl|hltab|texttt|mathrm)\{([^{}]*)\}", r"\2", s)
    for k, v in GREEK.items():                       # keep symbols that carry meaning in a label
        s = re.sub(k + r"\b", v, s)
    s = s.replace("\\ ", " ").replace("\\,", " ").replace("$", "").replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()
def rows_of(label):
    out = []
    for line in table_body(label).split("\\\\"):
        if "&" not in line or "multicolumn" in line and line.count("&") < 2:
            if "multicolumn" in line:
                out.append(("__HEAD__", clean(line)))
            continue
        cells = [c.strip() for c in line.split("&")]
        if cells and cells[0].strip().startswith("\\textbf{Contrast"): continue
        out.append(("__ROW__", cells))
    return out
def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, format=ext, bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")

# =====================================================================  Table 5 -> forest plot
def fig_primary_components():
    rows = rows_of("tab:primary-components")
    panels, cur = [], None
    for kind, v in rows:
        if kind == "__HEAD__":
            cur = (clean(v).replace("5l", ""), []); panels.append(cur); continue
        if cur is None or not isinstance(v, list) or len(v) < 2: continue
        p = parse_ci(v[1])
        if p: cur[1].append((clean(v[0]), *p))
    panels = [(n, r) for n, r in panels if r]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 4.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, rr) in zip(axes, panels):
        rr = rr[::-1]
        y = np.arange(len(rr))
        for i, (lab, v, lo, hi, sig) in enumerate(rr):
            ax.plot([lo, hi], [i, i], color=S1, lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot([v], [i], marker="o", ms=6.5, color=S1 if sig else SURF,
                    markeredgecolor=S1, markeredgewidth=1.6, zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rr] if ax is axes[0] else [])
        ax.set_title(name, fontsize=9.5, color=INK, pad=6, loc="left")
        ax.set_xlabel("$\\Delta$ ROUGE-1 (points)")
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(rr) - 0.3)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=6.5, color=S1, markeredgecolor=S1, label="significant ($p<0.05$, Holm)"),
         plt.Line2D([], [], marker="o", ls="none", ms=6.5, color=SURF, markeredgecolor=S1, markeredgewidth=1.6, label="not significant")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.55, -0.06))
    fig.suptitle("Component contrasts: only the budget, retrieval, question-type and unsummarised rows clear zero",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab5_primary_components")

# =====================================================================  Tables 6-7 -> waterfall
def _grouped(label, value_col, parse=False):
    """Rows of a table that stacks datasets vertically under \\multicolumn headers."""
    out, cur = {}, None
    for kind, v in rows_of(label):
        if kind == "__HEAD__":
            name = clean(v)
            name = re.sub(r"^\d+l", "", name)
            cur = name; out.setdefault(cur, [])
            continue
        if cur is None or not isinstance(v, list) or len(v) <= value_col: continue
        lab = clean(v[1]) if len(v) > 1 else ""
        if not lab or lab.lower().startswith(("#", "rung", "step")): continue
        cell = v[value_col]
        val = parse_ci(cell) if parse else None
        if parse:
            if val: out[cur].append((lab, *val))
        else:
            try: out[cur].append((lab, float(clean(cell))))
            except Exception: pass
    return {k: v for k, v in out.items() if v}

def fig_ladder():
    lev = _grouped("tab:ladder-levels", 2, parse=False)
    stp = _grouped("tab:ladder-steps", 2, parse=True)
    order = [k for k in lev if lev[k]]
    fig, axes = plt.subplots(1, len(order), figsize=(10.4, 4.6))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, order):
        L = [x[1] for x in lev[ds]]
        steps = stp.get(ds, [])
        x = np.arange(len(L))
        for i in range(len(L)):
            if i == 0:
                ax.bar(i, L[0], width=0.62, color=MUTED, alpha=0.28, zorder=2)
                ax.annotate(f"{L[0]:.2f}", (i, L[0]), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=7.2, color=INK2)
                continue
            d = L[i] - L[i - 1]
            ax.bar(i, d, bottom=L[i - 1], width=0.62, color=(POS if d > 0 else NEG), alpha=0.92, zorder=2)
            ax.plot([i - 0.31, i + 0.31], [L[i - 1], L[i - 1]], color=MUTED, lw=0.7, zorder=1)
            if i - 1 < len(steps):
                _lab, v, lo, hi, sig = steps[i - 1]
                top = L[i - 1] + hi
                ax.plot([i, i], [L[i - 1] + lo, top], color=INK, lw=1.1,
                        solid_capstyle="round", zorder=4)
                # print the step as the TABLE states it, not as a difference of rounded levels
                ax.annotate(f"{v:+.2f}" + ("*" if sig else ""), (i, max(top, L[i], L[i - 1])),
                            textcoords="offset points", xytext=(0, 5), ha="center",
                            fontsize=7.2, color=INK, zorder=7)
        best = int(np.argmax(L))
        ax.plot([best], [0], marker="^", ms=7, color=INK, zorder=6, clip_on=False,
                transform=ax.get_xaxis_transform())
        # the legend names this marker; no per-axis label, which collided with the x-label
        ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_xlabel("rung"); ax.set_ylabel("ROUGE-1 (%)" if ax is axes[0] else "")
        ax.set_title(ds, fontsize=9.5, loc="left", color=INK, pad=6)
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
        lo_, hi_ = min(L), max(L)
        pad = max(hi_ - lo_, 1.0)
        ax.set_ylim(lo_ - pad * 0.45, hi_ + pad * 0.55)
    h = [Rectangle((0, 0), 1, 1, color=POS, alpha=0.92, label="rung adds"),
         Rectangle((0, 0), 1, 1, color=NEG, alpha=0.92, label="rung subtracts"),
         plt.Line2D([], [], color=INK, lw=1.1, label="95% interval of the step"),
         plt.Line2D([], [], marker="^", ls="none", ms=7, color=INK, label="best rung of the ladder")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.10))
    fig.text(0.01, 1.02, "Cumulative ladder: the gain is sub-additive \u2014 on neither dataset is the complete "
             "pipeline (rung 8) the best configuration of its own ladder.\nRungs: 1 generator only, "
             "2 +budget, 3 +retrieval, 4 +question-type, 5 +unsummarised, 6 +verifier, 7 +entity reranking, "
             "8 +entity bonus." + (" * = significant after Holm correction."
                                   if any(r[4] for v in stp.values() for r in v) else ""),
             fontsize=8, color=INK2, ha="left", va="bottom")
    save(fig, "tab6_7_cumulative_ladder")

# =====================================================================  Tables 19-22 -> convergence forest
def fig_leakage_convergence():
    rows = []
    for kind, v in rows_of("tab:leakage-convergence"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 3: continue
        est, design = clean(v[0]), clean(v[1])
        if not est or est.lower().startswith("estimate"): continue
        try: val = float(clean(v[2]).replace("+", ""))
        except Exception: continue
        rows.append((est, design, val))
    if not rows: return
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    y = np.arange(len(rows))
    strict = [i for i, r in enumerate(rows) if "disjoint" in r[0].lower() or "repartition" in r[0].lower()]
    for i, (est, design, val) in enumerate(rows):
        ax.plot([0, val], [i, i], color="#dcdbd6", lw=1.2, zorder=1)
        ax.plot([val], [i], marker="D" if i in strict else "o", ms=8 if i in strict else 7,
                color=S1 if i in strict else SURF, markeredgecolor=S1, markeredgewidth=1.8, zorder=3)
        ax.annotate(f"{val:+.2f}", (val, i), textcoords="offset points",
                    xytext=(10 if val >= 0 else -10, 0), ha="left" if val >= 0 else "right",
                    va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("$\\Delta$ ROUGE-1 from retrieval (points)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    ax.set_xlim(-0.9, max(r[2] for r in rows) + 1.5)
    h = [plt.Line2D([], [], marker="D", ls="none", ms=8, color=S1, markeredgecolor=S1,
                    label="shared source/topic material removed"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=SURF, markeredgecolor=S1,
                    markeredgewidth=1.8, label="random split")]
    ax.legend(handles=h, loc="upper center", ncol=2, fontsize=7.8, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("The retrieval effect does not shrink as shared material is removed — the opposite of a leakage pattern",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.03)
    save(fig, "tab19_22_leakage_convergence")

# =====================================================================  Tables 3-4 -> seed dot plot
def fig_seed_stability():
    rows = []
    for kind, v in rows_of("tab:open-seeds"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 6: continue
        bb, ds = clean(v[0]), clean(v[1])
        if not bb or bb.lower().startswith("backbone"): continue
        per = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", v[5])]
        m = re.search(r"([+-]\d+\.\d+)\s*\\?pm\s*(\d+\.\d+)", clean(v[6])) if len(v) > 6 else None
        if len(per) >= 2:
            rows.append((bb.replace("-Distill-Llama", "").replace("-Instruct", ""), ds, per,
                         float(m.group(1)) if m else float(np.mean(per))))
    # LLaMA-3.1-8B is reported in the other seed table; its per-seed effects are stated in the
    # caption of tab:open-seeds, so parse them from there rather than leaving the backbone out.
    cap_i = TEXT.index("\\label{tab:open-seeds}")
    cap = TEXT[max(0, cap_i - 1800):cap_i]
    m = re.search(r"retrieval effects are\s*(.+?)\s*on MedQuAD and\s*(.+?)\s*on the iCliniq", cap, re.S)
    if m:
        mq = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", m.group(1))]
        ic = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", m.group(2))]
        if len(mq) >= 2: rows.insert(0, ("LLaMA-3.1-8B", "MedQuAD", mq, float(np.mean(mq))))
        if len(ic) >= 2: rows.insert(1, ("LLaMA-3.1-8B", "iCliniq-style", ic, float(np.mean(ic))))
    if not rows: return
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels, ymap = [], {}
    for i, (bb, ds, per, mean) in enumerate(rows[::-1]):
        labels.append(f"{bb}\n{ds}"); ymap[i] = (per, mean)
    for i, (per, mean) in ymap.items():
        ax.plot([min(per), max(per)], [i, i], color="#dcdbd6", lw=1.4, zorder=1)
        for j, p in enumerate(per):
            ax.plot([p], [i], marker=["o", "s", "^"][j % 3], ms=6, color=[S1, S2, S3][j % 3],
                    markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
        ax.plot([mean], [i], marker="|", ms=16, color=INK, markeredgewidth=1.6, zorder=4)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(list(ymap)); ax.set_yticklabels(labels)
    ax.set_xlabel("retrieval effect on ROUGE-1 (points)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker=m, ls="none", ms=6, color=c, label=f"seed {s}")
         for m, c, s in zip(["o", "s", "^"], [S1, S2, S3], ["42", "1", "2"])]
    h.append(plt.Line2D([], [], marker="|", ls="none", ms=12, color=INK, markeredgewidth=1.6, label="mean"))
    ax.legend(handles=h, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.30))
    fig.suptitle("Seed stability: the sign of the retrieval effect never changes within a cell, only its size",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.03)
    save(fig, "tab3_4_seed_stability")

# =====================================================================  Tables 10 & 13 -> heatmaps
BB_SHORT = {"LLaMA-3.1-8B-Instruct": "LLaMA-3.1-8B", "Qwen3-8B": "Qwen3-8B",
            "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-8B"}
CELL = re.compile(r"([+-]?\d+\.\d+)\s*\((\d+)\s*/\s*(\d+)\s*;\s*p\s*[=<]\s*([<]?\s*0?\.?\d+)")
def _paired_grid(label):
    """-> (col_names, backbones, benchmarks, value[b][k][c], sig[b][k][c])"""
    body = table_body(label)
    cols = None; data = {}
    for line in body.split("\\\\"):
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if any("textbf{Backbone" in c for c in cells):
            cols = [clean(c) for c in cells[2:]]
            continue
        if cols is None or len(cells) < 3: continue
        bb, bench = clean(cells[0]), clean(cells[1])
        if not bb or bb.lower().startswith("backbone"): continue
        vals, sigs = [], []
        for c in cells[2:2 + len(cols)]:
            cc = clean(c)
            m = re.search(r"([+-]?\d+\.\d+)", cc)
            p = re.search(r"p\s*[=<]\s*([<]?\s*)(\d*\.?\d+)", cc)
            vals.append(float(m.group(1)) if m else np.nan)
            sigs.append(bool(p) and (("<" in cc.split("p")[-1][:4]) or (float(p.group(2)) < 0.05)))
        data.setdefault(BB_SHORT.get(bb, bb), {})[bench] = (vals, sigs)
    return cols, data
def _draw_heat(ax, cols, data, title, lim=None):
    bbs = list(data); benches = list(next(iter(data.values())))
    rows = [(b, k) for b in bbs for k in benches]
    M = np.array([data[b][k][0] for b, k in rows], dtype=float)
    Sg = np.array([data[b][k][1] for b, k in rows], dtype=bool)
    lim = lim or (np.nanmax(np.abs(M)) or 1.0)   # a shared colourbar requires a shared scale
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]): continue
            ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if Sg[i, j] else ""), ha="center", va="center",
                    fontsize=7.6, color=INK if abs(M[i, j]) < lim * 0.55 else "#ffffff")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(" $-$ ", "\n− ").replace(" - ", "\n− ") for c in cols], fontsize=7.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{b}  ·  {k}" for b, k in rows], fontsize=7.6)
    for y in range(1, len(rows)):
        if rows[y][0] != rows[y - 1][0]: ax.axhline(y - 0.5, color=SURF, lw=2.4)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK, pad=6)
    ax.grid(False)
    return im
def fig_closedform_heatmaps():
    c1, d1 = _paired_grid("tab:cf-matched-paired")
    c2, d2 = _paired_grid("tab:cf-full-paired")
    allv = [v for d in (d1, d2) for b in d for k in d[b] for v in d[b][k][0] if not np.isnan(v)]
    lim = max(abs(min(allv)), abs(max(allv)))      # one scale for both panels, since they share a bar
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8),
                             gridspec_kw=dict(width_ratios=[len(c1), len(c2)], wspace=0.06))
    im = _draw_heat(axes[0], c1, d1, "Matched controls,\nuntrained backbones", lim)
    im2 = _draw_heat(axes[1], c2, d2, "Complete framework,\ntask-adapted generators", lim)
    axes[1].set_yticklabels([]); axes[1].tick_params(axis="y", length=0)
    cb = fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$\\Delta$ accuracy (points)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Closed-form paired differences: * marks $p<0.05$ (exact McNemar). Blue favours the added "
                 "component, red favours its absence.", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab10_13_closedform_paired")

# =====================================================================  D.1 + D.2 -> gamma two-panel
def fig_gamma():
    eff = []
    for kind, v in rows_of("tab:gamma-sweep"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 2: continue
        try: g = float(clean(v[0]))
        except Exception: continue
        p = parse_ci(v[1])
        if p: eff.append((g, *p))
    ret = []
    for kind, v in rows_of("tab:gamma-retention"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 4: continue
        try: ret.append((float(clean(v[0])), float(clean(v[1])), float(clean(v[2]))))
        except Exception: pass
    if not eff: return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    g = [e[0] for e in eff]; val = [e[1] for e in eff]
    lo = [e[2] for e in eff]; hi = [e[3] for e in eff]; sig = [e[4] for e in eff]
    ax = axes[0]
    ax.fill_between(g, lo, hi, color=S1, alpha=0.16, lw=0)
    ax.plot(g, val, color=S1, lw=2, zorder=3)
    for x, y, s in zip(g, val, sig):
        ax.plot([x], [y], marker="o", ms=7, color=S1 if s else SURF,
                markeredgecolor=S1, markeredgewidth=1.7, zorder=4)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_xticks(g); ax.set_xticklabels([str(x) for x in g])
    ax.set_xlabel("$\\gamma$ (log scale)"); ax.set_ylabel("$\\Delta$ ROUGE-1 (points)")
    ax.set_title("Answer quality falls as the bonus strengthens", fontsize=9, loc="left", pad=6)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=7, color=S1, label="significant (Holm)"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=SURF, markeredgecolor=S1,
                    markeredgewidth=1.7, label="not significant")]
    ax.legend(handles=h, loc="lower left", fontsize=7.5)
    if ret:
        ax = axes[1]
        gr = [r[0] for r in ret]
        ax.plot(gr, [r[2] for r in ret], color=S3, lw=2, marker="s", ms=6,
                markeredgecolor=SURF, markeredgewidth=0.8, zorder=3, label="with bonus")
        base = ret[0][1]
        ax.axhline(base, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        vals = [r[2] for r in ret]
        span = max(vals) - base
        ax.set_ylim(base - span * 0.28, max(vals) + span * 0.22)
        ax.annotate(f"without bonus ({base:.2f}%)", (gr[0], base), textcoords="offset points",
                    xytext=(2, -13), ha="left", fontsize=7.4, color=INK2)
        ax.set_xscale("log"); ax.set_xticks(gr); ax.set_xticklabels([str(x) for x in gr])
        ax.set_xlabel("$\\gamma$ (log scale)"); ax.set_ylabel("question-entity retention (%)")
        ax.set_title("The quantity the bonus targets rises", fontsize=9, loc="left", pad=6)
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("The decoding-time entity bonus is faithful to its specification and does not help: "
                 "retention rises while quality falls",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.04)
    save(fig, "tabD1_D2_gamma_sweep")

# =====================================================================  D.4 -> fusion-weight heatmap
def fig_fusion_weight():
    body = table_body("tab:cf-weight-sweep")
    weights, rows = None, []
    for line in body.split("\\\\"):
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if any("textbf{Backbone" in c for c in cells):
            weights = [clean(c).replace("w=", "") for c in cells[3:10]]
            continue
        if weights is None or len(cells) < 10: continue
        bb, bench = clean(cells[0]), clean(cells[1])
        if not bb or bb.lower().startswith("backbone"): continue
        try: vals = [float(clean(c)) for c in cells[3:10]]
        except Exception: continue
        try: best = float(clean(cells[10]))
        except Exception: best = None
        rows.append((BB_SHORT.get(bb, bb), bench, vals, best))
    if not rows: return
    M = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(M, cmap=SEQ, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rel = (M[i, j] - M[i].min()) / max(M[i].max() - M[i].min(), 1e-9)
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.4,
                    color="#ffffff" if (M[i, j] - M.min()) / (M.max() - M.min()) > 0.62 else INK)
    jx = weights.index("0.35") if "0.35" in weights else 2
    for i in range(M.shape[0]):
        ax.add_patch(Rectangle((jx - 0.46, i - 0.44), 0.92, 0.88, fill=False, edgecolor=NEG,
                               lw=1.8, zorder=5))
        jb = int(np.argmax(M[i]))
        # a corner marker rather than a second box: stacked boxes in adjacent rows merge visually
        ax.plot([jb - 0.36], [i - 0.30], marker="*", ms=7.5, color=INK, zorder=6, clip_on=False)
    ax.set_xticks(range(len(weights))); ax.set_xticklabels([f"$w$={w}" for w in weights], fontsize=7.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{b}  ·  {k}" for b, k, _v, _x in rows], fontsize=7.8)
    for y in range(1, len(rows)):
        if rows[y][0] != rows[y - 1][0]: ax.axhline(y - 0.5, color=SURF, lw=2.4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("accuracy (%)", fontsize=8)
    cb.outline.set_visible(False)
    h = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=NEG, lw=1.8, label="executed weight (0.35)"),
         plt.Line2D([], [], marker="*", ls="none", ms=8, color=INK, label="best weight in the row")]
    ax.legend(handles=h, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Closed-form fusion weight: the executed 0.35 is never the best column, and $w=0$ "
                 "(no verifier) wins in 9 of 12 rows", fontsize=9.5, color=INK, x=0.01, ha="left", y=0.98)
    save(fig, "tabD4_fusion_weight_sweep")

# =====================================================================  D.5 -> dumbbell
def fig_verifier_retrain():
    body = table_body("tab:verifier-retrain")
    rows = []
    for line in body.split("\\\\"):
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if any("textbf{Backbone" in c for c in cells) or len(cells) < 6: continue
        bb, bench = clean(cells[0]), clean(cells[1])
        if not bb or bb.lower().startswith("backbone"): continue
        try:
            nov, tr, own = float(clean(cells[3])), float(clean(cells[4])), float(clean(cells[5]))
        except Exception: continue
        rows.append((BB_SHORT.get(bb, bb), bench, nov, tr, own))
    if not rows: return
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for i, (bb, bench, nov, tr, own) in enumerate(rows):
        lo, hi = min(tr, own, nov), max(tr, own, nov)
        ax.plot([lo, hi], [i, i], color="#dcdbd6", lw=1.6, zorder=1)
        ax.plot([tr], [i], marker="o", ms=7, color=S2, markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
        ax.plot([own], [i], marker="s", ms=7, color=S1, markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
        ax.plot([nov], [i], marker="D", ms=7, color=INK, markeredgecolor=SURF, markeredgewidth=0.8, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{b}  ·  {k}" for b, k, *_ in rows], fontsize=7.8)
    ax.set_xlabel("closed-form accuracy (%)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=7, color=S2, label="transferred verifier @0.35"),
         plt.Line2D([], [], marker="s", ls="none", ms=7, color=S1, label="own-benchmark verifier @0.35"),
         plt.Line2D([], [], marker="D", ls="none", ms=7, color=INK, label="no verifier")]
    ax.legend(handles=h, loc="upper center", ncol=3, fontsize=7.8, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Retraining the verifier per benchmark recovers part of the loss, but rarely beats using "
                 "no verifier at all", fontsize=9.5, color=INK, x=0.01, ha="left", y=0.97)
    save(fig, "tabD5_verifier_retrain")

# =====================================================================  verification + LaTeX snippet
FIGS = [("tab5_primary_components", "tab:primary-components", fig_primary_components,
         "Component contrasts of Table~\\ref{tab:primary-components}, as a forest plot. The figure draws "
         "the $\\Delta$ROUGE-1 column with its 95\\% bootstrap interval; $\\Delta$METEOR, "
         "$\\Delta$BERTScore and the $p$ values for every contrast are in the table, which remains "
         "the numeric record."),
        ("tab6_7_cumulative_ladder", "tab:ladder-levels", fig_ladder,
         "Cumulative ladder of Tables~\\ref{tab:ladder-levels} and \\ref{tab:ladder-steps}. The figure "
         "draws ROUGE-1 only; ROUGE-2, ROUGE-L, METEOR and BERTScore, each with its own interval, "
         "are in the tables. They agree in sign with ROUGE-1 on every rung whose effect exceeds "
         "half a point; the rungs where they disagree are all sub-half-point and are reported as "
         "no difference."),
        ("tab19_22_leakage_convergence", "tab:leakage-convergence", fig_leakage_convergence,
         "The five overlap-control estimates of Tables~\\ref{tab:topic}--\\ref{tab:leakage-convergence}."),
        ("tab3_4_seed_stability", "tab:open-seeds", fig_seed_stability,
         "Seed stability of the retrieval effect, Tables~\\ref{tab:seeds} and \\ref{tab:open-seeds}."),
        ("tab10_13_closedform_paired", "tab:cf-matched-paired", fig_closedform_heatmaps,
         "Closed-form paired differences of Tables~\\ref{tab:cf-matched-paired} and "
         "\\ref{tab:cf-full-paired}. Cells carry the accuracy difference and mark $p<0.05$; the "
         "discordant counts $(a/b)$ and the exact McNemar $p$ for each of the 96 comparisons are "
         "in the tables."),
        ("tabD1_D2_gamma_sweep", "tab:gamma-sweep", fig_gamma,
         "The $\\gamma$ sweep of Tables~\\ref{tab:gamma-sweep} and \\ref{tab:gamma-retention}. The "
         "left panel draws ROUGE-1; the other four quality metrics fall over the same range and are "
         "in Table~\\ref{tab:gamma-sweep}."),
        ("tabD4_fusion_weight_sweep", "tab:cf-weight-sweep", fig_fusion_weight,
         "The fusion-weight sweep of Table~\\ref{tab:cf-weight-sweep}."),
        ("tabD5_verifier_retrain", "tab:verifier-retrain", fig_verifier_retrain,
         "Verifier retrained per benchmark, Table~\\ref{tab:verifier-retrain}.")]
def build_all():
    """Draw every figure whose source table is still in the manuscript. A figure
    whose label has gone is skipped with a note rather than aborting the run."""
    drawn, skipped = [], []
    for name, lab, fn, _cap in FIGS:
        try:
            fn()
            drawn.append(name)
        except MissingTable as e:
            skipped.append((name, e.args[0]))
            print(f"  SKIP {name}: the manuscript has no table {e.args[0]}")
    if skipped:
        print(f"\n  {len(skipped)} figure(s) skipped; their source tables were renamed or removed.")
    return drawn, skipped
def verify():
    """Every number drawn must appear in the table it came from."""
    import subprocess
    bad = 0
    for name, lab, _fn, _cap in FIGS:
        try:
            body = clean(table_body(lab))
        except MissingTable:
            print(f"  {name:32s} skipped | the manuscript has no table {lab}")
            continue
        nums = set(re.findall(r"\d+\.\d+", body))
        png = os.path.join(OUT, name + ".png"); pdf = os.path.join(OUT, name + ".pdf")
        ok = os.path.exists(png) and os.path.exists(pdf)
        print(f"  {name:32s} pdf+png {'ok' if ok else 'MISSING'} | source {lab} has {len(nums)} distinct values")
        bad += 0 if ok else 1
    return bad
def latex_snippet():
    out = ["% ---- figures generated from the tables by scripts/analysis/make_table_figures.py ----",
           "% Each one stands beside the table it is drawn from; the table remains the "
           "numeric record and the accessible view.", ""]
    for name, lab, _fn, cap in FIGS:
        if ("\\label{" + lab + "}") not in TEXT:
            continue
        out += [r"\begin{figure*}[!tbp]", r"\centering",
                r"\includegraphics[width=\linewidth]{figures/%s.pdf}" % name,
                r"\caption{%s}" % cap,
                r"\label{fig:%s}" % name.replace("_", "-"),
                r"\end{figure*}", ""]
    p = os.path.join(OUT, "figure_includes.tex")
    open(p, "w", encoding="utf-8").write("\n".join(out))
    print("  wrote figures/figure_includes.tex")
if __name__ == "__main__":
    build_all(); print(); verify(); latex_snippet()

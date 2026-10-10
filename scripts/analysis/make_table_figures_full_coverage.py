# -*- coding: utf-8 -*-
"""Full-coverage figures: every numeric column of every source table is drawn, so a figure can be
read in place of its table without losing a number. Built because the replies to and
promise intervals, paired tests and the complete ablation across ALL metrics, not ROUGE-1
alone.

Each builder records every value it encodes (as geometry or as text) through rec(); verify_full()
then checks that record against the numbers actually present in the source table, so coverage is
proved rather than asserted.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table_figures import (MissingTable, table_body, clean, save, BB_SHORT, S1, S2, S3, INK, INK2, MUTED, SURF,
                           POS, NEG, MID, DIVERGING, SEQ)

S4, S5 = "#eda100", "#4a3aa7"                 # documented categorical slots 4 and 7
METRIC5 = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"]
MCOL = [S1, S2, S3, S4, S5]

# ----------------------------------------------------------------- coverage bookkeeping
REC = []
def rec(*vals):
    for v in vals:
        if v is None: continue
        REC.append(round(float(v), 4))
def rec_reset(): REC.clear()
def table_numbers(label, skip=1):
    """Every number printed in the data columns of a table, as floats. `skip` label columns on the
    left are excluded, so row names like N=5 or k=3 are not mistaken for data."""
    out = []
    for line in table_body(label).split("\\midrule", 1)[-1].split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line or "multicolumn" in line: continue
        cells = line.split("&")[skip:]
        if not cells: continue
        txt = " ".join(cells)
        if re.search(r"textbf\{(ROUGE|METEOR|BERT|Benchmark|Rung|Step|Contrast)", txt): continue
        txt = re.sub(r"\\textbf\{([^}]*)\}", r"\1", txt)
        for m in re.finditer(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)(?![\w.])", txt):
            out.append(round(float(m.group(1)), 4))
    return out

# ----------------------------------------------------------------- parsers
STAR = re.compile(r"\^\s*\{?\s*\\?\*\s*\}?")      # the star sits inside the maths, not after it
CI = re.compile(r"([+-]?\d+\.\d+)\s*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]")
def _ci(cell):
    sig = bool(STAR.search(cell))
    t = STAR.sub("", cell).replace("\\,", "").replace("$", "")
    m = CI.search(t)
    if not m: return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), sig
def _panels(label, ncell):
    """Rows grouped by the \\multicolumn dataset headings that separate the panels."""
    out, cur = [], None
    for line in table_body(label).split("\\\\"):
        if "multicolumn" in line:
            name = clean(line).strip()
            name = re.sub(r"^\d+\s*[lcr]\s*", "", name)
            cur = (name, []); out.append(cur); continue
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < ncell: continue
        if cur is None: cur = ("", []); out.append(cur)
        cur[1].append(cells)
    return [(n, r) for n, r in out if r]

def prim_full():
    out = []
    for name, rows in _panels("tab:primary-components", 5):
        rr = []
        for c in rows:
            cis = [_ci(c[i]) for i in (1, 2, 3)]
            if any(v is None for v in cis): continue
            p = clean(c[4]).replace("$", "").strip()
            rr.append((clean(c[0]), cis, p))
        if rr: out.append((name, rr))
    return out
def levels_full():
    out = []
    for name, rows in _panels("tab:ladder-levels", 7):
        rr = []
        for c in rows:
            try: vals = [float(clean(c[i])) for i in range(2, 7)]
            except Exception: continue
            rr.append((clean(c[0]), clean(c[1]), vals))
        if rr: out.append((name, rr))
    return out
def steps_full():
    out = []
    for name, rows in _panels("tab:ladder-steps", 7):
        rr = []
        for c in rows:
            cis = [_ci(c[i]) for i in range(2, 7)]
            if any(v is None for v in cis): continue
            rr.append((clean(c[0]), clean(c[1]), cis))
        if rr: out.append((name, rr))
    return out
def gammaq_full():
    rows = []
    for line in table_body("tab:gamma-sweep").split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line: continue
        c = [x.strip() for x in line.split("&")]
        if len(c) < 6: continue
        try: g = float(clean(c[0]))
        except Exception: continue
        cis = [_ci(c[i]) for i in range(1, 6)]
        if any(v is None for v in cis): continue
        rows.append((g, cis))
    return rows

def gammap_full():
    """Second block of tab:gamma-sweep: Holm-corrected p per metric."""
    rows, seen_head = [], False
    for line in table_body("tab:gamma-sweep").split("\\\\"):
        if "multicolumn" in line and "Holm" in line: seen_head = True; continue
        if not seen_head: continue
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line: continue
        c = [clean(x) for x in line.split("&")]
        try: rows.append((float(c[0]), [float(x) for x in c[1:6]]))
        except Exception: pass
    return rows

def _forest(ax, y, pts, color, lw=1.4, ms=7):
    """One forest row: point with its 95% interval. Every number is recorded for the coverage check."""
    v, lo, hi, sig = pts
    rec(v, lo, hi)
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.plot([v], [y], marker="o", ms=ms, color=color if sig else SURF, markeredgecolor=color,
            markeredgewidth=1.8, zorder=4)

# ===================================================== Table 5: all three metrics + p
def fig_primary_full():
    panels = prim_full()
    nrow = sum(len(r) for _n, r in panels)
    ys, labels, blocks = [], [], []
    y = 0.0
    for name, rows in panels:
        blocks.append((name, y))
        for lab, _c, _p in rows:
            ys.append(y); labels.append(lab); y += 1.0
        y += 1.2
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 0.30 * nrow + 1.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.28], wspace=0.08))
    names = ["$\\Delta$ROUGE-1", "$\\Delta$METEOR", "$\\Delta$BERTScore"]
    k = 0
    for name, rows in panels:
        for lab, cis, p in rows:
            yy = ys[k]
            for j in range(3):
                _forest(axes[j], yy, cis[j], MCOL[j])
                axes[j].annotate(f"{cis[j][0]:+.2f}", (cis[j][2], yy), textcoords="offset points",
                                 xytext=(6, 0), va="center", fontsize=6.6, color=INK,
                                 bbox=dict(facecolor=SURF, edgecolor="none", pad=1.0))
            pv = p.replace("<", "").strip()
            rec(float(pv))
            axes[3].text(0.5, yy, ("<" if "<" in p else "") + pv, ha="center", va="center",
                         fontsize=7.0, color=INK if (("<" in p) or float(pv) < 0.05) else MUTED)
            k += 1
    for j in range(3):
        axes[j].axvline(0, color=INK2, lw=0.9, zorder=2)
        axes[j].set_title(names[j], fontsize=9, loc="left", pad=6, color=INK)
        axes[j].set_xlabel("points")
        axes[j].grid(axis="y", visible=False); axes[j].set_axisbelow(True)
    axes[3].set_title("$p$ (R-1)", fontsize=9, loc="center", pad=6, color=INK)
    axes[3].set_xticks([]); axes[3].grid(False)
    axes[3].tick_params(axis="y", length=0)
    for sp in axes[3].spines.values(): sp.set_visible(False)
    for j in range(3):                      # headroom on the right for the value labels
        lo, hi = axes[j].get_xlim(); axes[j].set_xlim(lo, hi + 0.18 * (hi - lo))
    axes[0].set_yticks(ys); axes[0].set_yticklabels(labels, fontsize=7.4)
    axes[0].set_ylim(max(ys) + 0.8, -1.8)
    for name, y0 in blocks:
        axes[0].text(0, y0 - 0.85, name, transform=axes[0].get_yaxis_transform(),
                     ha="left", va="center", fontsize=8.6, color=INK, style="italic",
                     clip_on=False)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=7, color=S1, label="interval excludes zero"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=SURF, markeredgecolor=S1,
                    markeredgewidth=1.8, label="interval includes zero")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=7.8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Primary component contrasts, every metric in Table 5: difference with its 95% "
                 "bootstrap interval, and the Holm-corrected $p$ for ROUGE-1",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tab5_primary_components_full")

# ===================================================== Table 6: cumulative levels, five metrics
def fig_levels_full():
    panels = levels_full()
    fig, axes = plt.subplots(len(panels), 5, figsize=(14.6, 3.5 * len(panels)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, (name, rows) in enumerate(panels):
        y = np.arange(len(rows))[::-1]
        for j in range(5):
            ax = axes[i][j]
            vals = [r[2][j] for r in rows]
            for k, v in enumerate(vals):
                rec(v)
                ax.plot([min(vals) - 0.08 * (max(vals) - min(vals) + 1e-9), v], [y[k], y[k]],
                        color=MCOL[j], lw=2.6, alpha=0.30, solid_capstyle="butt", zorder=2)
                ax.plot([v], [y[k]], marker="o", ms=7, color=MCOL[j], markeredgecolor=SURF,
                        markeredgewidth=1.0, zorder=3)
                ax.annotate(f"{v:.2f}", (v, y[k]), textcoords="offset points", xytext=(8, 0),
                            va="center", fontsize=6.8, color=INK)
            lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.42 + 0.05
            ax.set_xlim(lo - pad * 0.45, hi + pad)
            ax.set_title(METRIC5[j] if i == 0 else "", fontsize=9, loc="left", pad=6, color=INK)
            ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([(f"{r[0]}. {r[1]}" if r[0].strip() else r[1]) for r in rows],
                                   fontsize=7.2)
                ax.set_ylabel(name, fontsize=9.5)
            if i == len(panels) - 1: ax.set_xlabel("score", fontsize=8)
    fig.suptitle("Cumulative ablation ladder, every metric in Table 6: the score reached at each rung, "
                 "on both datasets", fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tab6_ladder_levels_full")

# ===================================================== Table 7: per-rung steps, five metrics
def fig_steps_full():
    panels = steps_full()
    fig, axes = plt.subplots(len(panels), 5, figsize=(15.0, 3.7 * len(panels)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, (name, rows) in enumerate(panels):
        y = np.arange(len(rows))[::-1]
        for j in range(5):
            ax = axes[i][j]
            for k, r in enumerate(rows):
                _forest(ax, y[k], r[2][j], MCOL[j], ms=6.5)
                ax.annotate(f"{r[2][j][0]:+.2f}", (r[2][j][2], y[k]), textcoords="offset points",
                            xytext=(6, 0), va="center", fontsize=6.6, color=INK,
                            bbox=dict(facecolor=SURF, edgecolor="none", pad=1.0))
            ax.axvline(0, color=INK2, lw=0.9, zorder=2)
            bnd = [b for r in rows for b in r[2][j][1:3]]
            pad = (max(bnd) - min(bnd)) * 0.34
            ax.set_xlim(min(bnd) - pad * 0.5, max(bnd) + pad)
            ax.set_title(METRIC5[j] if i == 0 else "", fontsize=9, loc="left", pad=6, color=INK)
            ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([(f"{r[0]}. {r[1]}" if r[0].strip() else r[1]) for r in rows],
                                   fontsize=7.2)
                ax.set_ylabel(name, fontsize=9.5)
            if i == len(panels) - 1: ax.set_xlabel("points", fontsize=8)
    fig.suptitle("Per-rung contribution, every metric in Table 7: each rung against the one above it, "
                 "with its 95% bootstrap interval. A filled marker means the interval excludes zero.",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tab7_ladder_steps_full")

def retention_full():
    rows = []
    for line in table_body("tab:gamma-retention").split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line: continue
        c = [clean(x) for x in line.split("&")]
        try: rows.append((float(c[0]), float(c[1]), float(c[2]), float(c[3])))
        except Exception: pass
    return rows

CFCELL = re.compile(r"([+-]?\d+\.\d+)\s*\(\s*(\d+)\s*/\s*(\d+)\s*;\s*p\s*([<=])\s*([\d.]+)\s*\)")
def cf_full(label):
    cols, rows = [], []
    for line in table_body(label).split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" not in line: continue
        c = [x.strip() for x in line.split("&")]
        if "Backbone" in clean(c[0]):
            cols = [clean(x).replace("\\", "") for x in c[2:]]; continue
        cells = []
        for x in c[2:]:
            m = CFCELL.search(clean(x))
            cells.append(None if not m else (float(m.group(1)), int(m.group(2)), int(m.group(3)),
                                             m.group(4), float(m.group(5))))
        if cells and all(v is not None for v in cells):
            rows.append((BB_SHORT.get(clean(c[0]), clean(c[0])), clean(c[1]), cells))
    return cols, rows

# ===================================================== D.1 + D.2: five metrics, Holm p, retention
def fig_gamma_full():
    q, pr, ret = gammaq_full(), gammap_full(), retention_full()
    pmap = dict(pr)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2),
                             gridspec_kw=dict(hspace=0.62, wspace=0.26))
    axes = axes.ravel()
    gs = [r[0] for r in q]
    y = np.arange(len(q))[::-1]
    for j in range(5):
        ax = axes[j]
        for k, (g, cis) in enumerate(q):
            _forest(ax, y[k], cis[j], MCOL[j], ms=6.5)
            pv = pmap.get(g, [None] * 5)[j]
            rec(pv)
            ax.annotate(f"{cis[j][0]:+.2f}   $p$={pv:.4f}", (cis[j][2], y[k]),
                        textcoords="offset points", xytext=(7, 0), va="center", fontsize=6.9,
                        color=INK if pv is not None and pv < 0.05 else INK2,
                        bbox=dict(facecolor=SURF, edgecolor="none", pad=1.0))
        ax.axvline(0, color=INK2, lw=0.9, zorder=2)
        bnd = [b for _g, cis in q for b in cis[j][1:3]]
        ax.set_xlim(min(bnd) - 0.25 * (max(bnd) - min(bnd)), max(bnd) + 1.45 * (max(bnd) - min(bnd)))
        ax.set_yticks(y); ax.set_yticklabels([f"$\\gamma$={g:g}" for g in gs], fontsize=7.6)
        ax.set_title(f"$\\Delta${METRIC5[j]}", fontsize=9, loc="left", pad=6, color=INK)
        ax.set_xlabel("points", fontsize=8)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    ax = axes[5]
    base = [r[1] for r in ret]; with_ = [r[2] for r in ret]; chg = [r[3] for r in ret]
    rec(*base, *with_, *chg)
    yy = np.arange(len(ret))[::-1]
    for k in range(len(ret)):
        ax.plot([base[k], with_[k]], [yy[k], yy[k]], color=MUTED, lw=1.5, zorder=2)
        ax.plot([base[k]], [yy[k]], marker="|", ms=11, mew=2.0, color=INK2, zorder=3)
        ax.plot([with_[k]], [yy[k]], marker="o", ms=7, color=S3, markeredgecolor=SURF,
                markeredgewidth=1.0, zorder=4)
        ax.annotate(f"{with_[k]:.2f}  ({chg[k]:+.2f})", (with_[k], yy[k]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=6.9, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([f"$\\gamma$={r[0]:g}" for r in ret], fontsize=7.6)
    ax.set_xlim(min(base) - 0.4, max(with_) + 2.6)
    ax.set_title("Question-entity retention (%)", fontsize=9, loc="left", pad=6, color=INK)
    ax.set_xlabel(f"without the bonus = {base[0]:.2f}% (tick); with it (dot), change in brackets",
                  fontsize=7.4)
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    fig.suptitle("The entity bonus, every metric in Tables D.1 and D.2: each quality metric with its "
                 "95% interval and Holm-corrected $p$, and the retention the bonus exists to raise. "
                 "A filled marker means the interval excludes zero.",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tabD1_D2_gamma_sweep_full")

# ===================================================== Tables 10 and 13: delta, counts and exact p
def _cf_grid(label, title, fname, width):
    cols, rows = cf_full(label)
    M = np.array([[c[0] for c in r[2]] for r in rows])
    lim = max(abs(M.min()), abs(M.max()))
    fig, ax = plt.subplots(figsize=(width, 0.52 * len(rows) + 2.3))
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                   aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(r[2]):
            d, a, b, rel, pv = c
            rec(d, a, b, pv)
            dark = abs(d) > lim * 0.55
            col = "#ffffff" if dark else INK
            ax.text(j, i - 0.21, f"{d:+.1f}", ha="center", va="center", fontsize=9.2, color=col)
            ax.text(j, i + 0.16, f"{a}/{b}", ha="center", va="center", fontsize=7.0, color=col)
            ax.text(j, i + 0.37, f"$p${rel}{pv:g}", ha="center", va="center", fontsize=7.0,
                    color=col if pv >= 0.05 else col, style="italic" if pv < 0.05 else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(" $-$ ", "\n$-$ ") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} · {r[1]}" for r in rows], fontsize=8)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("$\\Delta$ accuracy (points)", fontsize=8); cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=9.5, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, fname)
def fig_cf_matched_full():
    _cf_grid("tab:cf-matched-paired",
             "Matched closed-form controls, every number in Table 10: accuracy difference, the "
             "discordant counts $a/b$ and the exact McNemar $p$ on identical items. An italic $p$ "
             "is below 0.05.", "tab10_closedform_matched_full", 9.6)
def fig_cf_fullmodel_full():
    _cf_grid("tab:cf-full-paired",
             "Complete framework on task-adapted generators, every number in Table 13: accuracy "
             "difference, the discordant counts $a/b$ and the exact McNemar $p$ on identical items. "
             "An italic $p$ is below 0.05.", "tab13_closedform_full_full", 13.4)

# ----------------------------------------------------------------- remaining parsers
def _lines(label):
    for line in table_body(label).split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom|c)rule(\{[^}]*\})?", " ", line)
        if "&" in line: yield line
def seedsc_full():
    out = []
    for name, rows in _panels("tab:seeds", 6):
        rr = []
        for c in rows:
            ms = []
            for x in c[1:6]:
                m = re.search(r"([\d.]+)\s*\\pm\s*([\d.]+)", x.replace("$", ""))
                if m: ms.append((float(m.group(1)), float(m.group(2))))
            if len(ms) == 5: rr.append((clean(c[0]), ms))
        if rr: out.append((name, rr))
    return out
def openseeds_full():
    rows = []
    for line in _lines("tab:open-seeds"):
        c = [clean(x) for x in line.split("&")]
        if len(c) < 7 or c[0].lower().startswith("backbone"): continue
        try:
            n = int(c[2])
            r3 = [float(x) for x in c[3].split("/")]
            rm, rs = [float(x) for x in c[4].split("±")]
            e3 = [float(x) for x in c[5].split("/")]
            em, es = [float(x) for x in c[6].split("±")]
        except Exception: continue
        rows.append((BB_SHORT.get(c[0], c[0]).replace("-Distill-Llama", ""), c[1], n, r3, (rm, rs),
                     e3, (em, es)))
    return rows
def topic_full():
    rows = []
    for line in _lines("tab:topic"):
        c = [clean(x) for x in line.split("&")]
        if len(c) < 5: continue
        ci = _ci(c[4])
        try: n, a, b = int(c[1]), float(c[2]), float(c[3])
        except Exception: continue
        if ci: rows.append((c[0], n, a, b, ci))
    return rows
def weight_full():
    rows = []
    for line in _lines("tab:cf-weight-sweep"):
        c = [clean(x) for x in line.split("&")]
        if len(c) < 12 or c[0].lower().startswith("backbone"): continue
        try:
            n = int(c[2]); vals = [float(x) for x in c[3:10]]
            best = float(c[10]); ex = float(c[11])
        except Exception: continue
        rows.append((BB_SHORT.get(c[0], c[0]).replace("-Distill-Llama", ""), c[1], n, vals, best, ex))
    return rows
def retrainx_full():
    rows = []
    for line in _lines("tab:verifier-retrain"):
        c = [clean(x) for x in line.split("&")]
        if len(c) < 8 or c[0].lower().startswith("backbone"): continue
        m = re.search(r"([\d.]+)\s*\(\s*w\s*=\s*([\d.]+)\s*\)", c[7])
        try:
            n = int(c[2]); nov, tr, own, d = (float(c[3]), float(c[4]), float(c[5]), float(c[6]))
            bv, bw = float(m.group(1)), float(m.group(2))
        except Exception: continue
        rows.append((BB_SHORT.get(c[0], c[0]).replace("-Distill-Llama", ""), c[1], n, nov, tr, own,
                     d, bv, bw))
    return rows

# ===================================================== Table 3: seed means, five metrics
def fig_seeds_full():
    panels = seedsc_full()
    fig, axes = plt.subplots(len(panels), 5, figsize=(14.6, 2.9 * len(panels)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, (name, rows) in enumerate(panels):
        y = np.arange(len(rows))[::-1]
        for j in range(5):
            ax = axes[i][j]
            ms = [r[1][j] for r in rows]
            for k, (m, sd) in enumerate(ms):
                rec(m, sd)
                # a dot plot, not bars: these axes do not start at zero, and a bar on a truncated
                # axis misstates the ratio between rows
                ax.errorbar(m, y[k], xerr=sd, fmt="o", ms=6, color=MCOL[j], ecolor=MCOL[j],
                            elinewidth=1.4, capsize=3, markeredgecolor=SURF, markeredgewidth=1.0,
                            zorder=4)
                ax.annotate(f"{m:.2f}±{sd:.2f}", (m + sd, y[k]), textcoords="offset points",
                            xytext=(7, 0), va="center", fontsize=6.8, color=INK)
            lo = min(m - sd for m, sd in ms); hi = max(m + sd for m, sd in ms)
            ax.set_xlim(lo - (hi - lo) * 0.55 - 0.1, hi + (hi - lo) * 1.5 + 0.4)
            ax.set_title(METRIC5[j] if i == 0 else "", fontsize=9, loc="left", pad=6, color=INK)
            ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
                ax.set_ylabel(name, fontsize=9.5)
            if i == len(panels) - 1: ax.set_xlabel("score (mean $\\pm$ SD over three seeds)",
                                                   fontsize=7.6)
    fig.suptitle("Three-seed means, every metric in Table 3: mean with its standard deviation over "
                 "the three random seeds", fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tab3_seeds_full")

# ===================================================== Table 4: open-ended per-seed detail
def fig_openseeds_full():
    rows = openseeds_full()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 0.52 * len(rows) + 1.9), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    y = np.arange(len(rows))[::-1]
    for ax, col, ttl, unit in ((axes[0], 3, "ROUGE-1 per seed", "ROUGE-1"),
                               (axes[1], 5, "Retrieval effect per seed", "points")):
        for k, r in enumerate(rows):
            vals = r[col]; m, sd = r[col + 1]
            rec(*vals); rec(m, sd)
            if ax is axes[0]: rec(r[2])
            ax.plot([m - sd, m + sd], [y[k], y[k]], color=MUTED, lw=4.5, alpha=0.35,
                    solid_capstyle="round", zorder=2)
            for jj, v in enumerate(vals):
                ax.plot([v], [y[k]], marker=["o", "s", "^"][jj], ms=7.5, color=[S1, S2, S3][jj],
                        markeredgecolor=SURF, markeredgewidth=1.0, zorder=4)
            ax.plot([m], [y[k]], marker="|", ms=15, mew=2.2, color=INK, zorder=5)
            ax.annotate("  ".join(f"{v:+.2f}" if col == 5 else f"{v:.2f}" for v in vals)
                        + f"   mean {m:+.2f}±{sd:.2f}" if col == 5 else
                        "  ".join(f"{v:.2f}" for v in vals) + f"   mean {m:.2f}±{sd:.2f}",
                        (max(max(vals), m + sd), y[k]), textcoords="offset points", xytext=(10, 0),
                        va="center", fontsize=7.0, color=INK)
        if col == 5: ax.axvline(0, color=INK2, lw=0.9, zorder=3)
        allv = [v for r in rows for v in r[col]] + [r[col + 1][0] + r[col + 1][1] for r in rows]
        ax.set_xlim(min(allv) - 1.0, max(allv) + (max(allv) - min(allv)) * 1.15 + 1.0)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([f"{r[0]}\n{r[1]} ($n$ seeds = {r[2]})" for r in rows], fontsize=7.6)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(ttl, fontsize=9.5, loc="left", pad=6, color=INK)
        ax.set_xlabel(unit, fontsize=8)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    h = [plt.Line2D([], [], marker=m, ls="none", ms=7, color=c) for m, c in
         zip(["o", "s", "^"], [S1, S2, S3])]
    fig.legend(handles=[tuple(h), plt.Line2D([], [], marker="|", ls="none", ms=12, mew=2.2,
                                             color=INK)],
               labels=["the three random seeds, run separately", "mean, with $\\pm$1 SD shaded"],
               handler_map={tuple: __import__("matplotlib").legend_handler.HandlerTuple(
                   ndivide=None, pad=0.7)},
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("Open-ended seed detail, every number in Table 4: each seed's ROUGE-1 and each "
                 "seed's retrieval effect, with the mean and SD over the three",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tab4_open_seeds_full")

# ===================================================== Table 19: topic-overlap strata
def fig_topic_full():
    rows = topic_full()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(rows))[::-1]
    ax = axes[0]
    for k, (name, n, a, b, ci) in enumerate(rows):
        rec(n, a, b)
        ax.plot([a, b], [y[k], y[k]], color=MUTED, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([a], [y[k]], marker="o", ms=8, color="#9ec5f4", markeredgecolor=SURF,
                markeredgewidth=1.0, zorder=3)
        ax.plot([b], [y[k]], marker="o", ms=8, color=S1, markeredgecolor=SURF, markeredgewidth=1.0,
                zorder=4)
        ax.annotate(f"{a:.2f} → {b:.2f}", (max(a, b), y[k]), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=7.4, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n$n$={r[1]}" for r in rows], fontsize=7.8)
    allv = [v for r in rows for v in (r[2], r[3])]
    ax.set_xlim(min(allv) - 0.6, max(allv) + 3.4)
    ax.set_title("ROUGE-1 without retrieval → with retrieval", fontsize=9.5, loc="left", pad=6,
                 color=INK)
    ax.set_xlabel("ROUGE-1", fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=7, color="#9ec5f4", label="no retrieval"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=S1, label="with retrieval")]
    ax.legend(handles=h, loc="upper center", ncol=2, fontsize=7.4, frameon=False,
              bbox_to_anchor=(0.5, -0.17))
    ax = axes[1]
    for k, (name, n, a, b, ci) in enumerate(rows):
        # Table 19 carries no significance column, so the fill is derived: filled where the
        # interval excludes zero.
        _forest(ax, y[k], (ci[0], ci[1], ci[2], ci[1] > 0 or ci[2] < 0), S1, ms=8)
        ax.annotate(f"{ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}]", (ci[2], y[k]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=7.2, color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    bnd = [v for r in rows for v in r[4][1:3]]
    ax.set_xlim(min(bnd) - 0.8, max(bnd) + (max(bnd) - min(bnd)) * 0.95)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_title("Retrieval effect, with its 95% interval", fontsize=9.5, loc="left", pad=6,
                 color=INK)
    ax.set_xlabel("points", fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    fig.suptitle("Topic-overlap strata, every number in Table 19: the two scores, the stratum size "
                 "and the effect with its interval. A filled marker means the interval excludes zero.",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.03)
    save(fig, "tab19_topic_strata_full")

# ===================================================== D.4: fusion-weight sweep, every column
def fig_weight_full():
    rows = weight_full()
    W = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    bbs = sorted({r[0] for r in rows}, key=lambda b: [r[0] for r in rows].index(b))
    bks = sorted({r[1] for r in rows}, key=lambda b: [r[1] for r in rows].index(b))
    fig, axes = plt.subplots(len(bbs), len(bks), figsize=(14.6, 2.7 * len(bbs)), sharex=True)
    axes = np.atleast_2d(axes)
    idx = {(r[0], r[1]): r for r in rows}
    for i, bb in enumerate(bbs):
        for j, bk in enumerate(bks):
            ax = axes[i][j]; r = idx.get((bb, bk))
            if r is None: ax.axis("off"); continue
            _b, _k, n, vals, best, ex = r
            rec(n, best, ex); rec(*vals)
            ax.plot(W, vals, color=S1, lw=1.8, zorder=3)
            for w, v in zip(W, vals):
                ax.plot([w], [v], marker="o", ms=5, color=S1 if w != best else POS,
                        markeredgecolor=SURF, markeredgewidth=0.9, zorder=4)
                ax.annotate(f"{v:.1f}", (w, v), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=6.2, color=INK)
            ax.plot([best], [vals[W.index(best)]], marker="*", ms=15, color=POS,
                    markeredgecolor=SURF, markeredgewidth=0.8, zorder=5)
            ax.axvline(0.35, color=NEG, lw=1.2, ls=(0, (3, 2)), zorder=2)
            lo, hi = min(vals), max(vals)
            ax.set_ylim(lo - (hi - lo) * 0.26 - 1.2, hi + (hi - lo) * 0.32 + 2)
            ax.set_title(f"{bb} · {bk}", fontsize=8.6, loc="left", pad=5, color=INK)
            ax.annotate(f"$n$={n}   best $w$={best:g}   executed $-$ none = {ex:+.1f}",
                        (0.015, 0.035), xycoords="axes fraction", fontsize=7.0, color=INK2,
                        bbox=dict(facecolor=SURF, edgecolor="none", pad=1.6))
            ax.grid(axis="x", visible=False); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
            if j == 0: ax.set_ylabel("accuracy (%)", fontsize=8)
            if i == len(bbs) - 1: ax.set_xlabel("fusion weight $w$", fontsize=8)
    h = [plt.Line2D([], [], marker="*", ls="none", ms=12, color=POS, label="best weight for this cell"),
         plt.Line2D([], [], ls=(0, (3, 2)), color=NEG, lw=1.2, label="executed $w$=0.35")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Fusion-weight sweep, every number in Table D.4: the seven accuracies per cell, the "
                 "item count, the best weight and what the executed weight costs against no verifier",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tabD4_fusion_weight_sweep_full")

# ===================================================== D.5: verifier retraining, every column
def fig_retrain_full():
    rows = retrainx_full()
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(12.6, 0.56 * len(rows) + 2.4),
                                 gridspec_kw=dict(width_ratios=[1, 1.22], wspace=0.02), sharey=True)
    y = np.arange(len(rows))[::-1]
    for k, (bb, bk, n, nov, tr, own, d, bv, bw) in enumerate(rows):
        rec(n, nov, tr, own, d, bv, bw)
        ax.plot([min(nov, tr, own, bv), max(nov, tr, own, bv)], [y[k], y[k]], color="#e3e2de",
                lw=5.0, solid_capstyle="round", zorder=1)
        ax.plot([nov], [y[k]], marker="|", ms=14, mew=2.2, color=INK2, zorder=4)
        ax.plot([tr], [y[k]], marker="o", ms=8, color=S2, markeredgecolor=SURF, markeredgewidth=1.0,
                zorder=3)
        ax.plot([own], [y[k]], marker="o", ms=8, color=S1, markeredgecolor=SURF, markeredgewidth=1.0,
                zorder=3)
        ax.plot([bv], [y[k]], marker="D", ms=6.5, color=S3, markeredgecolor=SURF,
                markeredgewidth=1.0, zorder=3)
        tx.text(0.015, y[k], f"none {nov:.1f}    transf. {tr:.1f}    own {own:.1f}  ({d:+.1f})    "
                             f"best {bv:.1f} at $w$={bw:g}    $n$={n}",
                transform=tx.get_yaxis_transform(), va="center", fontsize=7.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f"{r[0]} · {r[1]}" for r in rows], fontsize=8)
    allv = [v for r in rows for v in (r[3], r[4], r[5], r[7])]
    ax.set_xlim(min(allv) - 2.5, max(allv) + 2.5)
    ax.set_xlabel("closed-form accuracy (%)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    tx.set_xticks([]); tx.set_xlim(0, 1); tx.grid(False)
    tx.tick_params(axis="y", length=0)
    for sp in tx.spines.values(): sp.set_visible(False)
    h = [plt.Line2D([], [], marker="|", ls="none", ms=12, mew=2.2, color=INK2, label="no verifier"),
         plt.Line2D([], [], marker="o", ls="none", ms=8, color=S2, label="transferred @0.35"),
         plt.Line2D([], [], marker="o", ls="none", ms=8, color=S1, label="own benchmark @0.35"),
         plt.Line2D([], [], marker="D", ls="none", ms=7, color=S3, label="own at its best $w$")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.8, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Verifier retraining, every number in Table D.5: the three operating points, the "
                 "own-minus-transferred difference, the best weight reachable and the item count",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tabD5_verifier_retrain_full")

FULL = [("tab5_primary_components_full", fig_primary_full, [("tab:primary-components", 1)]),
        ("tab6_ladder_levels_full", fig_levels_full, [("tab:ladder-levels", 2)]),
        ("tab7_ladder_steps_full", fig_steps_full, [("tab:ladder-steps", 2)]),
        ("tab3_seeds_full", fig_seeds_full, [("tab:seeds", 1)]),
        ("tab4_open_seeds_full", fig_openseeds_full, [("tab:open-seeds", 2)]),
        ("tab19_topic_strata_full", fig_topic_full, [("tab:topic", 1)]),
        ("tab10_closedform_matched_full", fig_cf_matched_full, [("tab:cf-matched-paired", 2)]),
        ("tab13_closedform_full_full", fig_cf_fullmodel_full, [("tab:cf-full-paired", 2)]),
        ("tabD1_D2_gamma_sweep_full", fig_gamma_full, [("tab:gamma-sweep", 1),
                                                       ("tab:gamma-retention", 1)]),
        ("tabD4_fusion_weight_sweep_full", fig_weight_full, [("tab:cf-weight-sweep", 2)]),
        ("tabD5_verifier_retrain_full", fig_retrain_full, [("tab:verifier-retrain", 2)])]

def verify_full():
    """Prove coverage: every number printed in each source table must be drawn by its figure."""
    from collections import Counter
    tot_t = tot_m = 0
    print("  figure                                  table values   drawn   missing")
    for name, fn, srcs in FULL:
        rec_reset()
        try:
            fn()
        except MissingTable as e:
            print(f"  {name:38s} skipped | no table {e.args[0]}")
            continue
        drawn = Counter(REC)
        want = Counter()
        for lab, sk in srcs: want.update(table_numbers(lab, sk))
        miss = want - drawn
        tot_t += sum(want.values()); tot_m += sum(miss.values())
        print(f"  {name:38s} {sum(want.values()):6d} {len(REC):8d} {sum(miss.values()):9d}"
              + ("" if not miss else "   " + str(sorted(miss.elements())[:8])))
    print(f"\n  {tot_t} table values across {len(FULL)} figures; {tot_m} not drawn")
    return tot_m
def latex_snippet_full():
    """Include file for the full-coverage set: every number of the source table is in the figure."""
    CAP = {
 "tab5_primary_components_full": ("Primary component contrasts, every column of "
   "Table~\\ref{tab:primary-components}: $\\Delta$ROUGE-1, $\\Delta$METEOR and $\\Delta$BERTScore, each "
   "with its 95\\% bootstrap interval, and the Holm-corrected $p$. A filled marker means the "
   "interval excludes zero."),
 "tab6_ladder_levels_full": ("Cumulative ablation ladder, every column of "
   "Table~\\ref{tab:ladder-levels}: the level reached at each rung on all five metrics, on both "
   "datasets."),
 "tab7_ladder_steps_full": ("Per-rung contribution, every column of "
   "Table~\\ref{tab:ladder-steps}: each rung against the one above it on all five metrics, with "
   "95\\% bootstrap intervals."),
 "tab3_seeds_full": ("Three-seed means, every column of Table~\\ref{tab:seeds}: mean and standard "
   "deviation over seeds for all five metrics."),
 "tab4_open_seeds_full": ("Open-ended seed detail, every column of Table~\\ref{tab:open-seeds}: "
   "each seed's ROUGE-1 and each seed's retrieval effect, with mean and SD."),
 "tab19_topic_strata_full": ("Topic-overlap strata, every column of Table~\\ref{tab:topic}: "
   "stratum size, the two scores and the retrieval effect with its interval."),
 "tab10_closedform_matched_full": ("Matched closed-form controls, every number of "
   "Table~\\ref{tab:cf-matched-paired}: accuracy difference, discordant counts $a/b$ and exact "
   "McNemar $p$ on identical items."),
 "tab13_closedform_full_full": ("Complete framework on task-adapted generators, every number of "
   "Table~\\ref{tab:cf-full-paired}: accuracy difference, discordant counts $a/b$ and exact "
   "McNemar $p$ on identical items."),
 "tabD1_D2_gamma_sweep_full": ("The entity bonus, every number of Tables~\\ref{tab:gamma-sweep} "
   "and \\ref{tab:gamma-retention}: five quality metrics with intervals and Holm-corrected $p$, "
   "and question-entity retention."),
 "tabD4_fusion_weight_sweep_full": ("Fusion-weight sweep, every column of "
   "Table~\\ref{tab:cf-weight-sweep}: the seven accuracies per cell, the item count, the best "
   "weight and the cost of the executed weight."),
 "tabD5_verifier_retrain_full": ("Verifier retraining, every column of "
   "Table~\\ref{tab:verifier-retrain}: the three operating points, the own-minus-transferred "
   "difference, the best weight reachable and the item count.")}
    out = ["% ---- full-coverage figures (scripts/analysis/make_table_figures_full_coverage.py) ----",
           "% Every number printed in the source table is drawn here; verify_full() checks this.",
           ""]
    for name, _fn, _s in FULL:
        if not os.path.exists(os.path.join(OUT, name + ".pdf")):
            continue
        out += [r"\begin{figure*}[!tbp]", r"\centering",
                r"\includegraphics[width=\linewidth]{figures/%s.pdf}" % name,
                r"\caption{%s}" % CAP[name],
                r"\label{fig:%s}" % name.replace("_", "-"),
                r"\end{figure*}", ""]
    import os as _os
    from make_table_figures import OUT
    p = _os.path.join(OUT, "figure_includes_full.tex")
    open(p, "w", encoding="utf-8").write("\n".join(out))
    print("  wrote figures/figure_includes_full.tex")
def build_full():
    """Draw every full-coverage figure whose source table is still in the
    manuscript; skip the rest with a note."""
    for name, fn, _s in FULL:
        rec_reset()
        try:
            fn()
        except MissingTable as e:
            print(f"  SKIP {name}: the manuscript has no table {e.args[0]}")
if __name__ == "__main__":
    build_full(); print(); verify_full(); print(); latex_snippet_full()

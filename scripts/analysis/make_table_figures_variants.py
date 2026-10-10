# -*- coding: utf-8 -*-
"""Alternative diagram types for the same tables, so the best form can be chosen by eye.

Every variant reuses the parsers, palette and save path of make_table_figures.py, so all versions of a
figure are drawn from the same parsed table values. Files are suffixed _v2 / _v3; the originals keep
their unsuffixed names.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table_figures import (MissingTable, rows_of, table_body, parse_ci, clean, save, _grouped, _paired_grid,
                           BB_SHORT, S1, S2, S3, INK, INK2, MUTED, SURF, POS, NEG, MID,
                           DIVERGING, SEQ, TEXT)

# --------------------------------------------------------------- shared getters
def primary_rows():
    panels, cur = [], None
    for kind, v in rows_of("tab:primary-components"):
        if kind == "__HEAD__":
            cur = (re.sub(r"^\d+l", "", clean(v)), []); panels.append(cur); continue
        if cur is None or not isinstance(v, list) or len(v) < 2: continue
        p = parse_ci(v[1])
        if p: cur[1].append((clean(v[0]), *p))
    return [(n, r) for n, r in panels if r]
def leakage_rows():
    out = []
    for kind, v in rows_of("tab:leakage-convergence"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 3: continue
        est = clean(v[0])
        if not est or est.lower().startswith("estimate"): continue
        m = re.search(r"([+-]?\d+\.\d+)", clean(v[2]))
        if m: out.append((est, clean(v[1]), float(m.group(1))))
    return out
def seed_rows():
    rows = []
    if "\\label{tab:open-seeds}" not in TEXT:
        raise MissingTable("tab:open-seeds")
    cap_i = TEXT.index("\\label{tab:open-seeds}")
    m = re.search(r"retrieval effects are\s*(.+?)\s*on MedQuAD and\s*(.+?)\s*on the iCliniq",
                  TEXT[max(0, cap_i - 1800):cap_i], re.S)
    if m:
        mq = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", m.group(1))]
        ic = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", m.group(2))]
        rows += [("LLaMA-3.1-8B", "MedQuAD", mq), ("LLaMA-3.1-8B", "iCliniq-style", ic)]
    for kind, v in rows_of("tab:open-seeds"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 6: continue
        bb = clean(v[0])
        if not bb or bb.lower().startswith("backbone"): continue
        per = [float(x) for x in re.findall(r"([+-]\d+\.\d+)", v[5])]
        if len(per) >= 2:
            rows.append((bb.replace("-Distill-Llama", "").replace("-Instruct", ""), clean(v[1]), per))
    return rows
def gamma_rows():
    eff, ret = [], []
    for kind, v in rows_of("tab:gamma-sweep"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 2: continue
        try: g = float(clean(v[0]))
        except Exception: continue
        p = parse_ci(v[1])
        if p: eff.append((g, *p))
    for kind, v in rows_of("tab:gamma-retention"):
        if kind != "__ROW__" or not isinstance(v, list) or len(v) < 4: continue
        try: ret.append((float(clean(v[0])), float(clean(v[1])), float(clean(v[2]))))
        except Exception: pass
    return eff, ret
def fusion_rows():
    body = table_body("tab:cf-weight-sweep"); weights, rows = None, []
    for line in body.split("\\\\"):
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if any("textbf{Backbone" in c for c in cells):
            weights = [float(clean(c).replace("w=", "")) for c in cells[3:10]]; continue
        if weights is None or len(cells) < 10: continue
        bb = clean(cells[0])
        if not bb or bb.lower().startswith("backbone"): continue
        try: vals = [float(clean(c)) for c in cells[3:10]]
        except Exception: continue
        rows.append((BB_SHORT.get(bb, bb), clean(cells[1]), vals))
    return weights, rows
def retrain_rows():
    out = []
    for line in table_body("tab:verifier-retrain").split("\\\\"):
        if "&" not in line: continue
        cells = [c.strip() for c in line.split("&")]
        if any("textbf{Backbone" in c for c in cells) or len(cells) < 6: continue
        bb = clean(cells[0])
        if not bb or bb.lower().startswith("backbone"): continue
        try: nov, tr, own = float(clean(cells[3])), float(clean(cells[4])), float(clean(cells[5]))
        except Exception: continue
        out.append((BB_SHORT.get(bb, bb), clean(cells[1]), nov, tr, own))
    return out

# ============================== Table 5: v2 sorted diverging bars, v3 dataset slope
def t5_v2():
    panels = primary_rows()
    fig, axes = plt.subplots(1, len(panels), figsize=(10.6, 4.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, rr) in zip(axes, panels):
        rr = sorted(rr, key=lambda r: r[1])
        y = np.arange(len(rr))
        for i, (lab, v, lo, hi, sig) in enumerate(rr):
            ax.barh(i, v, height=0.62, color=(POS if v >= 0 else NEG), alpha=0.95 if sig else 0.40,
                    edgecolor=(POS if v >= 0 else NEG), linewidth=1.0, zorder=2)
            ax.plot([lo, hi], [i, i], color=INK, lw=1.0, solid_capstyle="round", zorder=4)
            ax.annotate(f"{v:+.2f}" + ("*" if sig else ""), (max(hi, 0) if v >= 0 else min(lo, 0), i),
                        textcoords="offset points", xytext=(6 if v >= 0 else -6, 0),
                        ha="left" if v >= 0 else "right", va="center", fontsize=7.4, color=INK)
        ax.axvline(0, color=INK2, lw=0.9, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rr] if ax is axes[0] else [])
        ax.set_title(name, fontsize=9.5, loc="left", pad=6, color=INK)
        ax.set_xlabel("$\\Delta$ ROUGE-1 (points)")
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.set_xlim(-3.6, 5.6)
    h = [Rectangle((0, 0), 1, 1, color=POS, alpha=0.95, label="positive, significant"),
         Rectangle((0, 0), 1, 1, color=POS, alpha=0.40, label="positive, not significant"),
         Rectangle((0, 0), 1, 1, color=NEG, alpha=0.95, label="negative, significant"),
         Rectangle((0, 0), 1, 1, color=NEG, alpha=0.40, label="negative, not significant")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.55, -0.07))
    fig.suptitle("Component contrasts, sorted by effect size within each dataset",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab5_primary_components_v2")

def t5_v3():
    panels = primary_rows()
    if len(panels) < 2: return
    (n1, r1), (n2, r2) = panels[0], panels[1]
    d2 = {lab: (v, sig) for lab, v, lo, hi, sig in r2}
    pairs = [(lab, v, sig, *d2[lab]) for lab, v, lo, hi, sig in r1 if lab in d2]
    pairs.sort(key=lambda p: -(p[1] - p[3]))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    # label selectively: the rows that actually move. The null cluster is named once instead.
    notable = [p for p in pairs if max(abs(p[1]), abs(p[3])) >= 0.6 or abs(p[1] - p[3]) >= 1.0]
    nulls = [p for p in pairs if p not in notable]
    for (lab, a, sa, b, sb) in pairs:
        ax.plot([0, 1], [a, b], color="#d9d8d3", lw=1.3, zorder=1)
        ax.plot([0], [a], marker="o", ms=7, color=S1 if sa else SURF, markeredgecolor=S1,
                markeredgewidth=1.6, zorder=3)
        ax.plot([1], [b], marker="o", ms=7, color=S2 if sb else SURF, markeredgecolor=S2,
                markeredgewidth=1.6, zorder=3)
    used = []
    for (lab, a, sa, b, sb) in notable:
        yy = a
        while any(abs(yy - u) < 0.17 for u in used): yy += 0.17
        used.append(yy)
        ax.annotate(lab, (0, a), xytext=(-12, (yy - a) * 0 + 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.4, color=INK2)
    if nulls:
        ymid = float(np.mean([p[1] for p in nulls]))
        ax.annotate(f"{len(nulls)} selection and contrastive contrasts,\nall within $\\pm$0.6 points on both datasets",
                    (0, ymid), xytext=(-12, -4), textcoords="offset points", ha="right", va="center",
                    fontsize=7.4, color=MUTED)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(-0.75, 1.25); ax.set_xticks([0, 1]); ax.set_xticklabels([n1, n2], fontsize=9.5)
    ax.set_ylabel("$\\Delta$ ROUGE-1 (points)")
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=7, color=S1, label=f"{n1}, significant"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=S2, label=f"{n2}, significant"),
         plt.Line2D([], [], marker="o", ls="none", ms=7, color=SURF, markeredgecolor=INK2,
                    markeredgewidth=1.6, label="not significant")]
    ax.legend(handles=h, loc="upper center", ncol=3, fontsize=7.6, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("The same contrast on the two datasets: retrieval reverses sign, the null mechanisms do not move",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=0.98)
    save(fig, "tab5_primary_components_v3")

# ============================== Tables 6-7: v2 trajectory with CI band, v3 contribution bars
def t6_v2():
    lev = _grouped("tab:ladder-levels", 2); stp = _grouped("tab:ladder-steps", 2, parse=True)
    fig, axes = plt.subplots(1, len(lev), figsize=(10.0, 4.0))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, lev):
        L = [x[1] for x in lev[ds]]; x = np.arange(1, len(L) + 1)
        st = stp.get(ds, [])
        lo = [L[0]] + [L[i] + st[i - 1][2] - st[i - 1][1] for i in range(1, len(L))] if st else L
        hi = [L[0]] + [L[i] + st[i - 1][3] - st[i - 1][1] for i in range(1, len(L))] if st else L
        ax.fill_between(x, lo, hi, color=S1, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, L, color=S1, lw=2.2, marker="o", ms=5.5, markeredgecolor=SURF,
                markeredgewidth=0.9, zorder=3)
        b = int(np.argmax(L))
        ax.plot([x[b]], [L[b]], marker="o", ms=10, color=SURF, markeredgecolor=S1,
                markeredgewidth=2.2, zorder=4)
        ax.annotate(f"best: rung {x[b]}", (x[b], L[b]), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.6, color=INK)
        ax.annotate(f"{L[-1] - L[0]:+.2f} overall", (x[-1], L[-1]), textcoords="offset points",
                    xytext=(-4, -14), ha="right", fontsize=7.6, color=INK2)
        ax.set_xticks(x); ax.set_xlabel("rung"); ax.set_title(ds, fontsize=9.5, loc="left", pad=6)
        ax.set_ylabel("ROUGE-1 (%)" if ax is axes[0] else "")
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    fig.suptitle("Cumulative trajectory with the interval of each step; the band is the step uncertainty "
                 "carried onto the level", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab6_7_cumulative_ladder_v2")

def t6_v3():
    stp = _grouped("tab:ladder-steps", 2, parse=True)
    fig, axes = plt.subplots(1, len(stp), figsize=(10.4, 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    # the shared x range has to hold the widest interval in EITHER panel plus the value label that
    # hangs off its end; matplotlib's autoscale ignores annotations, which is what pushed the
    # exemplar-retrieval row off both panels
    bnds = [b for ds in stp for (_l, _v, lo, hi, _s) in stp[ds] for b in (lo, hi)]
    xlo, xhi = min(bnds), max(bnds)
    pad = 0.21 * (xhi - xlo)
    for ax, ds in zip(axes, stp):
        rr = stp[ds][::-1]
        y = np.arange(len(rr))
        for i, (lab, v, lo, hi, sig) in enumerate(rr):
            ax.barh(i, v, height=0.6, color=(POS if v >= 0 else NEG), alpha=0.95 if sig else 0.45,
                    zorder=2)
            ax.plot([lo, hi], [i, i], color=INK, lw=1.0, zorder=4)
            ax.annotate(f"{v:+.2f}", (hi if v >= 0 else lo, i), textcoords="offset points",
                        xytext=(6 if v >= 0 else -6, 0), ha="left" if v >= 0 else "right",
                        va="center", fontsize=7.4, color=INK)
        ax.axvline(0, color=INK2, lw=0.9, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rr] if ax is axes[0] else [], fontsize=7.8)
        ax.set_title(ds, fontsize=9.5, loc="left", pad=6)
        ax.set_xlabel("step contribution to ROUGE-1 (points)")
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    fig.suptitle("What each rung contributes on its own, ordered as the ladder is climbed "
                 "(bottom = first rung added)", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab6_7_cumulative_ladder_v3")

# ============================== Tables 19-22: v2 ordered bars by strictness, v3 slope
STRICT_ORDER = ["Random split, topic in pool", "Random split, all questions",
                "Random split, topic held out", "Source-disjoint retrieval", "Group-disjoint repartition"]
def t19_v2():
    rows = leakage_rows()
    rows.sort(key=lambda r: STRICT_ORDER.index(r[0]) if r[0] in STRICT_ORDER else 99)
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(rows)); vals = [r[2] for r in rows]
    strict = [i for i, r in enumerate(rows) if "disjoint" in r[0].lower() or "repartition" in r[0].lower()]
    cols = [S1 if i in strict else "#9ec5f4" for i in x]
    ax.bar(x, vals, width=0.6, color=cols, zorder=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap2(r[0]) for r in rows], fontsize=7.8)
    ax.set_ylabel("$\\Delta$ ROUGE-1 from retrieval (points)")
    top = max(vals)
    ax.set_ylim(min(0, min(vals)) - 0.4, top + 1.7)
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    # caption above the arrow, arrow above everything else, legend parked below both
    ax.annotate("", xy=(len(rows) - 0.6, top + 1.22), xytext=(0.6, top + 1.22),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.text((len(rows) - 1) / 2, top + 1.34, "stricter control of shared source and topic",
            ha="center", va="bottom", fontsize=7.8, color=MUTED)
    h = [Rectangle((0, 0), 1, 1, color=S1, label="shared material removed"),
         Rectangle((0, 0), 1, 1, color="#9ec5f4", label="random split")]
    ax.legend(handles=h, loc="upper left", fontsize=7.8, bbox_to_anchor=(0.0, 0.83))
    fig.suptitle("The retrieval effect does not fall as control tightens", fontsize=10, color=INK,
                 x=0.01, ha="left", y=1.02)
    save(fig, "tab19_22_leakage_convergence_v2")
def t19_v3():
    rows = leakage_rows()
    rows.sort(key=lambda r: STRICT_ORDER.index(r[0]) if r[0] in STRICT_ORDER else 99)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(rows)); vals = [r[2] for r in rows]
    ax.plot(x, vals, color=S1, lw=2.0, zorder=2)
    for i, (name, design, v) in enumerate(rows):
        strict = "disjoint" in name.lower() or "repartition" in name.lower()
        ax.plot([i], [v], marker="D" if strict else "o", ms=9 if strict else 8,
                color=S1 if strict else SURF, markeredgecolor=S1, markeredgewidth=1.8, zorder=4)
        ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([r[0].replace(", ", ",\n") for r in rows], fontsize=7.8)
    ax.set_ylabel("$\\Delta$ ROUGE-1 (points)")
    ax.set_xlabel("designs ordered by strictness of overlap control $\\rightarrow$")
    ax.set_ylim(min(vals) - 0.6, max(vals) + 0.9)
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="D", ls="none", ms=9, color=S1, label="shared material removed"),
         plt.Line2D([], [], marker="o", ls="none", ms=8, color=SURF, markeredgecolor=S1,
                    markeredgewidth=1.8, label="random split")]
    ax.legend(handles=h, loc="lower right", fontsize=7.8)
    fig.suptitle("Five estimates of one quantity, read left to right as control tightens",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab19_22_leakage_convergence_v3")

# ============================== Tables 3-4: v2 bar+points, v3 faceted strips
def t3_v2():
    rows = seed_rows()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(rows))
    means = [float(np.mean(r[2])) for r in rows]
    ax.bar(x, means, width=0.55, color=["#9ec5f4" if m > 0 else "#f3b2b2" for m in means], zorder=2)
    MK, MC = ["o", "s", "^"], [S1, S2, S3]
    OFF = [-0.17, 0.0, 0.17]                 # dodged so the three seed markers never sit on top of one another
    for i, (bb, ds, per) in enumerate(rows):
        for j, p in enumerate(per):
            ax.plot([i + OFF[j % 3]], [p], marker=MK[j % 3], ms=6.5, color=MC[j % 3],
                    markeredgecolor=SURF, markeredgewidth=0.9, zorder=4)
        # park the mean label clear of both the bar end and the highest/lowest seed marker
        up = means[i] > 0
        anchor = max([means[i]] + list(per)) if up else min([means[i]] + list(per))
        ax.annotate(f"{means[i]:+.2f}", (i, anchor), textcoords="offset points",
                    xytext=(0, 11 if up else -17), ha="center", fontsize=7.6, color=INK)
    ax.axhline(0, color=INK2, lw=0.9, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{d}" for b, d, _ in rows], fontsize=7.6)
    ax.set_ylabel("retrieval effect on ROUGE-1 (points)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.45, hi + 0.45)
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    # one entry carrying all three shapes: the seeds are unlabelled, so no shape claims an identity
    seeds = tuple(plt.Line2D([], [], marker=m, ls="none", ms=6.5, color=c,
                             markeredgecolor=SURF, markeredgewidth=0.9) for m, c in zip(MK, MC))
    ax.legend(handles=[seeds, Rectangle((0, 0), 1, 1, color="#9ec5f4")],
              labels=["the three random seeds, run separately", "mean over the three seeds"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
              loc="upper center", ncol=2, fontsize=7.6, bbox_to_anchor=(0.5, -0.20))
    fig.suptitle("Mean retrieval effect with the three random seeds overlaid",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab3_4_seed_stability_v2")
def t3_v3():
    rows = seed_rows()
    bbs = sorted({r[0] for r in rows}, key=lambda b: [r[0] for r in rows].index(b))
    fig, axes = plt.subplots(1, len(bbs), figsize=(9.6, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, bbs):
        sub = [r for r in rows if r[0] == bb]
        for i, (_b, ds, per) in enumerate(sub):
            jitter = np.linspace(-0.07, 0.07, len(per))
            for j, p in enumerate(per):
                ax.plot([i + jitter[j]], [p], marker=["o", "s", "^"][j % 3], ms=7,
                        color=[S1, S2, S3][j % 3], markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
            m = float(np.mean(per))
            ax.plot([i - 0.17, i + 0.17], [m, m], color=INK, lw=1.8, zorder=4)
        ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([s[1].replace("-style", "") for s in sub], fontsize=8)
        ax.set_title(bb, fontsize=9, loc="left", pad=6, color=INK)
        ax.set_ylabel("retrieval effect on ROUGE-1" if ax is axes[0] else "")
        ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker=m, ls="none", ms=7, color=c, label=f"seed {s}")
         for m, c, s in zip(["o", "s", "^"], [S1, S2, S3], ["42", "1", "2"])]
    h.append(plt.Line2D([], [], color=INK, lw=1.8, label="mean"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("One panel per backbone: within a cell the three seeds never straddle zero",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.03)
    save(fig, "tab3_4_seed_stability_v3")

# ============================== Tables 10/13: v2 grouped bars, v3 bubble matrix
def t10_v2():
    c1, d1 = _paired_grid("tab:cf-matched-paired")
    c2, d2 = _paired_grid("tab:cf-full-paired")
    for tag, cols, data, title in (("a", c1, d1, "Matched controls, untrained backbones (Table 10)"),
                                   ("b", c2, d2, "Complete framework, task-adapted generators (Table 13)")):
        bbs = list(data); benches = list(next(iter(data.values())))
        rows = [(b, k) for b in bbs for k in benches]
        n, m = len(rows), len(cols)
        fig, ax = plt.subplots(figsize=(9.0, 0.42 * n * m / 2 + 2.2))
        hgt = 0.8 / m
        for j in range(m):
            vals = [data[b][k][0][j] for b, k in rows]
            sig = [data[b][k][1][j] for b, k in rows]
            y = np.arange(n) + (j - (m - 1) / 2) * hgt
            ax.barh(y, vals, height=hgt * 0.92, color=[S1, S2, S3, "#eda100", "#4a3aa7"][j % 5],
                    alpha=0.95, zorder=2, label=cols[j].replace(" $-$ ", " − "))
            for yy, v, sg in zip(y, vals, sig):
                if sg:
                    ax.annotate("*", (v, yy), textcoords="offset points",
                                xytext=(4 if v >= 0 else -4, -3), ha="left" if v >= 0 else "right",
                                fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=0.9, zorder=3)
        ax.set_yticks(np.arange(n)); ax.set_yticklabels([f"{b} · {k}" for b, k in rows], fontsize=7.8)
        ax.set_xlabel("$\\Delta$ accuracy (points)")
        ax.set_title(title, fontsize=9.5, loc="left", pad=6, color=INK)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.legend(loc="upper center", ncol=min(m, 3), fontsize=7.4,
                  bbox_to_anchor=(0.5, -0.09 - 0.012 * n))
        fig.suptitle("* marks $p<0.05$ (exact McNemar)", fontsize=8.5, color=MUTED, x=0.01, ha="left", y=1.0)
        save(fig, f"tab10_13_closedform_paired_v2{tag}")
def t10_v3():
    c1, d1 = _paired_grid("tab:cf-matched-paired")
    c2, d2 = _paired_grid("tab:cf-full-paired")
    allv = [v for d in (d1, d2) for b in d for k in d[b] for v in d[b][k][0] if not np.isnan(v)]
    lim = max(abs(min(allv)), abs(max(allv)))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8),
                             gridspec_kw=dict(width_ratios=[len(c1), len(c2)], wspace=0.08))
    for ax, cols, data, title in ((axes[0], c1, d1, "Matched controls (Table 10)"),
                                  (axes[1], c2, d2, "Complete framework (Table 13)")):
        bbs = list(data); benches = list(next(iter(data.values())))
        rows = [(b, k) for b in bbs for k in benches]
        for i, (b, k) in enumerate(rows):
            for j, (v, sg) in enumerate(zip(data[b][k][0], data[b][k][1])):
                if np.isnan(v): continue
                r = 0.08 + 0.36 * (abs(v) / lim) ** 0.6
                ax.add_patch(Circle((j, i), r, facecolor=(POS if v > 0 else NEG), alpha=0.9,
                                    edgecolor=INK if sg else "none", linewidth=1.4, zorder=3))
                ax.annotate(f"{v:+.0f}" if abs(v) >= 1 else "", (j, i), ha="center", va="center",
                            fontsize=6.4, color="#ffffff", zorder=4)
        ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace(" $-$ ", "\n− ") for c in cols], fontsize=7.2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{b} · {k}" for b, k in rows] if ax is axes[0] else [], fontsize=7.6)
        ax.set_title(title, fontsize=9.5, loc="left", pad=6, color=INK)
        ax.grid(True, color="#eeeeea", lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=0)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=9, color=POS, label="favours the added component"),
         plt.Line2D([], [], marker="o", ls="none", ms=9, color=NEG, label="favours its absence"),
         plt.Line2D([], [], marker="o", ls="none", ms=11, markerfacecolor="#cccccc",
                    markeredgecolor=INK, markeredgewidth=1.4, label="$p<0.05$ (ring)"),
         plt.Line2D([], [], marker="o", ls="none", ms=5, color=MUTED, label="area $\\propto$ |$\\Delta$|")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Closed-form paired differences as a bubble matrix: size is magnitude, colour is sign, "
                 "a ring marks significance", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.01)
    save(fig, "tab10_13_closedform_paired_v3")

# ============================== D.1/D.2: v2 trade-off curve, v3 twin bars
def tD1_v2():
    eff, ret = gamma_rows()
    if not (eff and ret): return
    rmap = {r[0]: r[2] for r in ret}
    base = ret[0][1]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    xs = [rmap[e[0]] for e in eff if e[0] in rmap]
    ys = [e[1] for e in eff if e[0] in rmap]
    gs = [e[0] for e in eff if e[0] in rmap]
    sg = [e[4] for e in eff if e[0] in rmap]
    for x, y, g, s in zip(xs, ys, gs, sg):
        ax.plot([x], [y], marker="o", ms=10, color=S1 if s else SURF, markeredgecolor=S1,
                markeredgewidth=2.0, zorder=3)
        ax.annotate(f"$\\gamma$={g}", (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.axvline(base, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"no bonus\n({base:.2f}%)", (base, max(ys)), textcoords="offset points",
                xytext=(5, -4), va="top", fontsize=7.6, color=MUTED)
    for i in range(len(xs) - 1):
        ax.annotate("", (xs[i + 1], ys[i + 1]), (xs[i], ys[i]), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=7, shrinkB=7))
    ax.set_xlabel("question-entity retention (%) — what the bonus is built to raise")
    ax.set_ylabel("$\\Delta$ ROUGE-1 (points) — answer quality")
    ax.grid(True); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=9, color=S1, label="significant (Holm)"),
         plt.Line2D([], [], marker="o", ls="none", ms=9, color=SURF, markeredgecolor=S1,
                    markeredgewidth=2.0, label="not significant")]
    ax.legend(handles=h, loc="center left", fontsize=7.8)
    fig.suptitle("The trade-off directly: quality falls all the way along the path; retention only rises "
                 "as far as $\\gamma$=5",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=0.98)
    save(fig, "tabD1_D2_gamma_sweep_v2")
def tD1_v3():
    eff, ret = gamma_rows()
    if not (eff and ret): return
    rmap = {r[0]: r[2] for r in ret}; base = ret[0][1]
    gs = [e[0] for e in eff]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(gs)); w = 0.38
    q = [e[1] for e in eff]
    r = [rmap.get(g, np.nan) - base for g in gs]
    ax.bar(x - w / 2, q, width=w, color=S1, zorder=2, label="$\\Delta$ ROUGE-1 (quality)")
    ax.bar(x + w / 2, r, width=w, color=S3, alpha=0.9, zorder=2,
           label="$\\Delta$ retention vs no bonus")
    for i, (a, b) in enumerate(zip(q, r)):
        ax.annotate(f"{a:+.2f}", (i - w / 2, a), textcoords="offset points",
                    xytext=(0, 4 if a >= 0 else -12), ha="center", fontsize=7.2, color=INK)
        ax.annotate(f"{b:+.2f}", (i + w / 2, b), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.9, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"$\\gamma$={g}" for g in gs])
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.35, hi + 0.35)
    ax.set_ylabel("points")
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.8)
    fig.suptitle("Both quantities on one scale (both are points): retention gained, quality lost",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tabD1_D2_gamma_sweep_v3")

# ============================== D.4: v2 small-multiple lines, v3 cost-vs-w bars
def tD4_v2():
    weights, rows = fusion_rows()
    bbs = sorted({r[0] for r in rows}, key=lambda b: [r[0] for r in rows].index(b))
    benches = sorted({r[1] for r in rows}, key=lambda k: [r[1] for r in rows].index(k))
    # categorical slots 1-4 in the documented order, each also given its own marker so
    # identity never rests on colour alone (lines use the adjacent pairlist, which the order clears)
    COL = dict(zip(benches, [S1, S2, S3, "#eda100"]))
    MRK = dict(zip(benches, ["o", "s", "^", "D"]))
    fig, axes = plt.subplots(1, len(bbs), figsize=(10.6, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, bbs):
        for b, bench, vals in [r for r in rows if r[0] == bb]:
            ax.plot(weights, vals, color=COL[bench], lw=1.8, marker=MRK[bench], ms=5.5,
                    markeredgecolor=SURF, markeredgewidth=0.8, label=bench, zorder=3)
        ax.axvline(0.35, color=NEG, lw=1.3, ls=(0, (3, 2)), zorder=2)
        ax.set_title(bb, fontsize=9, loc="left", pad=6, color=INK)
        ax.set_xlabel("fusion weight $w$")
        ax.set_ylabel("accuracy (%)" if ax is axes[0] else "")
        ax.grid(True); ax.set_axisbelow(True)
    axes[0].annotate("executed $w$=0.35", (0.35, axes[0].get_ylim()[1]), textcoords="offset points",
                     xytext=(4, -11), fontsize=7.4, color=NEG)
    h = [plt.Line2D([], [], color=COL[k], marker=MRK[k], ms=6, lw=1.8, label=k) for k in benches]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Accuracy peaks at or just past $w$=0 in every one of the twelve cells; the executed "
                 "0.35 already sits on the falling side", fontsize=9.5, color=INK, x=0.01, ha="left",
                 y=1.02)
    save(fig, "tabD4_fusion_weight_sweep_v2")
def tD4_v3():
    weights, rows = fusion_rows()
    j35 = weights.index(0.35)
    cost = [(f"{b} · {k}", v[j35] - v[0]) for b, k, v in rows]
    cost.sort(key=lambda c: c[1])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(cost))
    ax.barh(y, [c[1] for c in cost], height=0.62, color=NEG, alpha=0.9, zorder=2)
    for i, (_n, v) in enumerate(cost):
        ax.annotate(f"{v:+.1f}", (v, i), textcoords="offset points", xytext=(-5, 0), ha="right",
                    va="center", fontsize=7.8, color=INK)
    ax.axvline(0, color=INK2, lw=0.9, zorder=3)
    ax.set_xlim(min(c[1] for c in cost) * 1.18, 0.4)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cost], fontsize=7.8)
    ax.set_xlabel("accuracy cost of the executed weight relative to no verifier (points)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    fig.suptitle("What the executed fusion weight costs: negative in all %d cells" % len(cost),
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tabD4_fusion_weight_sweep_v3")

# ============================== D.5: v2 grouped bars, v3 own-vs-transferred scatter
def tD5_v2():
    rows = retrain_rows()
    rows = sorted(rows, key=lambda r: r[4] - r[3])
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    y = np.arange(len(rows))
    for i, (b, k, nov, tr, own) in enumerate(rows):
        ax.plot([tr, own], [i, i], color=MUTED, lw=1.6, zorder=1,
                solid_capstyle="round")
        ax.plot([nov], [i], marker="|", ms=13, mew=2.0, color=INK2, zorder=4)
        ax.plot([tr], [i], marker="o", ms=9, color=S2, markeredgecolor=SURF, markeredgewidth=1.0,
                zorder=3)
        ax.plot([own], [i], marker="o", ms=9, color=S1, markeredgecolor=SURF, markeredgewidth=1.0,
                zorder=3)
        d = own - tr
        ax.annotate(f"{d:+.1f}", (max(tr, own, nov), i), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=7.6, color=POS if d > 0 else NEG)
    ax.set_yticks(y); ax.set_yticklabels([f"{b} · {k}" for b, k, *_ in rows], fontsize=8)
    ax.set_xlabel("closed-form accuracy (%)  \u2014  the number at the right of each row is "
                  "own $-$ transferred")
    ax.set_xlim(min(min(r[2:]) for r in rows) - 3, max(max(r[2:]) for r in rows) + 7)
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=8, color=S1, label="own-benchmark verifier @0.35"),
         plt.Line2D([], [], marker="o", ls="none", ms=8, color=S2, label="transferred verifier @0.35"),
         plt.Line2D([], [], marker="|", ls="none", ms=11, mew=2.0, color=INK2, label="no verifier")]
    ax.legend(handles=h, loc="lower center", ncol=3, fontsize=7.8, frameon=False,
              bbox_to_anchor=(0.5, -0.19))
    fig.suptitle("Retraining the verifier on each benchmark moves accuracy by at most a few points, "
                 "and often the wrong way", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.0)
    save(fig, "tabD5_verifier_retrain_v2")
def tD5_v3():
    rows = retrain_rows()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lo = min(min(r[3], r[4]) for r in rows) - 3
    hi = max(max(r[3], r[4]) for r in rows) + 3
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    marks = {"MedQA": "o", "MMLU-Med": "s", "MedMCQA": "^", "PubMedQA": "D"}
    cols = {"LLaMA-3.1-8B": S1, "Qwen3-8B": S2, "DeepSeek-R1-8B": S3}
    for b, k, nov, tr, own in rows:
        ax.plot([tr], [own], marker=marks.get(k, "o"), ms=9, color=cols.get(b, S1),
                markeredgecolor=SURF, markeredgewidth=0.9, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("transferred verifier @0.35 (% accuracy)")
    ax.set_ylabel("own-benchmark verifier @0.35 (% accuracy)")
    ax.annotate("above the line:\nretraining helps", (lo + 2, hi - 6), fontsize=7.8, color=MUTED)
    ax.grid(True); ax.set_axisbelow(True); ax.set_aspect("equal")
    h = [plt.Line2D([], [], marker="o", ls="none", ms=8, color=c, label=b) for b, c in cols.items()]
    h += [plt.Line2D([], [], marker=m, ls="none", ms=8, color=MUTED, label=k) for k, m in marks.items()]
    ax.legend(handles=h, loc="upper center", ncol=4, fontsize=7.2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Own-benchmark against transferred verifier; points above the diagonal are cells where "
                 "retraining helps", fontsize=9, color=INK, x=0.01, ha="left", y=0.99)
    save(fig, "tabD5_verifier_retrain_v3")

# ============================== D.5 further forms: v4 paired deltas, v5 slope facets,
# v6 diverging heatmap pair, v7 quadrant scatter. All measure against "no verifier",
# which is the baseline the decision actually turns on.
BENCH_ORDER = ["MedQA", "MMLU-Med", "MedMCQA", "PubMedQA"]
BB_COL = {"LLaMA-3.1-8B": S1, "Qwen3-8B": S2, "DeepSeek-R1-8B": S3}
BENCH_MRK = {"MedQA": "o", "MMLU-Med": "s", "MedMCQA": "^", "PubMedQA": "D"}

def tD5_v4():
    rows = sorted(retrain_rows(), key=lambda r: r[4] - r[2])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(rows)); h = 0.34
    do = [r[4] - r[2] for r in rows]; dt = [r[3] - r[2] for r in rows]
    ax.barh(y + h / 2, do, height=h, color=S1, zorder=2, label="own-benchmark verifier $-$ no verifier")
    ax.barh(y - h / 2, dt, height=h, color=S2, zorder=2, label="transferred verifier $-$ no verifier")
    for i, (a, b) in enumerate(zip(do, dt)):
        for v, off in ((a, h / 2), (b, -h / 2)):
            ax.annotate(f"{v:+.1f}", (v, i + off), textcoords="offset points",
                        xytext=(5 if v >= 0 else -5, 0), ha="left" if v >= 0 else "right",
                        va="center", fontsize=7.2, color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{b} \u00b7 {k}" for b, k, *_ in rows], fontsize=8)
    lo = min(min(do), min(dt)); hi = max(max(do), max(dt))
    ax.set_xlim(lo - 1.2, hi + 1.2)
    ax.set_xlabel("change in closed-form accuracy against using no verifier (points)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=2, fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, -0.19))
    ax.annotate("helps \u2192", (hi + 1.0, len(rows) - 0.3), ha="right", fontsize=7.6, color=POS)
    ax.annotate("\u2190 hurts", (lo - 1.0, len(rows) - 0.3), ha="left", fontsize=7.6, color=NEG)
    nO = sum(v > 0 for v in do); nT = sum(v > 0 for v in dt)
    fig.suptitle(f"Measured against no verifier at all: the retrained verifier helps in {nO} of 12 cells "
                 f"and the transferred one in {nT} of 12", fontsize=9.5, color=INK, x=0.01, ha="left",
                 y=1.0)
    save(fig, "tabD5_verifier_retrain_v4")

def tD5_v5():
    rows = retrain_rows()
    benches = [k for k in BENCH_ORDER if any(r[1] == k for r in rows)]
    fig, axes = plt.subplots(1, len(benches), figsize=(10.8, 3.9), sharex=True)
    axes = np.atleast_1d(axes)
    xs = [0, 1, 2]
    for ax, bench in zip(axes, benches):
        sub = [r for r in rows if r[1] == bench]
        for b, k, nov, tr, own in sub:
            c = BB_COL.get(b, MUTED)
            ax.plot(xs, [nov, tr, own], color=c, lw=1.9, marker="o", ms=7,
                    markeredgecolor=SURF, markeredgewidth=1.2, zorder=3, label=b)
            ax.annotate(f"{own:.1f}", (2, own), textcoords="offset points", xytext=(7, 0),
                        va="center", fontsize=7.2, color=INK)
            ax.annotate(f"{nov:.1f}", (0, nov), textcoords="offset points", xytext=(-7, 0),
                        ha="right", va="center", fontsize=7.2, color=INK)
        lo = min(min(r[2:]) for r in sub); hi = max(max(r[2:]) for r in sub)
        pad = max(2.0, (hi - lo) * 0.28)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.75, 2.75)
        ax.set_xticks(xs); ax.set_xticklabels(["none", "transferred", "own"], fontsize=7.6)
        ax.set_title(bench, fontsize=9, loc="left", pad=6, color=INK)
        ax.set_ylabel("closed-form accuracy (%)" if ax is axes[0] else "")
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=c, marker="o", ms=6, lw=1.9, label=b) for b, c in BB_COL.items()]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Each panel on its own scale: moving from no verifier to a transferred one to a "
                 "retrained one, per benchmark", fontsize=9.5, color=INK, x=0.01, ha="left", y=1.03)
    save(fig, "tabD5_verifier_retrain_v5")

def tD5_v6():
    rows = retrain_rows()
    bbs = sorted({r[0] for r in rows}, key=lambda b: [r[0] for r in rows].index(b))
    benches = [k for k in BENCH_ORDER if any(r[1] == k for r in rows)]
    idx = {(r[0], r[1]): r for r in rows}
    MT = np.array([[idx[(b, k)][3] - idx[(b, k)][2] for k in benches] for b in bbs])
    MO = np.array([[idx[(b, k)][4] - idx[(b, k)][2] for k in benches] for b in bbs])
    lim = max(abs(MT).max(), abs(MO).max())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), gridspec_kw=dict(wspace=0.08))
    for ax, M, t in ((axes[0], MT, "Transferred verifier"), (axes[1], MO, "Retrained on this benchmark")):
        im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                       aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if abs(v) > lim * 0.55 else INK)
        ax.set_xticks(range(len(benches))); ax.set_xticklabels(benches, fontsize=8)
        ax.set_yticks(range(len(bbs)))
        ax.set_yticklabels(bbs if ax is axes[0] else [], fontsize=8)
        ax.tick_params(length=0); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(t, fontsize=9, loc="left", pad=6, color=INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("$\\Delta$ accuracy vs no verifier (points)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Both verifiers against no verifier, in the heatmap form used for the closed-form "
                 "grids. Blue favours having a verifier, red favours not.", fontsize=9.5, color=INK,
                 x=0.01, ha="left", y=1.04)
    save(fig, "tabD5_verifier_retrain_v6")

def tD5_v7():
    rows = retrain_rows()
    xs = [r[3] - r[2] for r in rows]; ys = [r[4] - r[2] for r in rows]
    # one shared range on both axes, so the 45-degree diagonal still means equality, but fitted to
    # the data rather than forced symmetric about zero (which would leave three quadrants empty)
    xlo, xhi = min(xs) - 0.8, max(xs) + 0.8
    ylo, yhi = min(ys) - 0.8, max(ys) + 0.8
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.add_patch(Rectangle((0, 0), xhi, yhi, color=POS, alpha=0.055, zorder=0))
    ax.add_patch(Rectangle((xlo, ylo), -xlo, -ylo, color=NEG, alpha=0.055, zorder=0))
    d0, d1 = max(xlo, ylo), min(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=INK2, lw=1.0, zorder=2); ax.axvline(0, color=INK2, lw=1.0, zorder=2)
    for (b, k, nov, tr, own), x, yv in zip(rows, xs, ys):
        ax.plot([x], [yv], marker=BENCH_MRK.get(k, "o"), ms=10, color=BB_COL.get(b, MUTED),
                markeredgecolor=SURF, markeredgewidth=1.1, zorder=4)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("transferred verifier $-$ no verifier (points)")
    ax.set_ylabel("retrained verifier $-$ no verifier (points)")
    ax.annotate("both verifiers help", (xhi - 0.15, yhi - 0.1), ha="right", va="top", fontsize=7.6,
                color=POS)
    ax.annotate("neither helps", (xlo + 0.15, ylo + 0.1), ha="left", va="bottom", fontsize=7.6,
                color=NEG)
    ax.annotate("above the dashed line:\nretraining beats transferring (dashed)", (xlo + 0.15, yhi - 0.1),
                ha="left", va="top", fontsize=7.4, color=MUTED)
    ax.grid(True); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=8, color=c, label=b) for b, c in BB_COL.items()]
    h += [plt.Line2D([], [], marker=BENCH_MRK[k], ls="none", ms=8, color=MUTED, label=k)
          for k in BENCH_ORDER if k in {r[1] for r in rows}]
    ax.legend(handles=h, loc="lower center", ncol=4, fontsize=7.2, frameon=False,
              bbox_to_anchor=(0.5, -0.22))
    nb = sum(x > 0 and y > 0 for x, y in zip(xs, ys))
    nn = sum(x <= 0 and y <= 0 for x, y in zip(xs, ys))
    fig.suptitle(f"Only {nb} of 12 cells sit where both verifiers beat no verifier; {nn} sit where "
                 f"neither does", fontsize=9.5, color=INK, x=0.01, ha="left", y=0.99)
    save(fig, "tabD5_verifier_retrain_v7")


def _wrap2(name):
    """Two lines, split at the space nearest the middle, so no two tick labels can touch."""
    if len(name) < 16 or " " not in name: return name
    cuts = [i for i, c in enumerate(name) if c == " "]
    i = min(cuts, key=lambda i: abs(i - len(name) / 2))
    return name[:i].rstrip(",") + ",\n" + name[i + 1:] if name[i - 1] == "," else name[:i] + "\n" + name[i + 1:]

def _leak_sorted():
    rows = leakage_rows()
    rows.sort(key=lambda r: STRICT_ORDER.index(r[0]) if r[0] in STRICT_ORDER else 99)
    return rows

def t19_v4():
    """Horizontal lollipops, strictest at the bottom. Each design owns a row, so no label can
    collide with another; the value sits past the end of its own stem."""
    rows = _leak_sorted()[::-1]
    vals = [r[2] for r in rows]
    strict = [("disjoint" in r[0].lower()) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(rows))
    for i, (v, st) in enumerate(zip(vals, strict)):
        c = S1 if st else "#9ec5f4"
        ax.plot([0, v], [i, i], color=c, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot([v], [i], marker="o", ms=11, color=c, markeredgecolor=SURF, markeredgewidth=1.4,
                zorder=3)
        ax.annotate(f"{v:+.2f}", (v, i), textcoords="offset points", xytext=(13, 0), ha="left",
                    va="center", fontsize=8.4, color=INK)
    ax.axvline(0, color=INK2, lw=0.9, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-0.15, max(vals) + 1.15)
    ax.set_xlabel("$\\Delta$ ROUGE-1 from retrieval (points)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    ax.annotate("", xy=(-0.335, -0.35), xytext=(-0.335, len(rows) - 0.65),
                xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1), annotation_clip=False)
    ax.text(-0.365, (len(rows) - 1) / 2, "stricter control", rotation=90, ha="center", va="center",
            fontsize=7.8, color=MUTED, transform=ax.get_yaxis_transform())
    h = [plt.Line2D([], [], marker="o", ls="none", ms=9, color=S1, label="shared material removed"),
         plt.Line2D([], [], marker="o", ls="none", ms=9, color="#9ec5f4", label="random split")]
    ax.legend(handles=h, loc="lower right", fontsize=7.8, frameon=False)
    fig.suptitle("The retrieval effect does not fall as control tightens", fontsize=10, color=INK,
                 x=0.01, ha="left", y=1.0)
    save(fig, "tab19_22_leakage_convergence_v4")

def t19_v5():
    """Every estimate read against the paper's own headline number, so the question does tightening the control shrink the effect? - is answered on the zero line itself."""
    rows = _leak_sorted()
    base = dict((r[0], r[2]) for r in rows)["Random split, all questions"]
    rows = [r for r in rows if r[0] != "Random split, all questions"][::-1]
    d = [r[2] - base for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    y = np.arange(len(rows))
    ax.barh(y, d, height=0.52, color=[POS if v >= 0 else NEG for v in d], zorder=2)
    for i, (r, v) in enumerate(zip(rows, d)):
        ax.annotate(f"{v:+.2f}  (={r[2]:+.2f})", (v, i), textcoords="offset points",
                    xytext=(8 if v >= 0 else -8, 0), ha="left" if v >= 0 else "right", va="center",
                    fontsize=8.0, color=INK)
    ax.axvline(0, color=INK2, lw=1.1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    lo, hi = min(d), max(d)
    ax.set_xlim(lo - 1.5, hi + 1.5)
    ax.set_xlabel(f"difference from the primary estimate of {base:+.2f} points "
                  f"(own value in brackets)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    ax.annotate(f"primary estimate {base:+.2f}", (0, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(0, 5), ha="center", va="bottom",
                fontsize=7.8, color=MUTED, annotation_clip=False)
    fig.suptitle("Three of the four alternative designs land above the paper's primary estimate, "
                 "not below it", fontsize=10, color=INK, x=0.01, ha="left", y=1.05)
    save(fig, "tab19_22_leakage_convergence_v5")

def t19_v6():
    """Stat tiles: five numbers whose message is that they do not shrink. Each tile is its own box,
    so nothing shares space with anything else."""
    import textwrap
    rows = _leak_sorted()
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(11.4, 3.1))
    top = max(r[2] for r in rows)
    for ax, (name, note, v) in zip(np.atleast_1d(axes), rows):
        st = "disjoint" in name.lower()
        c = S1 if st else "#9ec5f4"
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                               facecolor="#f4f3f0" if st else SURF, edgecolor="#e3e2de", lw=1.0))
        ax.text(0.5, 0.86, "\n".join(textwrap.wrap(name, 18)), ha="center", va="top", fontsize=8.2,
                color=INK)
        ax.text(0.5, 0.45, f"{v:+.2f}", ha="center", va="center", fontsize=23, color=c)
        ax.text(0.5, 0.315, "$\\Delta$ ROUGE-1 points", ha="center", va="center", fontsize=7.0,
                color=MUTED)
        ax.add_patch(Rectangle((0.14, 0.12), 0.72 * (v / top if top else 0), 0.055,
                               transform=ax.transAxes, facecolor=c, edgecolor="none"))
        ax.add_patch(Rectangle((0.14, 0.12), 0.72, 0.055, transform=ax.transAxes,
                               facecolor="none", edgecolor="#e3e2de", lw=0.8))
    fig.suptitle("Five designs, from the loosest control on the left to the strictest on the right. "
                 "The two in the darker blue are the ones that remove shared material.",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=1.02)
    save(fig, "tab19_22_leakage_convergence_v6")

def build_all_alt():
    for f in (t5_v2, t5_v3, t6_v2, t6_v3, t19_v2, t19_v3, t3_v2, t3_v3,
              t10_v2, t10_v3, tD1_v2, tD1_v3, tD4_v2, tD4_v3, tD5_v2, tD5_v3,
              tD5_v4, tD5_v5, tD5_v6, tD5_v7, t19_v4, t19_v5, t19_v6):
        try:
            f()
        except MissingTable as e:
            print(f"  SKIP {f.__name__}: the manuscript has no table {e.args[0]}")
if __name__ == "__main__":
    build_all_alt()

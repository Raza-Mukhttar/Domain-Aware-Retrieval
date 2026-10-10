# -*- coding: utf-8 -*-
"""Independent audit: every number a figure draws must appear in the table it was drawn from."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import make_table_figures as T
from make_table_figures import MissingTable
CHECKS = []
def chk(name, drawn, label):
    try:
        body = T.clean(T.table_body(label))
    except MissingTable:
        print(f"  {name:38s} skipped | the manuscript has no table {label}")
        return
    have = set(re.findall(r"\d+\.\d+", body))
    missing = [f"{v:.2f}" for v in drawn if f"{abs(v):.2f}" not in have and f"{abs(v):.1f}" not in have]
    CHECKS.append((name, len(drawn), len(missing), missing[:4]))
# 1 forest
vals = []
for kind, v in T.rows_of("tab:primary-components"):
    if kind == "__ROW__" and isinstance(v, list) and len(v) > 1:
        p = T.parse_ci(v[1])
        if p: vals += [p[0], p[1], p[2]]
chk("tab5_primary_components", vals, "tab:primary-components")
# 2 ladder
lv = T._grouped("tab:ladder-levels", 2)
chk("tab6_7_cumulative_ladder", [x[1] for d in lv.values() for x in d], "tab:ladder-levels")
# 3 leakage
lk = []
for kind, v in (T.rows_of("tab:leakage-convergence")
                if "\\label{tab:leakage-convergence}" in T.TEXT else []):
    if kind == "__ROW__" and isinstance(v, list) and len(v) > 2:
        m = re.search(r"([+-]?\d+\.\d+)", T.clean(v[2]))
        if m: lk.append(float(m.group(1)))
chk("tab19_22_leakage_convergence", lk, "tab:leakage-convergence")
# 5 heatmaps
for lab, nm in (("tab:cf-matched-paired", "tab10_13 (matched)"), ("tab:cf-full-paired", "tab10_13 (full)")):
    cols, data = T._paired_grid(lab)
    chk(nm, [v for b in data for k in data[b] for v in data[b][k][0] if not np.isnan(v)], lab)
# 6 gamma
gv = []
for kind, v in T.rows_of("tab:gamma-sweep"):
    if kind == "__ROW__" and isinstance(v, list) and len(v) > 1:
        p = T.parse_ci(v[1])
        if p: gv += [p[0], p[1], p[2]]
chk("tabD1_D2_gamma_sweep", gv, "tab:gamma-sweep")
print(f"{'figure':34s} {'values':>7s} {'missing':>8s}")
bad = 0
for n, tot, miss, ex in CHECKS:
    print(f"{n:34s} {tot:7d} {miss:8d} {'' if not miss else ex}")
    bad += miss
print(f"\nvalues drawn that do not appear in their source table: {bad}")
sys.exit(1 if bad else 0)

"""Assemble mouse_embryo_ct_distribution_summary.xlsx from the analysis CSVs.

Sheets: distribution (30-class counts/proportions), correlation (metrics),
z_freq (long table, original 13 layers + reconstructed z bins).

Run (env: ntf):
  /home/gongyuqiao/.conda/envs/ntf/bin/python make_summary_xlsx.py
"""

import pandas as pd

dist = pd.read_csv("ct_distribution.csv")
metrics = pd.read_csv("ct_distribution_metrics.csv")
zfreq = pd.read_csv("z_freq_long.csv")

with pd.ExcelWriter("mouse_embryo_ct_distribution_summary.xlsx",
                    engine="openpyxl") as xw:
    dist.to_excel(xw, sheet_name="distribution", index=False)
    metrics.to_excel(xw, sheet_name="correlation", index=False)
    zfreq.to_excel(xw, sheet_name="z_freq", index=False)

print("saved mouse_embryo_ct_distribution_summary.xlsx",
      {s: df.shape for s, df in
       [("distribution", dist), ("correlation", metrics), ("z_freq", zfreq)]})

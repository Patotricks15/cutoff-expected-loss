# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CR06 - Cutoff economics (LendingClub)
#
# Goal: which PD cutoff maximizes expected profit? A loan is approved when its PD is at or below
# the cutoff. The PD comes from the CR02 monotone LightGBM challenger, trained on loans issued
# 2007 to 2014 and scored on loans issued 2015 to 2016.
#
# Profit per approved loan (per unit of exposure, EAD):
# `(1 - PD) * margin - PD * LGD`. LGD and margin are stated assumptions, not data.
# A loan is worth approving when PD < margin / (margin + LGD), the breakeven PD.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from credit_core.economics import cutoff_profit, expected_loss
from credit_core.metrics import gini
from credit_core.plots import plot_profit_curve
from cutoff_econ.econ import best_cutoff, breakeven_pd, sensitivity_grid
from cutoff_econ.pipeline import load_test_book

FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Scores and exposure on the test window

# %%
book = load_test_book()
good_score = 1.0 - book.pd_
print(f"Test loans: {len(book.pd_):,}")
print(f"Observed bad rate (resolved only): {book.defaulted.mean():.3f}")
print(f"Mean PD: {book.pd_.mean():.3f}")
print(f"Gini (challenger, test): {gini(1 - book.defaulted, good_score):.3f}")
print(f"Total EAD: {book.ead.sum():,.0f}")

# %% [markdown]
# ## 2. Assumptions
#
# Base case: LGD = 0.80 of EAD (low recoveries on unsecured consumer loans) and margin = 0.12 of EAD
# on a repaid loan (interest earned over the loan life). Both are assumptions. Section 4 varies them.

# %%
LGD, MARGIN = 0.80, 0.12
BREAKEVEN = breakeven_pd(MARGIN, LGD)
print(f"Breakeven PD at LGD={LGD}, margin={MARGIN}: {BREAKEVEN:.4f}")

# %% [markdown]
# ## 3. Profit curve and optimal cutoff

# %%
cutoffs = np.arange(0.01, 0.601, 0.005)
table = cutoff_profit(book.pd_, book.defaulted, book.ead, margin=MARGIN, lgd=LGD, cutoffs=cutoffs)
best = best_cutoff(table)
print(f"Optimal cutoff (expected profit): PD <= {best['cutoff_pd']:.3f}")
print(f"Approval rate at optimum: {best['approval_rate']:.1%}")
print(f"Expected profit at optimum: {best['expected_profit']:,.0f}")
print(f"Realized profit at optimum: {best['realized_profit']:,.0f}")
print(f"Expected loss at optimum: {expected_loss(book.pd_[book.pd_ <= best['cutoff_pd']], LGD, book.ead[book.pd_ <= best['cutoff_pd']]).sum():,.0f}")

# %%
fig = plot_profit_curve(table)
fig.axes[0].axvline(best["approval_rate"], color="grey", linestyle="--", label="optimum")
fig.axes[0].legend()
fig.savefig(FIG_DIR / "cr06_profit_curve.png", dpi=120, bbox_inches="tight")
plt.show()

# %%
last = table.iloc[-1]
print(f"Widest cutoff PD <= {last['cutoff_pd']:.3f}: approval {last['approval_rate']:.1%}, "
      f"expected profit {last['expected_profit']:,.0f}, realized profit {last['realized_profit']:,.0f}")
print(f"Calibration check: mean PD {book.pd_.mean():.3f} vs observed bad rate {book.defaulted.mean():.3f}")

# %% [markdown]
# The expected-profit curve peaks at the breakeven PD, where expected profit is about 188M. Realized
# profit at the same cutoff is about 104M, roughly half of that. The mean PD (0.171) is below the
# observed bad rate (0.215), so the challenger under-predicts defaults on this window. This matches
# the rising bad rate found in CR05. Expected profit is therefore optimistic, and the cutoff should
# not be set from it alone.

# %% [markdown]
# ## 4. Sensitivity to LGD and margin

# %%
lgds = [0.60, 0.70, 0.80]
margins = [0.08, 0.12, 0.16]
grid = sensitivity_grid(book.pd_, book.defaulted, book.ead, cutoffs, lgds, margins)
print(grid.round(4).to_string(index=False))

# %%
pivot = grid.pivot(index="margin", columns="lgd", values="best_cutoff")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Optimal PD cutoff"}, ax=ax)
ax.set(xlabel="LGD", ylabel="Margin", title="Optimal PD cutoff by LGD and margin")
fig.savefig(FIG_DIR / "cr06_sensitivity.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Summary
#
# - At the base case (LGD 0.80, margin 0.12), the optimal cutoff is the breakeven PD of 0.130. It
#   approves about 44% of loans. Approving every loan has negative expected profit.
# - The optimal cutoff moves a lot with the assumptions: 0.09 to 0.21 across the grid. A lower
#   LGD or a higher margin widens the approval band. Set the cutoff from a real LGD and margin.
# - Expected profit is about 1.8 times realized profit at the optimum. The challenger under-predicts
#   defaults on 2015 to 2016 (mean PD 0.171 vs observed bad rate 0.215), which matches the drift in
#   CR05. Recalibrate before using expected profit for a live cutoff.
# - Realized profit uses resolved loans only. Loans still open in 2016 are excluded, and they are
#   likely riskier, so realized profit is somewhat optimistic in absolute terms.
# - LGD and margin are assumptions. This analysis does not estimate them from data.

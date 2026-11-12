# CR06: Cutoff economics

Which PD cutoff maximizes expected profit? This project reads the PD from the CR02 monotone
LightGBM challenger and chooses a cutoff using a profit model with stated LGD and margin.

## Setup

- Data: LendingClub accepted loans. See [data/README.md](data/README.md).
- Score: the CR02 challenger, trained on loans issued 2007 to 2014. Scored on loans issued 2015 to
  2016 (n = 668,651). Test Gini is 0.430.
- Exposure: `EAD` = `loan_amnt`.
- Target: default = Charged Off or Default. Only resolved loans are kept.

## Method

Profit for a cutoff `c` approves loans with PD at or below `c`:

```
profit(c) = sum over approved of [(1 - PD) * margin - PD * LGD] * EAD
```

- **Breakeven PD** = `margin / (margin + LGD)`. A loan is worth approving when its PD is below this.
- **Expected profit** uses the predicted PD. **Realized profit** uses the observed default flag.
- **Sensitivity** repeats the optimum for LGD in {0.60, 0.70, 0.80} and margin in {0.08, 0.12, 0.16}.
- **Expected loss** of the approved book is `sum(PD * LGD * EAD)`.

LGD and margin are **assumptions**, not estimated from data. The base case is LGD 0.80 and margin 0.12.

## Results (base case)

- Optimal cutoff is PD ≤ 0.130, the breakeven PD. It approves 44.4% of loans.
- Expected profit at the optimum is about 188M. Realized profit at the same cutoff is about 104M.
- Approving 99.7% of loans gives negative expected profit (about −466M).
- Across the sensitivity grid, the optimal cutoff ranges from 0.09 to 0.21.

Expected profit is about 1.8 times realized profit. Mean PD is 0.171, but the observed bad rate is
0.215. The challenger under-predicts defaults on this window. This is the same drift found in
CR05. Recalibrate before using expected profit to set a live cutoff.

See [notebooks/06_cutoff_economics.py](notebooks/06_cutoff_economics.py) and the figures:
[cr06_profit_curve.png](reports/figures/cr06_profit_curve.png) and
[cr06_sensitivity.png](reports/figures/cr06_sensitivity.png).

## Limitations

- **LGD and margin are assumed.** The optimal cutoff moves a lot with them (see the grid). They
  should come from the lender's own recovery and pricing data.
- **Under-prediction:** the challenger's PD is below the observed bad rate on this window, so
  expected profit is optimistic.
- **Resolved-only selection:** loans still open in 2016 are excluded. They are likely riskier, so
  realized profit is somewhat optimistic in absolute terms.
- **Single margin:** margin is a constant share of EAD. The real return depends on each loan's rate
  and term.
- **Censoring:** 2017 and later are excluded for the same reason as in CR02.

## Reproduce

```bash
make install
make test
make notebook
```

Notebooks run with the `credit-venv` kernel. The notebook takes about 3 minutes.

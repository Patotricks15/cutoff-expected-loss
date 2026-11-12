"""Cutoff choice on top of credit_core.economics.

A loan approved at PD p returns `margin` per unit EAD when repaid and loses `lgd` per unit EAD
when it defaults. It is worth approving when (1 - p) * margin > p * lgd, which is p < margin / (margin + lgd).
"""

import numpy as np
import pandas as pd

from credit_core.economics import cutoff_profit, expected_loss

__all__ = ["breakeven_pd", "best_cutoff", "sensitivity_grid", "expected_loss", "cutoff_profit"]


def breakeven_pd(margin: float, lgd: float) -> float:
    if margin < 0 or lgd <= 0:
        raise ValueError("margin must be >= 0 and lgd must be > 0")
    return margin / (margin + lgd)


def best_cutoff(table: pd.DataFrame, column: str = "expected_profit") -> pd.Series:
    return table.loc[table[column].idxmax()]


def sensitivity_grid(pd_, defaulted, ead, cutoffs, lgds, margins) -> pd.DataFrame:
    """Best expected-profit cutoff for each (LGD, margin) pair, with realized profit at that cutoff."""
    rows = []
    for lgd in lgds:
        for margin in margins:
            table = cutoff_profit(pd_, defaulted, ead, margin=margin, lgd=lgd, cutoffs=cutoffs)
            best = best_cutoff(table)
            rows.append(
                {
                    "lgd": lgd,
                    "margin": margin,
                    "breakeven_pd": breakeven_pd(margin, lgd),
                    "best_cutoff": best["cutoff_pd"],
                    "approval_rate": best["approval_rate"],
                    "expected_profit": best["expected_profit"],
                    "realized_profit": best["realized_profit"],
                }
            )
    return pd.DataFrame(rows)

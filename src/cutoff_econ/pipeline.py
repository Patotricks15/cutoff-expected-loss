"""PD and EAD on the CR02 out-of-time window (loans issued 2015 to 2016), for the cutoff analysis."""

from dataclasses import dataclass

import numpy as np

from credit_core.datasets import load_lendingclub, split_by_issue_date
from gbm_monotone.pipeline import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    challenger_scores,
    fit_challenger,
    load_oot,
)


@dataclass
class PdBook:
    pd_: np.ndarray
    defaulted: np.ndarray
    ead: np.ndarray


def load_test_book() -> PdBook:
    d = load_oot()
    model = fit_challenger(d)
    p_good = challenger_scores(model, d, d.X_test)

    df = load_lendingclub()
    _, te = split_by_issue_date(df, TRAIN_END, TEST_START)
    te = te[te["issue_date"] < TEST_END]
    y = d.y_test.to_numpy()
    if len(te) != len(y) or not np.array_equal(te["y"].to_numpy(), y):
        raise RuntimeError("EAD rows do not align with the CR02 test window")

    return PdBook(
        pd_=1.0 - p_good,
        defaulted=1.0 - y.astype(float),
        ead=te["EAD"].to_numpy(dtype=float),
    )

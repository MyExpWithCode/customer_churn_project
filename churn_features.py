"""Row-wise feature engineering for the Telco customer-churn model.

This module is the single source of truth for the engineered features. It is
imported by the analysis notebook (section 3) and by the prediction API
(``app.py``, section 7), and ``add_features`` travels inside the saved pipeline
as its first step -- so a request to ``POST /predict`` supplies only the raw
customer fields and the artifact derives everything else itself.

It has to be an importable module rather than notebook code for a concrete
reason. ``joblib`` pickles a function by *reference*, storing ``module`` and
``qualname`` rather than the function body; a function defined in a notebook
records its module as ``"__main__"``, so ``joblib.load`` inside ``app.py``
could not resolve it and would raise. Defining it here makes the stored
reference ``churn_features.add_features``, which any process started from the
project root can import.

Every feature below reads a single customer's own attributes and nothing else.
None of them consult a training-set aggregate -- no percentile bands, no group
means, no fitted state -- so the value computed for one incoming request is
identical to the value computed during training, and no target information can
reach the model through the feature layer.

Dependencies: numpy and pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The six optional services that require an internet subscription. Customers
#: without internet carry the sentinel ``"No internet service"`` in all six.
ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Fixed cut points, deliberately not data-driven percentiles: such a boundary
# is a property of the training set, and reproducing it for one unseen customer
# would mean shipping training data to prediction time.
#
# The top edge is np.inf rather than 72 (the observed maximum) so that no
# future tenure can fall outside the bins and become NaN. Section 2's analysis
# used a closed 72 edge, which is right for describing the data in hand and
# wrong for an endpoint that will eventually see a longer-tenured customer.
TENURE_BINS = [0, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]

ENGINEERED_NUMERIC = ["n_addons"]
ENGINEERED_BINARY = ["is_high_risk", "automatic_payment"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_FEATURES = (
    ENGINEERED_NUMERIC + ENGINEERED_BINARY + ENGINEERED_CATEGORICAL
)

#: Ranked below Cramer's V = 0.05 in section 2.5 and confirmed by the section 3
#: ablation: removing these three improves recall, precision and F1 together.
DROP_FEATURES = ["gender", "PhoneService", "MultipleLines"]

#: Raw columns ``add_features`` needs. Checked up front so a malformed request
#: fails with a readable message rather than a bare KeyError from deep inside
#: a pipeline step.
REQUIRED_COLUMNS = list(
    dict.fromkeys(
        ["tenure", "Contract", "InternetService", "PaymentMethod"] + ADDON_COLUMNS
    )
)


def tenure_bucket(tenure):
    """Tenure in months -> one of five fixed bands, as an ordered Categorical.

    Ordered rather than plain strings so tables and charts sort by band instead
    of alphabetically, which would give ``0-6, 13-24, 25-48, 49+, 7-12``. The
    categories are fixed by ``TENURE_LABELS``, so a single row produces the
    same set of one-hot columns as a full batch. Negative input is clipped to
    zero, so a malformed request lands in the first band rather than becoming
    a null the encoder would have to guess about.
    """
    months = tenure if isinstance(tenure, pd.Series) else pd.Series(tenure)
    return pd.cut(
        months.clip(lower=0),
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
        ordered=True,
    ).rename("tenure_bucket")


def addon_count(frame):
    """How many of the six optional add-ons the customer holds, 0 to 6.

    Customers without internet are recorded as ``"No internet service"`` for
    every add-on, so they score 0 here -- the same as a customer who has
    internet and declined all six. Section 2.8 showed that conflating those two
    groups inverts the apparent relationship with churn, which is why the model
    also receives ``InternetService`` (already one-hot encoded by section 1's
    preprocessor) and can separate them itself.
    """
    return (frame[ADDON_COLUMNS] == "Yes").sum(axis=1).rename("n_addons")


def is_high_risk(frame):
    """The four-condition high-risk segment identified in section 2.9.

    Month-to-month *and* fibre *and* no OnlineSecurity *and* no TechSupport:
    21.6% of training customers, churning at 60.4%, holding 49.1% of all
    churners. Supplied pre-computed because a four-way conjunction costs a
    decision tree four levels of depth, which a shallow, readable tree cannot
    afford to spend -- see the ablation at section 3.4.
    """
    return (
        (frame["Contract"] == "Month-to-month")
        & (frame["InternetService"] == "Fiber optic")
        & (frame["OnlineSecurity"] == "No")
        & (frame["TechSupport"] == "No")
    ).astype("int64").rename("is_high_risk")


def is_automatic_payment(frame):
    """1 if the customer pays by an automatic method, 0 if they pay manually.

    Collapses the four payment methods to two. Section 2.6 found electronic
    check churning at 45.6% against 14.7-16.3% for the two automatic methods --
    though mailed check is equally manual at 19.5%, so this flag merges two
    quite different groups and the ablation treats it as a candidate to test
    rather than an obvious win.
    """
    return (
        frame["PaymentMethod"]
        .str.contains("automatic", case=False, na=False)
        .astype("int64")
        .rename("automatic_payment")
    )


def add_features(frame):
    """Return a copy of ``frame`` with the four engineered columns appended.

    Copies first: inside a scikit-learn ``Pipeline`` this runs on the caller's
    own frame, and a transformer that mutates its input is a bug waiting for
    the second ``fit``. Behaves identically on one row and on 4,930 -- asserted
    in section 3.3, because that equivalence is what makes the same code usable
    for training and for a single API request.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(
            f"add_features needs columns that are not present: {missing}"
        )

    out = frame.copy()
    out["tenure_bucket"] = tenure_bucket(out["tenure"])
    out["n_addons"] = addon_count(out)
    out["is_high_risk"] = is_high_risk(out)
    out["automatic_payment"] = is_automatic_payment(out)
    return out

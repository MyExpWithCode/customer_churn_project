# Customer Churn Prediction — Telco

End-to-end machine learning solution predicting telecom customer churn, from raw data
through to a REST prediction API.

**Business problem.** Customers who leave ("churn") cost far more to replace than to
retain. This project identifies customers likely to churn so a retention team can engage
them before they go.

- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 columns
- **Target:** `Churn` (Yes / No), 26.54% positive class
- **Model:** Decision Tree Classifier (`entropy`, `min_samples_leaf=50`, `class_weight="balanced"`)
- **Split:** 70:30, `random_state=42`, stratified on the target
- **Held-out test performance:** recall **0.781**, precision **0.505**, F1 **0.613**,
  ROC-AUC **0.820**, accuracy **0.739** — finds 438 of 561 churners, protecting
  **80.1% of at-risk monthly revenue**

---

## Build status

| Section | Status |
|---|---|
| 1. Data understanding & preparation | ✅ Complete |
| 2. Exploratory data analysis | ✅ Complete |
| 3. Feature engineering | ✅ Complete |
| 4. Model development | ✅ Complete |
| 5. Model evaluation | ✅ Complete |
| 6. Model interpretation | ✅ Complete |
| 7. Model saving & API | Not started |

---

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv                   raw dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv  column types and expected values
├── notebook/
│   └── churn_analysis.ipynb                     full analysis and modelling
├── model/
│   └── churn_model.pkl                          saved pipeline (created by section 7)
├── churn_features.py                            engineered features, shared by notebook and API
├── app.py                                       FastAPI prediction service
├── sample_request.json                          example API payload
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.12 or later. Verified on Python 3.14.3.

**1. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv && source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the analysis

```bash
jupyter lab notebook/churn_analysis.ipynb
```

Run the cells top to bottom. The notebook is being built one section at a time — see
**Build status** above for what is currently in it. When complete it will cover data
preparation, EDA, feature engineering, model development, evaluation and interpretation,
and will write the trained pipeline to `model/churn_model.pkl`.

Note that the notebook reads the dataset via the relative path `../data/`, so it must be
run from within the `notebook/` directory (which is the default when opened as above).

## Running the API

The API requires `model/churn_model.pkl` to exist, so run the notebook through to the end
at least once first.

```bash
uvicorn app:app --reload --port 8000
```

Interactive docs are then available at `http://127.0.0.1:8000/docs`.

---

## API reference

### `POST /predict`

Accepts a single customer's attributes as JSON, applies the saved preprocessing pipeline,
and returns the churn prediction with its probability.

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Response**

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

**Invalid input** returns HTTP 422 with a field-level description of what failed.
Validation is handled by a Pydantic request model, so unknown categorical values, missing
fields, and wrong types are all rejected before reaching the model.

---

## Notes on methodology

**Data leakage.** The train/test split happens before anything is fitted. Section 1
separates the encoding *recipe* from the *fit*: a `build_preprocessor()` factory defines an
unfitted `ColumnTransformer` (so that cell learns nothing and cannot leak), and the single
`.fit` call in the section sits below the split and takes `X_train` alone. The test set is
touched only by `transform` and by shape reporting. Section 4 calls the same factory for a
fresh transformer inside a `Pipeline`, because `Pipeline` fits its steps in place rather
than cloning them.

**EDA reads the training split only.** Section 2 builds its own
`train_df = X_train.join(y_train)` and never touches the test set. Nothing in that section calls
`.fit`, so it cannot leak mechanically — but exploration is not passive. Every figure there exists to
justify a section 3 feature decision, so exploring all 7,043 rows would let the test set shape the
model through the analyst rather than through a fitted attribute, and the held-out 30% would stop
being an honest estimate. Section 2 also ends by re-verifying a snapshot of the section 1 values it
inherited, so an accidental variable rebind fails the notebook instead of quietly changing what an
earlier cell appears to say.

**Reuse on unseen data.** The preprocessor selects columns by name, uses
`handle_unknown="ignore"` so an unseen category degrades to an all-zero block instead of
raising, and carries a constant-`0.0` imputer so a never-billed customer is repaired inside
the pickled artifact rather than by a notebook cell. Section 1.13 demonstrates all of this
on a single-row payload, including a `joblib` round-trip, so the API path is proven before
`app.py` exists.

**Engineered features live in a module, not the notebook.** `churn_features.py` holds the
four features section 3 builds, and both the notebook and `app.py` import it. This is
not tidiness: `joblib` pickles a function by reference, storing its module and name
rather than its body, so a feature function defined in the notebook would be recorded as
`__main__.add_features` and `joblib.load` inside the API would fail to resolve it. With a
real module the reference is `churn_features.add_features`, the saved pipeline carries
feature engineering as its first step, and `POST /predict` can accept the raw customer
fields. The alternative — re-implementing the rules in `app.py` — would put the same logic
in two places, where the first edit to either makes the API score customers differently
from the notebook that validated it.

**Feature engineering was measured, not assumed.** Section 3 reports a cross-validated
ablation showing that three of the four engineered features change decision-tree recall by
exactly nothing, because a tree is invariant to monotone transformations and builds
conjunctions itself given depth. The exception is `is_high_risk` at `max_depth=2`, worth
+3.4 points of recall where the tree cannot afford to rebuild a four-way conjunction. The
change that actually helps is a removal: dropping `gender`, `PhoneService` and
`MultipleLines` improves recall, precision and F1 together.

**Model selection used ROC-AUC, not recall.** Recall is the primary *reported* metric, but
it cannot be the objective: flagging every customer scores recall 1.000. Tuning a
threshold for F2 is barely better — it lands at 0.11 and flags over half the base.
Section 4 therefore selects the configuration on ROC-AUC, which measures how well the
model ranks customers by risk independent of the threshold, then sets the operating point
separately, because that is a decision about how many customers the retention team can
contact rather than a modelling choice. `class_weight="balanced"` is kept for the same
reason: it does not improve the ranking, it places the default 0.50 threshold near the F1
optimum so the shipped artifact behaves sensibly out of the box.

**Class imbalance and the two benchmarks.** At 26.54% churn, a model predicting "No" for
every customer scores 73.46% accuracy while finding none of the churners. Both benchmarks
were pre-registered before any model existed, and section 5 reports the outcome of each.

*Benchmark 1 — the majority-class floor — is cleared, and clearing it shows how little
accuracy is worth here.* On test the model scores 73.88% accuracy against the baseline's
73.45%: an improvement of 0.43 of a percentage point, while finding 438 churners against
zero. Anyone comparing the two on accuracy alone would call them equivalent.

*Benchmark 2 is not cleared as originally worded.* Section 2.9's four-condition rule
(month-to-month, fibre, no OnlineSecurity, no TechSupport) was set as a bar the model "has
to beat". At matched 21.8% coverage on test, the rule reaches 50.5% recall at 61.4%
precision and the model reaches 49.6% at 60.3% — about a point behind on both, reversing
the training-set result. The two select nearly the same people (the model flags 443 of the
rule's 461), the gap is roughly five customers, and 14 of the 461 are chosen by
tie-breaking because a 71-leaf tree produces only 64 distinct probabilities. The honest
reading is that the model *matches* a hand-built rule in the rule's narrow operating
region and earns its place by extending to regions the rule cannot reach at all — 41%
coverage for 78.1% recall. Four conditions read off a cross-tab are competitive with a
tuned decision tree, which is worth knowing.

**What drives the predictions.** Four features do essentially all the work:
`Contract`, `MonthlyCharges`, `tenure` and the engineered `is_high_risk` flag reach
ROC-AUC 0.8309 in six encoded columns against the full 46-column model's 0.8319 — 99.7%
of the lift over chance. `Contract` alone reaches 0.7401, so the model is substantially a
contract-type classifier refined by price, tenure and unprotected fibre. Section 6 reports
both Gini and permutation importance, the latter on train *and* test, which is what
exposed `Dependents` and `Partner` as fit rather than signal: both carry small positive
importance on training data and none on held-out data.

**The highest-risk rules are directly actionable.** Every one of the tree's five riskiest
leaves requires month-to-month *and* unprotected fibre *and* tenure under about ten
months. Those five rules cover 7.8% of customers and 24.3% of all churners, the worst at a
92.6% churn rate, and each is expressible as a database query rather than needing the
model. Section 6.4 extracts them and verifies each reconstructed rule reproduces the
tree's own leaf assignment exactly.

**Precision or recall?** Recall, subject to a precision floor. On the test set a missed
churner costs real revenue — the 123 the model misses bill $8,172 a month, $98,065 a year,
leaving with no chance of intervention — while a false positive costs one retention
contact to somebody who was staying. Those are not symmetric. But precision cannot
collapse: contact cost scales with volume, discounts to customers who would have stayed
are pure margin loss, and over-contacting is itself a churn risk. Hence the 0.50 precision
floor applied during model selection. The exact optimum depends on retention cost and
customer value, neither of which is in this dataset, so section 5 reports a break-even
contact cost instead of inventing an ROI.

---

## Library versions

Built against pandas 3.0 and scikit-learn 1.9. Two behaviours differ from older tutorials:
pandas 3.0 gives text columns a dedicated `str` dtype rather than `object`, and
copy-on-write is the default, so chained assignment does not modify the original frame.
Exact versions are pinned in `requirements.txt`.

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

---

## Build status

| Section | Status |
|---|---|
| 1. Data understanding & preparation | ✅ Complete |
| 2. Exploratory data analysis | ✅ Complete |
| 3. Feature engineering | ✅ Complete |
| 4. Model development | ✅ Complete |
| 5. Model evaluation | Not started |
| 6. Model interpretation | Not started |
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
every customer scores 73.46% accuracy while finding none of the churners. All results are
therefore judged against that baseline rather than against zero, and recall on the churn
class is treated as the primary metric. Section 2 adds a second, harder benchmark: a crude
four-condition rule (month-to-month, fibre, no OnlineSecurity, no TechSupport) already
reaches 49.1% recall at 60.4% precision on the training set while touching 21.6% of
customers, so a trained model has to beat that to be worth deploying.

---

## Library versions

Built against pandas 3.0 and scikit-learn 1.9. Two behaviours differ from older tutorials:
pandas 3.0 gives text columns a dedicated `str` dtype rather than `object`, and
copy-on-write is the default, so chained assignment does not modify the original frame.
Exact versions are pinned in `requirements.txt`.

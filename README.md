# Customer Churn Prediction — Telco

End-to-end machine learning solution predicting telecom customer churn, from raw data
through to a REST prediction API.

**Business problem.** Customers who leave ("churn") cost far more to replace than to
retain. This project identifies customers likely to churn so a retention team can engage
them before they go.

- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 columns
- **Target:** `Churn` (Yes / No), 26.54% positive class
- **Model:** Decision Tree Classifier
- **Split:** 70:30, `random_state=42`, stratified on the target

---

## Build status

| Section | Status |
|---|---|
| 1. Data understanding & preparation | ✅ Complete |
| 2. Exploratory data analysis | Not started |
| 3. Feature engineering | Not started |
| 4. Model development | Not started |
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

**Reuse on unseen data.** The preprocessor selects columns by name, uses
`handle_unknown="ignore"` so an unseen category degrades to an all-zero block instead of
raising, and carries a constant-`0.0` imputer so a never-billed customer is repaired inside
the pickled artifact rather than by a notebook cell. Section 1.13 demonstrates all of this
on a single-row payload, including a `joblib` round-trip, so the API path is proven before
`app.py` exists.

**Class imbalance.** At 26.54% churn, a model predicting "No" for every customer scores
73.46% accuracy while being useless. All results are therefore judged against that
baseline rather than against zero, and recall on the churn class is treated as the primary
metric — see the notebook's evaluation section for the reasoning.

---

## Library versions

Built against pandas 3.0 and scikit-learn 1.9. Two behaviours differ from older tutorials:
pandas 3.0 gives text columns a dedicated `str` dtype rather than `object`, and
copy-on-write is the default, so chained assignment does not modify the original frame.
Exact versions are pinned in `requirements.txt`.

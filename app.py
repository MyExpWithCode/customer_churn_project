"""FastAPI service exposing the Telco customer-churn model.

Run from the project root:

    uvicorn app:app --reload --port 8000

Then POST a customer record to ``/predict``, or open ``/docs`` for the generated
interactive documentation.

    curl -X POST http://127.0.0.1:8000/predict \\
      -H "Content-Type: application/json" -d @sample_request.json

The endpoint receives the 19 raw customer fields and nothing else. Feature
engineering and encoding happen **inside** the saved pipeline: its first step is
``FunctionTransformer(churn_features.add_features)``, which derives the four
engineered columns, and its second is the ``ColumnTransformer`` fitted on the
training split in section 4.6 of the notebook. Nothing here re-implements any of
that, which is the point -- the API cannot drift from the model that was
evaluated, because it *is* that model.

Two consequences worth knowing:

* The service needs no training data at runtime. It needs the ~22 KB artifact
  and ``churn_features.py``, because joblib stores a *reference* to
  ``churn_features.add_features`` rather than its code.
* Three of the 19 fields -- ``gender``, ``PhoneService`` and ``MultipleLines``
  -- are discarded by the pipeline, which section 3 dropped on the evidence of a
  cross-validated ablation. They are still required here, because a customer
  record is a customer record, and because a future model may want them.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import churn_features as cf

#: Resolved from this file rather than the working directory: uvicorn runs from
#: the project root and the notebook runs from notebook/, and both must find it.
MODEL_PATH = Path(__file__).resolve().parent / "model" / "churn_model.pkl"

#: The operating point section 4.4 shipped. F1 on the training folds peaks at
#: 0.49, so the conventional 0.50 was left alone rather than tuned.
DECISION_THRESHOLD = 0.50

#: Every boundary is a number an earlier section established, not an invention:
#: the training base rate (2.0), the shipped threshold (4.4), and the level above
#: which section 6.3 found ten distinct high-risk leaves.
TRAINING_BASE_RATE = 0.2653
CRITICAL_PROBABILITY = 0.80

#: Fields the fitted encoder never sees, so their allowed values cannot be
#: cross-checked against the model at startup. They come from the data
#: dictionary instead.
UNCHECKABLE_FIELDS = ("gender", "PhoneService", "MultipleLines")

YesNo = Literal["No", "Yes"]
AddOn = Literal["No", "No internet service", "Yes"]


class CustomerRequest(BaseModel):
    """One customer record, exactly as the data dictionary defines it.

    ``extra="forbid"`` is deliberate: without it a misspelled field name would
    produce a confusing "missing field" error while the typo was silently
    ignored. With it, the caller is told which key it did not recognise.

    Numeric bounds are generous rather than set to the observed training range.
    Section 3 made ``tenure_bucket`` open-topped on purpose so a customer with
    longer tenure than anyone in the training data is scored rather than
    rejected, and rejecting them here would undo that.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 94.35, "TotalCharges": 283.05,
            }
        },
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=1200, description="Whole months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(gt=0, le=10_000, description="Current monthly bill")
    TotalCharges: float = Field(ge=0, le=1_000_000, description="Lifetime billed to date")


class PredictionResponse(BaseModel):
    """The two fields the brief requires, plus two that explain them."""

    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(description="Probability at or above which the label is Yes")
    risk_band: Literal["low", "moderate", "high", "critical"]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    encoded_features: int | None = None
    threshold: float


def risk_band(probability: float) -> str:
    """Bucket a probability using thresholds the notebook established."""
    if probability >= CRITICAL_PROBABILITY:
        return "critical"
    if probability >= DECISION_THRESHOLD:
        return "high"
    if probability >= TRAINING_BASE_RATE:
        return "moderate"
    return "low"


def declared_choices(field: str) -> tuple:
    """The values ``CustomerRequest`` accepts for one field."""
    return get_args(CustomerRequest.model_fields[field].annotation)


def validate_schema_against_model(pipeline) -> dict:
    """Refuse to serve a model whose categories the schema does not match.

    This is the check that earns its keep. A model retrained on data containing
    a new payment method, or an edit to the literals above, would otherwise
    produce an API that rejects perfectly valid customers with a 422 -- and
    looks like it is working. Comparing the two at startup turns that into a
    boot failure.

    The three fields in ``UNCHECKABLE_FIELDS`` are dropped by the pipeline, so
    the encoder has no opinion about them and they are reported as such.
    """
    steps = [name for name, _ in pipeline.steps]
    if steps != ["features", "preprocess", "tree"]:
        raise RuntimeError(f"unexpected pipeline steps: {steps}")

    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    model_categories = dict(zip(encoder.feature_names_in_, encoder.categories_))

    mismatches = []
    checked = 0
    for field in CustomerRequest.model_fields:
        if field not in model_categories:
            continue
        declared = set(str(value) for value in declared_choices(field))
        fitted = set(str(value) for value in model_categories[field])
        if declared != fitted:
            mismatches.append(
                f"{field}: schema {sorted(declared)} vs model {sorted(fitted)}"
            )
        checked += 1
    if mismatches:
        raise RuntimeError(
            "the request schema does not match the fitted model: " + "; ".join(mismatches)
        )
    return {
        "checked": checked,
        "unverifiable": [f for f in UNCHECKABLE_FIELDS if f in CustomerRequest.model_fields],
        "encoded_features": len(pipeline.named_steps["preprocess"].get_feature_names_out()),
    }


def load_pipeline():
    """Load the saved artifact, or explain precisely what is missing."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"no model at {MODEL_PATH}. Run notebook/churn_analysis.ipynb through "
            "section 7.1, which writes it."
        )
    return joblib.load(MODEL_PATH)


STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once per process, and validate it before serving."""
    pipeline = load_pipeline()
    STATE["schema_check"] = validate_schema_against_model(pipeline)
    STATE["pipeline"] = pipeline
    yield
    STATE.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    version="1.0.0",
    summary="Predicts whether a telecom customer is likely to churn.",
    description=(
        "Scores a single customer record with a decision tree trained on the IBM "
        "Telco Customer Churn dataset. Feature engineering and encoding are carried "
        "inside the saved pipeline, so the endpoint takes raw customer fields."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": app.title, "version": app.version, "docs": "/docs",
            "predict": "POST /predict"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus whether the model actually loaded."""
    loaded = "pipeline" in STATE
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        encoded_features=STATE.get("schema_check", {}).get("encoded_features"),
        threshold=DECISION_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRequest) -> PredictionResponse:
    """Score one customer.

    Pydantic has already rejected anything malformed with a 422 naming the
    offending field, so by this point the payload is a valid customer record.

    The label is derived from the probability and ``DECISION_THRESHOLD``
    explicitly rather than by calling ``pipeline.predict``. Both give the same
    answer at 0.50, but doing it here means the ``threshold`` in the response is
    demonstrably the one that produced the label.
    """
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    frame = pd.DataFrame([customer.model_dump()])
    try:
        probability = float(pipeline.predict_proba(frame)[0, 1])
    except Exception as error:  # pragma: no cover - a scoring failure is a bug, not input
        raise HTTPException(status_code=500, detail=f"scoring failed: {error}") from error

    return PredictionResponse(
        prediction="Yes" if probability >= DECISION_THRESHOLD else "No",
        churn_probability=round(probability, 4),
        threshold=DECISION_THRESHOLD,
        risk_band=risk_band(probability),
    )

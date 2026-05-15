"""
Shared ML configuration imported by the ETL pipeline (transform.py, pipeline.py)
and model training (train.py).

One change here propagates everywhere automatically.  Do NOT redefine these
values inline in other files — divergence causes silent data bugs.

INPATIENT_CLASSES
    Encounter classes treated as inpatient admissions throughout the pipeline.
    Used by: add_readmission_label (label generation), build_encounter_ml_features
    (feature scope), pipeline.py (DB load filter).  All three MUST agree or the
    loaded encounters, their features, and their labels won't match.

SPLIT_TEST_SIZE / SPLIT_RANDOM_STATE
    Control the patient-level train/test split.  transform.py uses these to fit
    LOINC population statistics on training encounters only; train.py uses them
    to define the actual split.  If they diverge, the stored z-scores are fitted
    on a different set than the model trains on — silent leakage.
"""

INPATIENT_CLASSES: frozenset[str] = frozenset({"inpatient", "IMP", "emergency", "EMER"})

SPLIT_TEST_SIZE: float = 0.2
SPLIT_RANDOM_STATE: int = 42

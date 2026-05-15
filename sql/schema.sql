-- MediSight star schema
-- Run once to initialize the database

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Dimension tables ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_patient (
    patient_id      VARCHAR PRIMARY KEY,
    first_name      VARCHAR,
    last_name       VARCHAR,
    birth_date      DATE,
    gender          VARCHAR(10),
    race            VARCHAR(50),
    ethnicity       VARCHAR(50),
    address_city    VARCHAR,
    address_state   VARCHAR,
    zip_code        VARCHAR(10),
    insurance_type  VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_diagnosis (
    diagnosis_id    VARCHAR PRIMARY KEY,
    encounter_id    VARCHAR,
    patient_id      VARCHAR REFERENCES dim_patient(patient_id),
    icd_code        VARCHAR(20),
    icd_description VARCHAR(500),
    onset_date      DATE,
    abatement_date  DATE,
    clinical_status VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_medication (
    medication_id   VARCHAR PRIMARY KEY,
    encounter_id    VARCHAR,
    patient_id      VARCHAR REFERENCES dim_patient(patient_id),
    medication_code VARCHAR(50),
    medication_name VARCHAR(500),
    start_date      DATE,
    end_date        DATE,
    status          VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_lab_result (
    lab_id          VARCHAR PRIMARY KEY,
    encounter_id    VARCHAR,
    patient_id      VARCHAR REFERENCES dim_patient(patient_id),
    category        VARCHAR(50),    -- "laboratory" or "vital-signs"
    loinc_code      VARCHAR(20),
    lab_name        VARCHAR(500),
    value_numeric   FLOAT,
    value_string    VARCHAR(200),
    unit            VARCHAR(50),
    reference_low   FLOAT,
    reference_high  FLOAT,
    observation_date TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ─── Fact table ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_encounters (
    encounter_id            VARCHAR PRIMARY KEY,
    patient_id              VARCHAR REFERENCES dim_patient(patient_id),
    encounter_class         VARCHAR(50),   -- inpatient, outpatient, emergency
    encounter_type          VARCHAR(200),
    admission_date          TIMESTAMP,
    discharge_date          TIMESTAMP,
    length_of_stay_days     FLOAT,
    readmitted_30d          BOOLEAN,       -- ML target 1
    primary_diagnosis_code  VARCHAR(20),
    primary_diagnosis_desc  VARCHAR(500),
    provider_org            VARCHAR(200),
    total_claim_cost        FLOAT,
    payer_coverage          FLOAT,
    created_at              TIMESTAMP DEFAULT NOW()
);

-- ─── ML features table (populated after feature engineering) ─────────────────

CREATE TABLE IF NOT EXISTS ml_features (
    patient_id              VARCHAR PRIMARY KEY REFERENCES dim_patient(patient_id),
    age                     INT,
    age_bucket              VARCHAR(20),   -- pediatric / adult / middle / senior
    gender_encoded          INT,           -- 0/1
    comorbidity_count       INT,
    chronic_condition_count INT,
    medication_count        INT,
    prior_admissions_12m    INT,
    prior_admissions_total  INT,
    days_since_last_visit   INT,
    avg_lab_deviation       FLOAT,         -- mean z-score across labs vs normal range
    insurance_risk_tier     INT,           -- 1-4 based on insurance type
    total_encounters        INT,
    last_los_days           FLOAT,
    updated_at              TIMESTAMP DEFAULT NOW()
);

-- ─── Encounter-level ML features (one row per inpatient/emergency encounter) ──

CREATE TABLE IF NOT EXISTS ml_encounter_features (
    encounter_id                VARCHAR PRIMARY KEY REFERENCES fact_encounters(encounter_id),
    patient_id                  VARCHAR REFERENCES dim_patient(patient_id),
    readmitted_30d              BOOLEAN,        -- ML target

    -- Patient demographics at time of admission
    age_at_admission            INT,
    gender_encoded              INT,
    insurance_risk_tier         INT,

    -- This encounter
    encounter_class_encoded     INT,            -- 0=outpatient 1=emergency 2=inpatient
    length_of_stay_days         FLOAT,
    total_claim_cost            FLOAT,

    -- Diagnoses during this encounter
    num_diagnoses_this_visit    INT,
    has_heart_failure           BOOLEAN,
    has_diabetes                BOOLEAN,
    has_copd                    BOOLEAN,
    has_ckd                     BOOLEAN,
    has_hypertension            BOOLEAN,

    -- Labs during this encounter
    num_labs_this_visit         INT,
    num_abnormal_labs_this_visit INT,
    avg_lab_deviation_this_visit FLOAT,

    -- Medications during this encounter
    num_meds_this_visit         INT,

    -- Patient history BEFORE this encounter (no leakage)
    prior_admissions_6m         INT,
    prior_admissions_12m        INT,
    prior_admissions_total      INT,
    days_since_previous_visit   INT,
    is_first_admission          INT,            -- 1 = no prior inpatient history
    comorbidity_count_prior     INT,

    created_at                  TIMESTAMP DEFAULT NOW()
);

-- ─── Migrations (idempotent column additions and constraint additions) ────────
-- All statements use IF NOT EXISTS / DO NOTHING patterns — safe to re-run.
ALTER TABLE ml_encounter_features ADD COLUMN IF NOT EXISTS is_first_admission INT;

-- FK constraints linking dimension encounter_id columns to fact_encounters.
-- NOTE: PostgreSQL has no ADD CONSTRAINT IF NOT EXISTS syntax (unlike ADD COLUMN).
-- We use DO $$ ... EXCEPTION WHEN duplicate_object THEN NULL END $$ blocks,
-- which are the idiomatic idempotent pattern for constraint additions in PostgreSQL.
-- Load order in pipeline.py (patients → encounters → diagnoses/labs/meds) satisfies
-- the FK dependency; truncate_all() uses one TRUNCATE statement so PostgreSQL handles
-- inter-table FK dependencies atomically.
DO $$ BEGIN
    ALTER TABLE dim_diagnosis ADD CONSTRAINT fk_diag_encounter
        FOREIGN KEY (encounter_id) REFERENCES fact_encounters(encounter_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE dim_medication ADD CONSTRAINT fk_med_encounter
        FOREIGN KEY (encounter_id) REFERENCES fact_encounters(encounter_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE dim_lab_result ADD CONSTRAINT fk_lab_encounter
        FOREIGN KEY (encounter_id) REFERENCES fact_encounters(encounter_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ─── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_encounters_patient ON fact_encounters(patient_id);
CREATE INDEX IF NOT EXISTS idx_encounters_admission ON fact_encounters(admission_date);
CREATE INDEX IF NOT EXISTS idx_diagnosis_patient ON dim_diagnosis(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnosis_icd ON dim_diagnosis(icd_code);
CREATE INDEX IF NOT EXISTS idx_lab_patient ON dim_lab_result(patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_encounter ON dim_lab_result(encounter_id);
CREATE INDEX IF NOT EXISTS idx_medication_patient ON dim_medication(patient_id);

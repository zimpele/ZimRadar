-- Store raw feature vector + SHAP values alongside each risk assessment.
ALTER TABLE risk_assessments
    ADD COLUMN IF NOT EXISTS features_json JSONB,
    ADD COLUMN IF NOT EXISTS shap_values   JSONB,
    ADD COLUMN IF NOT EXISTS confidence    FLOAT;

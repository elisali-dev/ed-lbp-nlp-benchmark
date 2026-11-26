-- Extract recorded ICD diagnoses.

SELECT
    stay_id,
    icd_code,
    icd_version
FROM `physionet-data.mimiciv_ed.diagnosis`;
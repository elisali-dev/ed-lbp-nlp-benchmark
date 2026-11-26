-- Check diagnosis completeness and ICD version distribution.

SELECT
    icd_version,
    COUNT(*) AS diagnosis_count,
    COUNT(DISTINCT stay_id) AS encounter_count,
    COUNTIF(icd_code IS NULL) AS missing_code_count,
    COUNTIF(TRIM(icd_code) = '') AS empty_code_count
FROM `physionet-data.mimiciv_ed.diagnosis`
GROUP BY icd_version
ORDER BY icd_version;
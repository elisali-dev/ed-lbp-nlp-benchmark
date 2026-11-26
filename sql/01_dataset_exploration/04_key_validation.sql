-- Check identifier completeness and encounter-level uniqueness.

SELECT
    'edstays' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT stay_id) AS unique_stays,
    COUNT(DISTINCT subject_id) AS unique_patients,
    COUNTIF(stay_id IS NULL) AS missing_stay_id,
    COUNTIF(subject_id IS NULL) AS missing_subject_id
FROM `physionet-data.mimiciv_ed.edstays`

UNION ALL

SELECT
    'triage',
    COUNT(*),
    COUNT(DISTINCT stay_id),
    COUNT(DISTINCT subject_id),
    COUNTIF(stay_id IS NULL),
    COUNTIF(subject_id IS NULL)
FROM `physionet-data.mimiciv_ed.triage`

UNION ALL

SELECT
    'diagnosis',
    COUNT(*),
    COUNT(DISTINCT stay_id),
    COUNT(DISTINCT subject_id),
    COUNTIF(stay_id IS NULL),
    COUNTIF(subject_id IS NULL)
FROM `physionet-data.mimiciv_ed.diagnosis`;
-- Count records in each source table.

SELECT
    'edstays' AS table_name,
    COUNT(*) AS row_count
FROM `physionet-data.mimiciv_ed.edstays`

UNION ALL

SELECT
    'triage',
    COUNT(*)
FROM `physionet-data.mimiciv_ed.triage`

UNION ALL

SELECT
    'diagnosis',
    COUNT(*)
FROM `physionet-data.mimiciv_ed.diagnosis`

ORDER BY table_name;
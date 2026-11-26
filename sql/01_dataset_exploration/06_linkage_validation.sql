-- Check encounter linkage and patient identifier consistency.

SELECT
    'triage' AS source_table,
    COUNT(*) AS source_rows,
    COUNTIF(e.stay_id IS NULL) AS unmatched_rows,
    COUNTIF(
        e.stay_id IS NOT NULL
        AND t.subject_id IS DISTINCT FROM e.subject_id
    ) AS subject_id_mismatches
FROM `physionet-data.mimiciv_ed.triage` AS t
LEFT JOIN `physionet-data.mimiciv_ed.edstays` AS e
    ON t.stay_id = e.stay_id

UNION ALL

SELECT
    'diagnosis',
    COUNT(*),
    COUNTIF(e.stay_id IS NULL),
    COUNTIF(
        e.stay_id IS NOT NULL
        AND d.subject_id IS DISTINCT FROM e.subject_id
    )
FROM `physionet-data.mimiciv_ed.diagnosis` AS d
LEFT JOIN `physionet-data.mimiciv_ed.edstays` AS e
    ON d.stay_id = e.stay_id;
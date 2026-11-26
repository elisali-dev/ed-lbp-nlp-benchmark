-- Check chief complaint availability.

SELECT
    COUNT(*) AS total_encounters,
    COUNTIF(chiefcomplaint IS NULL) AS missing_count,
    COUNTIF(TRIM(chiefcomplaint) = '') AS empty_count,
    COUNTIF(TRIM(chiefcomplaint) != '') AS available_count
FROM `physionet-data.mimiciv_ed.triage`;
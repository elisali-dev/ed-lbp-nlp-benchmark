-- Examine the distribution of ED encounters per patient.

SELECT
    subject_id,
    COUNT(DISTINCT stay_id) AS encounter_count
FROM `physionet-data.mimiciv_ed.edstays`
WHERE subject_id IS NOT NULL
    AND stay_id IS NOT NULL
GROUP BY subject_id
ORDER BY encounter_count DESC;
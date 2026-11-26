-- Examine combined chief complaint and diagnosis availability.

SELECT
    CASE
        WHEN t.chiefcomplaint IS NULL
            OR TRIM(t.chiefcomplaint) = ''
        THEN 'Unavailable'
        ELSE 'Available'
    END AS chief_complaint_status,

    CASE
        WHEN d.stay_id IS NULL
        THEN 'Unavailable'
        ELSE 'Available'
    END AS diagnosis_status,

    COUNT(*) AS encounter_count

FROM `physionet-data.mimiciv_ed.edstays` AS e

LEFT JOIN `physionet-data.mimiciv_ed.triage` AS t
    ON e.stay_id = t.stay_id

LEFT JOIN (
    SELECT DISTINCT stay_id
    FROM `physionet-data.mimiciv_ed.diagnosis`
) AS d
    ON e.stay_id = d.stay_id

GROUP BY
    chief_complaint_status,
    diagnosis_status

ORDER BY
    chief_complaint_status,
    diagnosis_status;
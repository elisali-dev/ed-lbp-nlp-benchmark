-- Extract encounter and patient identifiers.

SELECT
    stay_id,
    subject_id
FROM `physionet-data.mimiciv_ed.edstays`;
-- Extract raw chief complaint text.

SELECT
    stay_id,
    chiefcomplaint
FROM `physionet-data.mimiciv_ed.triage`;
-- Check uniqueness of the diagnosis composite key.

SELECT
    stay_id,
    seq_num,
    COUNT(*) AS row_count
FROM `physionet-data.mimiciv_ed.diagnosis`
GROUP BY stay_id, seq_num
HAVING COUNT(*) > 1;
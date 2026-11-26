-- Inspect the schemas of the three source tables.

SELECT
    table_name,
    ordinal_position,
    column_name,
    data_type,
    is_nullable
FROM
    `physionet-data.mimiciv_ed.INFORMATION_SCHEMA.COLUMNS`
WHERE
    table_name IN ('edstays', 'triage', 'diagnosis')
ORDER BY
    table_name,
    ordinal_position;
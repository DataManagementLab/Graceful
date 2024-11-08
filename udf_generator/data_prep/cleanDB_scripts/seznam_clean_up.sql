DELETE
FROM "client"
WHERE "client_id" IS NULL
   OR "client_id" = 0;
DELETE
FROM "client"
WHERE "kraj" IS NULL
   OR "obor" IS NULL;
DELETE
FROM "dobito"
WHERE "client_id" IS NULL
   OR "month_year_datum_transakce" IS NULL
   OR "sluzba" IS NULL
   OR "kc_dobito" IS NULL;
UPDATE "dobito"
SET "client_id" = "client_id" + 1,
    "kc_dobito" = "kc_dobito" + 1;
DELETE
FROM "probehnuto"
WHERE "client_id" IS NULL
   OR "month_year_datum_transakce" IS NULL
   OR "sluzba" IS NULL
   OR "kc_proklikano" IS NULL;
UPDATE "probehnuto"
SET "client_id"     = "client_id" + 1,
    "kc_proklikano" = "kc_proklikano" + 1;
DELETE
FROM "probehnuto_mimo_penezenku"
WHERE "client_id" IS NULL
   OR "client_id" = 0;
DELETE
FROM "probehnuto_mimo_penezenku"
WHERE "Month_Year" IS NULL
   OR "probehla_inzerce_mimo_penezenku" IS NULL;

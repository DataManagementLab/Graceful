DELETE
FROM "atom"
WHERE "atomid" IS NULL
   OR "drug" IS NULL
   OR "atomtype" IS NULL
   OR "charge" IS NULL
   OR "name" IS NULL;
DELETE
FROM "canc"
WHERE "drug_id" IS NULL
   OR "col_class" IS NULL;
DELETE
FROM "sbond_1"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "sbond_1"
WHERE "drug" IS NULL
   OR "atomid" IS NULL
   OR "atomid_2" IS NULL;
DELETE
FROM "sbond_2"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "sbond_2"
WHERE "drug" IS NULL
   OR "atomid" IS NULL
   OR "atomid_2" IS NULL;
DELETE
FROM "sbond_3"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "sbond_3"
WHERE "drug" IS NULL
   OR "atomid" IS NULL
   OR "atomid_2" IS NULL;
DELETE
FROM "sbond_7"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "sbond_7"
WHERE "drug" IS NULL
   OR "atomid" IS NULL
   OR "atomid_2" IS NULL;

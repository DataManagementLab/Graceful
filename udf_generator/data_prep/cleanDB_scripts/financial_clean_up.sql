DELETE FROM "account" WHERE "account_id" IS NULL OR "account_id" = 0;
DELETE FROM "account" WHERE "district_id" IS NULL OR "frequency" IS NULL OR "date" IS NULL;
UPDATE "account" SET "district_id" = "district_id" + 1;
DELETE FROM "card" WHERE "card_id" IS NULL OR "card_id" = 0;
DELETE FROM "card" WHERE "disp_id" IS NULL OR "type" IS NULL OR "issued" IS NULL;
UPDATE "card" SET "disp_id" = "disp_id" + 1;
DELETE FROM "client" WHERE "client_id" IS NULL OR "client_id" = 0;
DELETE FROM "client" WHERE "gender" IS NULL OR "birth_date" IS NULL OR "district_id" IS NULL;
UPDATE "client" SET "district_id" = "district_id" + 1;
DELETE FROM "disp" WHERE "disp_id" IS NULL OR "disp_id" = 0;
DELETE FROM "disp" WHERE "client_id" IS NULL OR "account_id" IS NULL OR "type" IS NULL;
UPDATE "disp" SET "client_id" = "client_id" + 1, "account_id" = "account_id" + 1;
DELETE FROM "district" WHERE "district_id" IS NULL OR "district_id" = 0;
DELETE
FROM "district"
WHERE "A2" IS NULL
   OR "A3" IS NULL
   OR "A4" IS NULL
   OR "A5" IS NULL
   OR "A6" IS NULL
   OR "A7" IS NULL
   OR "A8" IS NULL
   OR "A9" IS NULL
   OR "A10" IS NULL
   OR "A11" IS NULL
   OR "A12" IS NULL
   OR "A13" IS NULL
   OR "A14" IS NULL
   OR "A15" IS NULL
   OR "A16" IS NULL;
UPDATE "district"
SET "A4"  = "A4" + 1,
    "A5"  = "A5" + 1,
    "A6"  = "A6" + 1,
    "A7"  = "A7" + 1,
    "A8"  = "A8" + 1,
    "A9"  = "A9" + 1,
    "A10" = "A10" + 1,
    "A11" = "A11" + 1,
    "A12" = "A12" + 1,
    "A14" = "A14" + 1,
    "A15" = "A15" + 1,
    "A16" = "A16" + 1;
DELETE FROM "loan" WHERE "loan_id" IS NULL OR "loan_id" = 0;
DELETE
FROM "loan"
WHERE "account_id" IS NULL
   OR "date" IS NULL
   OR "amount" IS NULL
   OR "duration" IS NULL
   OR "payments" IS NULL
   OR "status" IS NULL;
UPDATE "loan"
SET "account_id" = "account_id" + 1,
    "amount"     = "amount" + 1,
    "duration"   = "duration" + 1,
    "payments"   = "payments" + 1;
DELETE
FROM "orders"
WHERE "order_id" IS NULL
   OR "order_id" = 0;
DELETE
FROM "orders"
WHERE "account_id" IS NULL
   OR "bank_to" IS NULL
   OR "account_to" IS NULL
   OR "amount" IS NULL
   OR "k_symbol" IS NULL;
UPDATE "orders"
SET "account_id" = "account_id" + 1,
    "account_to" = "account_to" + 1,
    "amount"     = "amount" + 1;
DELETE FROM "trans" WHERE "trans_id" IS NULL OR "trans_id" = 0;
DELETE FROM "trans" WHERE "account_id" IS NULL OR "date" IS NULL OR "type" IS NULL OR "operation" IS NULL OR "amount" IS NULL OR "balance" IS NULL;
UPDATE "trans" SET "account_id" = "account_id" + 1, "amount" = "amount" + 1, "balance" = "balance" + 1;

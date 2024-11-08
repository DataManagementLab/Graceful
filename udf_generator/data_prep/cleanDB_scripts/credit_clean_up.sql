DELETE FROM "category" WHERE "category_no" IS NULL OR "category_no" = 0;
DELETE FROM "category" WHERE "category_desc" IS NULL;
DELETE FROM "charge" WHERE "charge_no" IS NULL OR "charge_no" = 0;
DELETE
FROM "charge"
WHERE "member_no" IS NULL
   OR "provider_no" IS NULL
   OR "category_no" IS NULL
   OR "charge_dt" IS NULL
   OR "charge_amt" IS NULL
   OR "statement_no" IS NULL;
UPDATE "charge"
SET "member_no"    = "member_no" + 1,
    "provider_no"  = "provider_no" + 1,
    "category_no"  = "category_no" + 1,
    "charge_amt"   = "charge_amt" + 1,
    "statement_no" = "statement_no" + 1;
DELETE FROM "corporation" WHERE "corp_no" IS NULL OR "corp_no" = 0;
DELETE FROM "corporation" WHERE "corp_name" IS NULL OR "street" IS NULL OR "city" IS NULL OR "expr_dt" IS NULL OR "region_no" IS NULL;
UPDATE "corporation" SET "region_no" = "region_no" + 1;
DELETE FROM "member" WHERE "member_no" IS NULL OR "member_no" = 0;
DELETE
FROM "member"
WHERE "lastname" IS NULL
   OR "firstname" IS NULL
   OR "street" IS NULL
   OR "city" IS NULL
   OR "issue_dt" IS NULL
   OR "expr_dt" IS NULL
   OR "region_no" IS NULL
   OR "prev_balance" IS NULL
   OR "curr_balance" IS NULL;
UPDATE "member"
SET "region_no"    = "region_no" + 1,
    "prev_balance" = "prev_balance" + 1,
    "curr_balance" = "curr_balance" + 1;
DELETE
FROM "payment"
WHERE "payment_no" IS NULL
   OR "payment_no" = 0;
DELETE
FROM "payment"
WHERE "member_no" IS NULL
   OR "payment_dt" IS NULL
   OR "payment_amt" IS NULL
   OR "statement_no" IS NULL;
UPDATE "payment"
SET "member_no"    = "member_no" + 1,
    "payment_amt"  = "payment_amt" + 1,
    "statement_no" = "statement_no" + 1;
DELETE FROM "provider" WHERE "provider_no" IS NULL OR "provider_no" = 0;
DELETE FROM "provider" WHERE "provider_name" IS NULL OR "street" IS NULL OR "city" IS NULL OR "issue_dt" IS NULL OR "expr_dt" IS NULL OR "region_no" IS NULL;
UPDATE "provider" SET "region_no" = "region_no" + 1;
DELETE FROM "region" WHERE "region_no" IS NULL OR "region_no" = 0;
DELETE FROM "region" WHERE "region_name" IS NULL OR "street" IS NULL OR "city" IS NULL;
DELETE FROM "statement" WHERE "statement_no" IS NULL OR "statement_no" = 0;
DELETE
FROM "statement"
WHERE "member_no" IS NULL
   OR "statement_dt" IS NULL
   OR "due_dt" IS NULL
   OR "statement_amt" IS NULL;
UPDATE "statement"
SET "member_no"     = "member_no" + 1,
    "statement_amt" = "statement_amt" + 1;

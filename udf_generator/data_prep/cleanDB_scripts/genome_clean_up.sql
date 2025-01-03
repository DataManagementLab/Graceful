DELETE FROM "ATT_CLASSES" WHERE "ATT_CLASS_ID" IS NULL OR "ATT_CLASS_ID" = 0;
DELETE FROM "ATT_CLASSES" WHERE "ATT_CLASS" IS NULL;
DELETE FROM "IMG_OBJ" WHERE "IMG_ID" IS NULL OR "IMG_ID" = 0 OR "OBJ_SAMPLE_ID" IS NULL OR "OBJ_SAMPLE_ID" = 0;
DELETE FROM "IMG_OBJ" WHERE "OBJ_CLASS_ID" IS NULL OR "X" IS NULL OR "Y" IS NULL OR "W" IS NULL OR "H" IS NULL;
UPDATE "IMG_OBJ" SET "OBJ_CLASS_ID" = "OBJ_CLASS_ID" + 1, "X" = "X" + 1, "Y" = "Y" + 1, "W" = "W" + 1, "H" = "H" + 1;
DELETE FROM "IMG_OBJ_ATT" WHERE "IMG_ID" IS NULL OR "IMG_ID" = 0 OR "ATT_CLASS_ID" IS NULL OR "ATT_CLASS_ID" = 0 OR "OBJ_SAMPLE_ID" IS NULL OR "OBJ_SAMPLE_ID" = 0;
DELETE FROM "IMG_REL" WHERE "IMG_ID" IS NULL OR "IMG_ID" = 0 OR "PRED_CLASS_ID" IS NULL OR "PRED_CLASS_ID" = 0 OR "OBJ1_SAMPLE_ID" IS NULL OR "OBJ1_SAMPLE_ID" = 0 OR "OBJ2_SAMPLE_ID" IS NULL OR "OBJ2_SAMPLE_ID" = 0;
DELETE FROM "OBJ_CLASSES" WHERE "OBJ_CLASS_ID" IS NULL OR "OBJ_CLASS_ID" = 0;
DELETE FROM "OBJ_CLASSES" WHERE "OBJ_CLASS" IS NULL;
DELETE FROM "PRED_CLASSES" WHERE "PRED_CLASS_ID" IS NULL OR "PRED_CLASS_ID" = 0;
DELETE FROM "PRED_CLASSES" WHERE "PRED_CLASS" IS NULL;

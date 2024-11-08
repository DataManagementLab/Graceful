DELETE FROM "key" WHERE "store_nbr" IS NULL OR "store_nbr" = 0;
DELETE FROM "key" WHERE "station_nbr" IS NULL;
UPDATE "key" SET "station_nbr" = "station_nbr" + 1;
DELETE FROM "station" WHERE "station_nbr" IS NULL OR "station_nbr" = 0;
DELETE FROM "train" WHERE "store_nbr" IS NULL OR "store_nbr" = 0 OR "item_nbr" IS NULL OR "item_nbr" = 0;
DELETE FROM "train" WHERE "date" IS NULL OR "units" IS NULL;
UPDATE "train" SET "units" = "units" + 1;

DELETE FROM "customer" WHERE "c_custkey" IS NULL OR "c_custkey" = 0;
DELETE FROM "customer" WHERE "c_name" IS NULL OR "c_address" IS NULL OR "c_city" IS NULL OR "c_nation" IS NULL OR "c_region" IS NULL OR "c_phone" IS NULL OR "c_mktsegment" IS NULL;
DELETE
FROM "dim_date"
WHERE "d_datekey" IS NULL
   OR "d_datekey" = 0;
DELETE
FROM "dim_date"
WHERE "d_date" IS NULL
   OR "d_dayofweek" IS NULL
   OR "d_month" IS NULL
   OR "d_year" IS NULL
   OR "d_yearmonthnum" IS NULL
   OR "d_yearmonth" IS NULL
   OR "d_daynuminweek" IS NULL
   OR "d_daynuminmonth" IS NULL
   OR "d_daynuminyear" IS NULL
   OR "d_monthnuminyear" IS NULL
   OR "d_weeknuminyear" IS NULL
   OR "d_sellingseason" IS NULL
   OR "d_lastdayinweekfl" IS NULL
   OR "d_lastdayinmonthfl" IS NULL
   OR "d_holidayfl" IS NULL
   OR "d_weekdayfl" IS NULL;
UPDATE "dim_date"
SET "d_year"             = "d_year" + 1,
    "d_yearmonthnum"     = "d_yearmonthnum" + 1,
    "d_daynuminweek"     = "d_daynuminweek" + 1,
    "d_daynuminmonth"    = "d_daynuminmonth" + 1,
    "d_daynuminyear"     = "d_daynuminyear" + 1,
    "d_monthnuminyear"   = "d_monthnuminyear" + 1,
    "d_weeknuminyear"    = "d_weeknuminyear" + 1,
    "d_lastdayinweekfl"  = "d_lastdayinweekfl" + 1,
    "d_lastdayinmonthfl" = "d_lastdayinmonthfl" + 1,
    "d_holidayfl"        = "d_holidayfl" + 1,
    "d_weekdayfl"        = "d_weekdayfl" + 1;
DELETE
FROM "lineorder"
WHERE "lo_orderkey" IS NULL
   OR "lo_linenumber" IS NULL
   OR "lo_custkey" IS NULL
   OR "lo_partkey" IS NULL
   OR "lo_suppkey" IS NULL
   OR "lo_orderdate" IS NULL
   OR "lo_orderpriority" IS NULL
   OR "lo_shippriority" IS NULL
   OR "lo_quantity" IS NULL
   OR "lo_extendedprice" IS NULL
   OR "lo_ordertotalprice" IS NULL
   OR "lo_discount" IS NULL
   OR "lo_revenue" IS NULL
   OR "lo_supplycost" IS NULL
   OR "lo_tax" IS NULL
   OR "lo_commitdate" IS NULL
   OR "lo_shipmode" IS NULL;
UPDATE "lineorder"
SET "lo_orderkey"        = "lo_orderkey" + 1,
    "lo_linenumber"      = "lo_linenumber" + 1,
    "lo_custkey"         = "lo_custkey" + 1,
    "lo_partkey"         = "lo_partkey" + 1,
    "lo_suppkey"         = "lo_suppkey" + 1,
    "lo_orderdate"       = "lo_orderdate" + 1,
    "lo_shippriority"    = "lo_shippriority" + 1,
    "lo_quantity"        = "lo_quantity" + 1,
    "lo_extendedprice"   = "lo_extendedprice" + 1,
    "lo_ordertotalprice" = "lo_ordertotalprice" + 1,
    "lo_discount"        = "lo_discount" + 1,
    "lo_revenue"         = "lo_revenue" + 1,
    "lo_supplycost"      = "lo_supplycost" + 1,
    "lo_tax"             = "lo_tax" + 1,
    "lo_commitdate"      = "lo_commitdate" + 1;
DELETE FROM "part" WHERE "p_partkey" IS NULL OR "p_partkey" = 0;
DELETE FROM "part" WHERE "p_name" IS NULL OR "p_mfgr" IS NULL OR "p_category" IS NULL OR "p_brand1" IS NULL OR "p_color" IS NULL OR "p_type" IS NULL OR "p_size" IS NULL OR "p_container" IS NULL;
UPDATE "part" SET "p_size" = "p_size" + 1;
DELETE FROM "supplier" WHERE "s_suppkey" IS NULL OR "s_suppkey" = 0;
DELETE FROM "supplier" WHERE "s_name" IS NULL OR "s_address" IS NULL OR "s_city" IS NULL OR "s_nation" IS NULL OR "s_region" IS NULL OR "s_phone" IS NULL;

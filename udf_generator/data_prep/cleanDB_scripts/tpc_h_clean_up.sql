DELETE
FROM "customer"
WHERE "c_custkey" IS NULL
   OR "c_name" IS NULL
   OR "c_address" IS NULL
   OR "c_nationkey" IS NULL
   OR "c_phone" IS NULL
   OR "c_acctbal" IS NULL
   OR "c_mktsegment" IS NULL
   OR "c_comment" IS NULL;
UPDATE "customer"
SET "c_custkey"   = "c_custkey" + 1,
    "c_nationkey" = "c_nationkey" + 1,
    "c_acctbal"   = "c_acctbal" + 1;
DELETE
FROM "lineitem"
WHERE "l_orderkey" IS NULL
   OR "l_partkey" IS NULL
   OR "l_suppkey" IS NULL
   OR "l_linenumber" IS NULL
   OR "l_quantity" IS NULL
   OR "l_extendedprice" IS NULL
   OR "l_discount" IS NULL
   OR "l_tax" IS NULL
   OR "l_returnflag" IS NULL
   OR "l_linestatus" IS NULL
   OR "l_shipinstruct" IS NULL
   OR "l_shipmode" IS NULL
   OR "l_comment" IS NULL;
UPDATE "lineitem"
SET "l_orderkey"      = "l_orderkey" + 1,
    "l_partkey"       = "l_partkey" + 1,
    "l_suppkey"       = "l_suppkey" + 1,
    "l_linenumber"    = "l_linenumber" + 1,
    "l_quantity"      = "l_quantity" + 1,
    "l_extendedprice" = "l_extendedprice" + 1,
    "l_discount"      = "l_discount" + 1,
    "l_tax"           = "l_tax" + 1;
DELETE FROM "nation" WHERE "n_nationkey" IS NULL OR "n_name" IS NULL OR "n_regionkey" IS NULL OR "n_comment" IS NULL;
UPDATE "nation" SET "n_nationkey" = "n_nationkey" + 1, "n_regionkey" = "n_regionkey" + 1;
DELETE
FROM "orders"
WHERE "o_orderkey" IS NULL
   OR "o_custkey" IS NULL
   OR "o_orderstatus" IS NULL
   OR "o_totalprice" IS NULL
   OR "o_orderpriority" IS NULL
   OR "o_clerk" IS NULL
   OR "o_shippriority" IS NULL
   OR "o_comment" IS NULL;
UPDATE "orders"
SET "o_orderkey"     = "o_orderkey" + 1,
    "o_custkey"      = "o_custkey" + 1,
    "o_totalprice"   = "o_totalprice" + 1,
    "o_shippriority" = "o_shippriority" + 1;
DELETE
FROM "part"
WHERE "p_partkey" IS NULL
   OR "p_name" IS NULL
   OR "p_mfgr" IS NULL
   OR "p_brand" IS NULL
   OR "p_type" IS NULL
   OR "p_size" IS NULL
   OR "p_container" IS NULL
   OR "p_retailprice" IS NULL
   OR "p_comment" IS NULL;
UPDATE "part"
SET "p_partkey"     = "p_partkey" + 1,
    "p_size"        = "p_size" + 1,
    "p_retailprice" = "p_retailprice" + 1;
DELETE
FROM "partsupp"
WHERE "ps_partkey" IS NULL
   OR "ps_suppkey" IS NULL
   OR "ps_availqty" IS NULL
   OR "ps_supplycost" IS NULL
   OR "ps_comment" IS NULL;
UPDATE "partsupp"
SET "ps_partkey"    = "ps_partkey" + 1,
    "ps_suppkey"    = "ps_suppkey" + 1,
    "ps_availqty"   = "ps_availqty" + 1,
    "ps_supplycost" = "ps_supplycost" + 1;
DELETE FROM "region" WHERE "r_regionkey" IS NULL OR "r_name" IS NULL OR "r_comment" IS NULL;
UPDATE "region" SET "r_regionkey" = "r_regionkey" + 1;
DELETE
FROM "supplier"
WHERE "s_suppkey" IS NULL
   OR "s_name" IS NULL
   OR "s_address" IS NULL
   OR "s_nationkey" IS NULL
   OR "s_phone" IS NULL
   OR "s_acctbal" IS NULL
   OR "s_comment" IS NULL;
UPDATE "supplier"
SET "s_suppkey"   = "s_suppkey" + 1,
    "s_nationkey" = "s_nationkey" + 1,
    "s_acctbal"   = "s_acctbal" + 1;

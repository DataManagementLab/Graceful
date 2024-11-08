DELETE
FROM "aka_name"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "aka_name"
WHERE "person_id" IS NULL
   OR "name" IS NULL
   OR "name_pcode_cf" IS NULL
   OR "name_pcode_nf" IS NULL
   OR "md5sum" IS NULL;
UPDATE "aka_name"
SET "person_id" = "person_id" + 1;
DELETE
FROM "aka_title"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "aka_title"
WHERE "movie_id" IS NULL
   OR "title" IS NULL
   OR "kind_id" IS NULL
   OR "production_year" IS NULL
   OR "phonetic_code" IS NULL
   OR "note" IS NULL
   OR "md5sum" IS NULL;
UPDATE "aka_title"
SET "movie_id"        = "movie_id" + 1,
    "kind_id"         = "kind_id" + 1,
    "production_year" = "production_year" + 1;
DELETE
FROM "cast_info"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "cast_info"
WHERE "person_id" IS NULL
   OR "movie_id" IS NULL
   OR "role_id" IS NULL;
UPDATE "cast_info"
SET "person_id" = "person_id" + 1,
    "movie_id"  = "movie_id" + 1,
    "role_id"   = "role_id" + 1;
DELETE
FROM "char_name"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "char_name"
WHERE "name" IS NULL
   OR "name_pcode_nf" IS NULL
   OR "surname_pcode" IS NULL
   OR "md5sum" IS NULL;
DELETE
FROM "comp_cast_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "comp_cast_type"
WHERE "kind" IS NULL;
DELETE
FROM "company_name"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "company_name"
WHERE "name" IS NULL
   OR "country_code" IS NULL
   OR "name_pcode_nf" IS NULL
   OR "name_pcode_sf" IS NULL
   OR "md5sum" IS NULL;
DELETE
FROM "company_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "company_type"
WHERE "kind" IS NULL;
DELETE
FROM "complete_cast"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "complete_cast"
WHERE "movie_id" IS NULL
   OR "subject_id" IS NULL
   OR "status_id" IS NULL;
UPDATE "complete_cast"
SET "movie_id"   = "movie_id" + 1,
    "subject_id" = "subject_id" + 1,
    "status_id"  = "status_id" + 1;
DELETE
FROM "info_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "info_type"
WHERE "info" IS NULL;
DELETE
FROM "keyword"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "keyword"
WHERE "keyword" IS NULL
   OR "phonetic_code" IS NULL;
DELETE
FROM "kind_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "kind_type"
WHERE "kind" IS NULL;
DELETE
FROM "link_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "link_type"
WHERE "link" IS NULL;
DELETE
FROM "movie_companies"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "movie_companies"
WHERE "movie_id" IS NULL
   OR "company_id" IS NULL
   OR "company_type_id" IS NULL;
UPDATE "movie_companies"
SET "movie_id"        = "movie_id" + 1,
    "company_id"      = "company_id" + 1,
    "company_type_id" = "company_type_id" + 1;
DELETE
FROM "movie_info"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "movie_info"
WHERE "movie_id" IS NULL
   OR "info_type_id" IS NULL
   OR "info" IS NULL;
UPDATE "movie_info"
SET "movie_id"     = "movie_id" + 1,
    "info_type_id" = "info_type_id" + 1;
DELETE
FROM "movie_info_idx"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "movie_info_idx"
WHERE "movie_id" IS NULL
   OR "info_type_id" IS NULL
   OR "info" IS NULL;
UPDATE "movie_info_idx"
SET "movie_id"     = "movie_id" + 1,
    "info_type_id" = "info_type_id" + 1;
DELETE
FROM "movie_keyword"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "movie_keyword"
WHERE "movie_id" IS NULL
   OR "keyword_id" IS NULL;
UPDATE "movie_keyword"
SET "movie_id"   = "movie_id" + 1,
    "keyword_id" = "keyword_id" + 1;
DELETE
FROM "movie_link"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "movie_link"
WHERE "movie_id" IS NULL
   OR "linked_movie_id" IS NULL
   OR "link_type_id" IS NULL;
UPDATE "movie_link"
SET "movie_id"        = "movie_id" + 1,
    "linked_movie_id" = "linked_movie_id" + 1,
    "link_type_id"    = "link_type_id" + 1;
DELETE
FROM "name"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "name"
WHERE "name" IS NULL
   OR "name_pcode_cf" IS NULL
   OR "name_pcode_nf" IS NULL
   OR "md5sum" IS NULL;
DELETE
FROM "person_info"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "person_info"
WHERE "person_id" IS NULL
   OR "info_type_id" IS NULL
   OR "info" IS NULL;
UPDATE "person_info"
SET "person_id"    = "person_id" + 1,
    "info_type_id" = "info_type_id" + 1;
DELETE
FROM "role_type"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "role_type"
WHERE "role" IS NULL;
DELETE
FROM "title"
WHERE "id" IS NULL
   OR "id" = 0;
DELETE
FROM "title"
WHERE "title" IS NULL
   OR "kind_id" IS NULL
   OR "production_year" IS NULL
   OR "md5sum" IS NULL;
UPDATE "title"
SET "kind_id"         = "kind_id" + 1,
    "production_year" = "production_year" + 1;

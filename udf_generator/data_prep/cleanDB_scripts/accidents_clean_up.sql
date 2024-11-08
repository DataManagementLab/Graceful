DELETE FROM "nesreca" WHERE "id_nesreca" IS NULL OR "klas_nesreca" IS NULL OR "upravna_enota" IS NULL OR "cas_nesreca" IS NULL OR "naselje_ali_izven" IS NULL OR "kategorija_cesta" IS NULL OR "oznaka_cesta_ali_naselje" IS NULL OR "tekst_cesta_ali_naselje" IS NULL OR "oznaka_odsek_ali_ulica" IS NULL OR "tekst_odsek_ali_ulica" IS NULL OR "opis_prizorisce" IS NULL OR "vzrok_nesreca" IS NULL OR "tip_nesreca" IS NULL OR "vreme_nesreca" IS NULL OR "stanje_promet" IS NULL OR "stanje_vozisce" IS NULL OR "stanje_povrsina_vozisce" IS NULL OR "x" IS NULL OR "y" IS NULL OR "x_wgs84" IS NULL OR "y_wgs84" IS NULL;
UPDATE "nesreca" SET "x" = "x" + 1, "y" = "y" + 1, "x_wgs84" = "x_wgs84" + 1, "y_wgs84" = "y_wgs84" + 1;
DELETE
FROM "oseba"
WHERE "id_nesreca" IS NULL
   OR "povzrocitelj_ali_udelezenec" IS NULL
   OR "starost" IS NULL
   OR "spol" IS NULL
   OR "upravna_enota" IS NULL
   OR "drzavljanstvo" IS NULL
   OR "poskodba" IS NULL
   OR "vrsta_udelezenca" IS NULL
   OR "varnostni_pas_ali_celada" IS NULL
   OR "vozniski_staz_LL" IS NULL
   OR "vozniski_staz_MM" IS NULL
   OR "starost_d" IS NULL
   OR "vozniski_staz_d" IS NULL
   OR "alkotest_d" IS NULL
   OR "strokovni_pregled_d" IS NULL;
UPDATE "oseba"
SET "starost"          = "starost" + 1,
    "vozniski_staz_LL" = "vozniski_staz_LL" + 1,
    "vozniski_staz_MM" = "vozniski_staz_MM" + 1;
DELETE FROM "upravna_enota" WHERE "id_upravna_enota" IS NULL OR "ime_upravna_enota" IS NULL OR "st_prebivalcev" IS NULL OR "povrsina" IS NULL;
UPDATE "upravna_enota" SET "st_prebivalcev" = "st_prebivalcev" + 1, "povrsina" = "povrsina" + 1;

DELETE
FROM "bod_schuze"
WHERE "id_bod" IS NULL
   OR "id_schuze" IS NULL
   OR "id_tisk" IS NULL
   OR "id_typ" IS NULL
   OR "bod" IS NULL
   OR "uplny_naz" IS NULL
   OR "uplny_kon" IS NULL
   OR "id_bod_stav" IS NULL
   OR "rj" IS NULL
   OR "zkratka" IS NULL;
UPDATE "bod_schuze" SET "id_bod" = "id_bod" + 1, "id_schuze" = "id_schuze" + 1, "bod" = "bod" + 1, "id_bod_stav" = "id_bod_stav" + 1;
DELETE FROM "bod_stav" WHERE "id_bod_stav" IS NULL OR "id_bod_stav" = 0;
DELETE FROM "bod_stav" WHERE "popis" IS NULL;
DELETE FROM "funkce" WHERE "id_funkce" IS NULL OR "id_funkce" = 0;
DELETE
FROM "funkce"
WHERE "id_organ" IS NULL
   OR "id_typ_funkce" IS NULL
   OR "nazev_funkce_cz" IS NULL
   OR "priorita" IS NULL;
UPDATE "funkce" SET "id_organ" = "id_organ" + 1, "id_typ_funkce" = "id_typ_funkce" + 1, "priorita" = "priorita" + 1;
DELETE FROM "hl_check" WHERE "id_hlasovani" IS NULL OR "turn" IS NULL OR "mode" IS NULL;
UPDATE "hl_check" SET "id_hlasovani" = "id_hlasovani" + 1, "turn" = "turn" + 1, "mode" = "mode" + 1;
DELETE FROM "hl_hlasovani" WHERE "id_hlasovani" IS NULL OR "id_hlasovani" = 0;
DELETE
FROM "hl_hlasovani"
WHERE "id_organ" IS NULL
   OR "schuze" IS NULL
   OR "cislo" IS NULL
   OR "bod" IS NULL
   OR "datum" IS NULL
   OR "pro" IS NULL
   OR "proti" IS NULL
   OR "zdrzel" IS NULL
   OR "nehlasoval" IS NULL
   OR "prihlaseno" IS NULL
   OR "kvorum" IS NULL
   OR "druh_hlasovani" IS NULL
   OR "vysledek" IS NULL
   OR "nazev_dlouhy" IS NULL;
UPDATE "hl_hlasovani" SET "id_organ" = "id_organ" + 1, "schuze" = "schuze" + 1, "cislo" = "cislo" + 1, "bod" = "bod" + 1, "pro" = "pro" + 1, "proti" = "proti" + 1, "zdrzel" = "zdrzel" + 1, "nehlasoval" = "nehlasoval" + 1, "prihlaseno" = "prihlaseno" + 1, "kvorum" = "kvorum" + 1;
DELETE FROM "hl_poslanec" WHERE "id_poslanec" IS NULL OR "id_hlasovani" IS NULL OR "vysledek" IS NULL;
UPDATE "hl_poslanec" SET "id_poslanec" = "id_poslanec" + 1, "id_hlasovani" = "id_hlasovani" + 1;
DELETE FROM "hl_vazby" WHERE "id_hlasovani" IS NULL OR "turn" IS NULL OR "typ" IS NULL;
UPDATE "hl_vazby" SET "id_hlasovani" = "id_hlasovani" + 1, "turn" = "turn" + 1, "typ" = "typ" + 1;
DELETE FROM "hl_zposlanec" WHERE "id_hlasovani" IS NULL OR "id_osoba" IS NULL OR "mode" IS NULL;
UPDATE "hl_zposlanec" SET "id_hlasovani" = "id_hlasovani" + 1, "id_osoba" = "id_osoba" + 1, "mode" = "mode" + 1;
DELETE FROM "omluvy" WHERE "id_organ" IS NULL OR "id_poslanec" IS NULL OR "den" IS NULL;
UPDATE "omluvy" SET "id_organ" = "id_organ" + 1, "id_poslanec" = "id_poslanec" + 1;
DELETE FROM "organy" WHERE "id_organ" IS NULL OR "id_organ" = 0;
DELETE
FROM "organy"
WHERE "organ_id_organ" IS NULL
   OR "id_typ_organu" IS NULL
   OR "zkratka" IS NULL
   OR "nazev_organu_cz" IS NULL
   OR "nazev_organu_en" IS NULL
   OR "od_organ" IS NULL
   OR "cl_organ_base" IS NULL;
UPDATE "organy" SET "organ_id_organ" = "organ_id_organ" + 1, "id_typ_organu" = "id_typ_organu" + 1, "cl_organ_base" = "cl_organ_base" + 1;
DELETE FROM "osoby" WHERE "id_osoba" IS NULL OR "id_osoba" = 0;
DELETE FROM "osoby" WHERE "jmeno" IS NULL OR "prijmeni" IS NULL OR "narozeni" IS NULL OR "pohlavi" IS NULL;
DELETE FROM "pkgps" WHERE "id_poslanec" IS NULL OR "adresa" IS NULL OR "sirka" IS NULL OR "delka" IS NULL;
UPDATE "pkgps" SET "id_poslanec" = "id_poslanec" + 1, "sirka" = "sirka" + 1, "delka" = "delka" + 1;
DELETE FROM "poslanec" WHERE "id_poslanec" IS NULL OR "id_poslanec" = 0;
DELETE FROM "poslanec" WHERE "id_osoba" IS NULL OR "id_kraj" IS NULL OR "id_kandidatka" IS NULL OR "id_obdobi" IS NULL OR "web" IS NULL OR "ulice" IS NULL OR "obec" IS NULL OR "psc" IS NULL OR "email" IS NULL OR "telefon" IS NULL OR "fax" IS NULL OR "psp_telefon" IS NULL OR "facebook" IS NULL OR "foto" IS NULL;
UPDATE "poslanec" SET "id_osoba" = "id_osoba" + 1, "id_kraj" = "id_kraj" + 1, "id_kandidatka" = "id_kandidatka" + 1, "id_obdobi" = "id_obdobi" + 1, "foto" = "foto" + 1;
DELETE FROM "schuze" WHERE "id_schuze" IS NULL OR "id_schuze" = 0;
DELETE FROM "schuze" WHERE "id_organ" IS NULL OR "schuze" IS NULL OR "od_schuze" IS NULL OR "do_schuze" IS NULL OR "aktualizace" IS NULL;
UPDATE "schuze" SET "id_organ" = "id_organ" + 1, "schuze" = "schuze" + 1;
DELETE FROM "schuze_stav" WHERE "id_schuze" IS NULL OR "stav" IS NULL OR "typ" IS NULL;
UPDATE "schuze_stav" SET "id_schuze" = "id_schuze" + 1, "stav" = "stav" + 1;
DELETE FROM "typ_funkce" WHERE "id_typ_funkce" IS NULL OR "id_typ_funkce" = 0;
DELETE FROM "typ_funkce" WHERE "id_typ_org" IS NULL OR "typ_funkce_cz" IS NULL OR "priorita" IS NULL;
UPDATE "typ_funkce" SET "id_typ_org" = "id_typ_org" + 1, "priorita" = "priorita" + 1;
DELETE FROM "typ_organu" WHERE "id_typ_org" IS NULL OR "id_typ_org" = 0;
DELETE FROM "typ_organu" WHERE "typ_id_typ_org" IS NULL OR "nazev_typ_org_cz" IS NULL OR "priorita" IS NULL;
UPDATE "typ_organu" SET "priorita" = "priorita" + 1;
DELETE FROM "zarazeni" WHERE "id_osoba" IS NULL OR "id_of" IS NULL OR "cl_funkce" IS NULL OR "od_o" IS NULL OR "do_o" IS NULL;
UPDATE "zarazeni" SET "id_osoba" = "id_osoba" + 1, "id_of" = "id_of" + 1, "cl_funkce" = "cl_funkce" + 1;
DELETE FROM "zmatecne" WHERE "id_hlasovani" IS NULL;
UPDATE "zmatecne" SET "id_hlasovani" = "id_hlasovani" + 1;

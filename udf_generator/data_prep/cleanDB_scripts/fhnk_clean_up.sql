DELETE FROM "pripady" WHERE "Identifikace_pripadu" IS NULL OR "Identifikace_pripadu" = 0;
DELETE FROM "pripady" WHERE "Identifikator_pacienta" IS NULL OR "Kod_zdravotni_pojistovny" IS NULL OR "Datum_prijeti" IS NULL OR "Datum_propusteni" IS NULL OR "Delka_hospitalizace" IS NULL OR "Vekovy_Interval_Pacienta" IS NULL OR "Pohlavi_pacienta" IS NULL OR "Zakladni_diagnoza" IS NULL OR "Seznam_vedlejsich_diagnoz" IS NULL OR "DRG_skupina" IS NULL OR "PSC" IS NULL;
UPDATE "pripady" SET "Identifikator_pacienta" = "Identifikator_pacienta" + 1, "Kod_zdravotni_pojistovny" = "Kod_zdravotni_pojistovny" + 1, "Delka_hospitalizace" = "Delka_hospitalizace" + 1, "DRG_skupina" = "DRG_skupina" + 1;
DELETE
FROM "vykony"
WHERE "Identifikace_pripadu" IS NULL
   OR "Identifikace_pripadu" = 0
   OR "Kod_polozky" IS NULL
   OR "Kod_polozky" = 0;
DELETE
FROM "vykony"
WHERE "Datum_provedeni_vykonu" IS NULL
   OR "Typ_polozky" IS NULL
   OR "Pocet" IS NULL
   OR "Body" IS NULL;
UPDATE "vykony"
SET "Typ_polozky" = "Typ_polozky" + 1,
    "Pocet"       = "Pocet" + 1,
    "Body"        = "Body" + 1;
DELETE FROM "zup" WHERE "Identifikace_pripadu" IS NULL OR "Identifikace_pripadu" = 0 OR "Kod_polozky" IS NULL OR "Kod_polozky" = 0;
DELETE
FROM "zup"
WHERE "Datum_provedeni_vykonu" IS NULL
   OR "Typ_polozky" IS NULL
   OR "Pocet" IS NULL
   OR "Cena" IS NULL;
UPDATE "zup"
SET "Typ_polozky" = "Typ_polozky" + 1,
    "Pocet"       = "Pocet" + 1,
    "Cena"        = "Cena" + 1;

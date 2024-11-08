DELETE FROM "regular_season_compact_results" WHERE "season" IS NULL OR "season" = 0 OR "daynum" IS NULL OR "daynum" = 0 OR "wteam" IS NULL OR "wteam" = 0 OR "lteam" IS NULL OR "lteam" = 0;
DELETE FROM "regular_season_compact_results" WHERE "wscore" IS NULL OR "lscore" IS NULL OR "wloc" IS NULL OR "numot" IS NULL;
UPDATE "regular_season_compact_results" SET "wscore" = "wscore" + 1, "lscore" = "lscore" + 1, "numot" = "numot" + 1;
DELETE FROM "regular_season_detailed_results" WHERE "season" IS NULL OR "season" = 0 OR "daynum" IS NULL OR "daynum" = 0 OR "wteam" IS NULL OR "wteam" = 0 OR "lteam" IS NULL OR "lteam" = 0;
DELETE FROM "regular_season_detailed_results" WHERE "wscore" IS NULL OR "lscore" IS NULL OR "wloc" IS NULL OR "numot" IS NULL OR "wfgm" IS NULL OR "wfga" IS NULL OR "wfgm3" IS NULL OR "wfga3" IS NULL OR "wftm" IS NULL OR "wfta" IS NULL OR "wor" IS NULL OR "wdr" IS NULL OR "wast" IS NULL OR "wto" IS NULL OR "wstl" IS NULL OR "wblk" IS NULL OR "wpf" IS NULL OR "lfgm" IS NULL OR "lfga" IS NULL OR "lfgm3" IS NULL OR "lfga3" IS NULL OR "lftm" IS NULL OR "lfta" IS NULL OR "lor" IS NULL OR "ldr" IS NULL OR "last" IS NULL OR "lto" IS NULL OR "lstl" IS NULL OR "lblk" IS NULL OR "lpf" IS NULL;
UPDATE "regular_season_detailed_results" SET "wscore" = "wscore" + 1, "lscore" = "lscore" + 1, "numot" = "numot" + 1, "wfgm" = "wfgm" + 1, "wfga" = "wfga" + 1, "wfgm3" = "wfgm3" + 1, "wfga3" = "wfga3" + 1, "wftm" = "wftm" + 1, "wfta" = "wfta" + 1, "wor" = "wor" + 1, "wdr" = "wdr" + 1, "wast" = "wast" + 1, "wto" = "wto" + 1, "wstl" = "wstl" + 1, "wblk" = "wblk" + 1, "wpf" = "wpf" + 1, "lfgm" = "lfgm" + 1, "lfga" = "lfga" + 1, "lfgm3" = "lfgm3" + 1, "lfga3" = "lfga3" + 1, "lftm" = "lftm" + 1, "lfta" = "lfta" + 1, "lor" = "lor" + 1, "ldr" = "ldr" + 1, "last" = "last" + 1, "lto" = "lto" + 1, "lstl" = "lstl" + 1, "lblk" = "lblk" + 1, "lpf" = "lpf" + 1;
DELETE FROM "seasons" WHERE "season" IS NULL OR "season" = 0;
DELETE FROM "seasons" WHERE "dayzero" IS NULL OR "regionW" IS NULL OR "regionX" IS NULL OR "regionY" IS NULL OR "regionZ" IS NULL;
DELETE
FROM "target"
WHERE "id" IS NULL
   OR "season" IS NULL
   OR "team_id1" IS NULL
   OR "team_id2" IS NULL
   OR "pred" IS NULL
   OR "team_id1_wins" IS NULL
   OR "team_id2_wins" IS NULL;
UPDATE "target"
SET "season"        = "season" + 1,
    "team_id1"      = "team_id1" + 1,
    "team_id2"      = "team_id2" + 1,
    "pred"          = "pred" + 1,
    "team_id1_wins" = "team_id1_wins" + 1,
    "team_id2_wins" = "team_id2_wins" + 1;
DELETE FROM "teams" WHERE "team_id" IS NULL OR "team_id" = 0;
DELETE FROM "teams" WHERE "team_name" IS NULL;
DELETE FROM "tourney_compact_results" WHERE "season" IS NULL OR "season" = 0 OR "daynum" IS NULL OR "daynum" = 0 OR "wteam" IS NULL OR "wteam" = 0 OR "lteam" IS NULL OR "lteam" = 0;
DELETE FROM "tourney_compact_results" WHERE "wscore" IS NULL OR "lscore" IS NULL OR "wloc" IS NULL OR "numot" IS NULL;
UPDATE "tourney_compact_results" SET "wscore" = "wscore" + 1, "lscore" = "lscore" + 1, "numot" = "numot" + 1;
DELETE FROM "tourney_detailed_results" WHERE "season" IS NULL OR "season" = 0 OR "daynum" IS NULL OR "daynum" = 0 OR "wteam" IS NULL OR "wteam" = 0 OR "lteam" IS NULL OR "lteam" = 0;
DELETE FROM "tourney_detailed_results" WHERE "wscore" IS NULL OR "lscore" IS NULL OR "wloc" IS NULL OR "numot" IS NULL OR "wfgm" IS NULL OR "wfga" IS NULL OR "wfgm3" IS NULL OR "wfga3" IS NULL OR "wftm" IS NULL OR "wfta" IS NULL OR "wor" IS NULL OR "wdr" IS NULL OR "wast" IS NULL OR "wto" IS NULL OR "wstl" IS NULL OR "wblk" IS NULL OR "wpf" IS NULL OR "lfgm" IS NULL OR "lfga" IS NULL OR "lfgm3" IS NULL OR "lfga3" IS NULL OR "lftm" IS NULL OR "lfta" IS NULL OR "lor" IS NULL OR "ldr" IS NULL OR "last" IS NULL OR "lto" IS NULL OR "lstl" IS NULL OR "lblk" IS NULL OR "lpf" IS NULL;
UPDATE "tourney_detailed_results" SET "wscore" = "wscore" + 1, "lscore" = "lscore" + 1, "numot" = "numot" + 1, "wfgm" = "wfgm" + 1, "wfga" = "wfga" + 1, "wfgm3" = "wfgm3" + 1, "wfga3" = "wfga3" + 1, "wftm" = "wftm" + 1, "wfta" = "wfta" + 1, "wor" = "wor" + 1, "wdr" = "wdr" + 1, "wast" = "wast" + 1, "wto" = "wto" + 1, "wstl" = "wstl" + 1, "wblk" = "wblk" + 1, "wpf" = "wpf" + 1, "lfgm" = "lfgm" + 1, "lfga" = "lfga" + 1, "lfgm3" = "lfgm3" + 1, "lfga3" = "lfga3" + 1, "lftm" = "lftm" + 1, "lfta" = "lfta" + 1, "lor" = "lor" + 1, "ldr" = "ldr" + 1, "last" = "last" + 1, "lto" = "lto" + 1, "lstl" = "lstl" + 1, "lblk" = "lblk" + 1, "lpf" = "lpf" + 1;
DELETE FROM "tourney_seeds" WHERE "season" IS NULL OR "season" = 0;
DELETE FROM "tourney_seeds" WHERE "seed" IS NULL OR "team" IS NULL;
UPDATE "tourney_seeds" SET "team" = "team" + 1;
DELETE FROM "tourney_slots" WHERE "season" IS NULL OR "season" = 0;
DELETE FROM "tourney_slots" WHERE "slot" IS NULL OR "strongseed" IS NULL OR "weakseed" IS NULL;

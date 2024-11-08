DELETE FROM "awards_coaches" WHERE "id" IS NULL OR "id" = 0;
DELETE FROM "awards_coaches" WHERE "year" IS NULL OR "coachID" IS NULL OR "award" IS NULL OR "lgID" IS NULL;
UPDATE "awards_coaches" SET "year" = "year" + 1;
DELETE FROM "awards_players" WHERE "year" IS NULL OR "year" = 0;
DELETE FROM "awards_players" WHERE "playerID" IS NULL OR "award" IS NULL OR "lgID" IS NULL;
DELETE FROM "coaches" WHERE "year" IS NULL OR "year" = 0 OR "stint" IS NULL OR "stint" = 0;
DELETE FROM "coaches" WHERE "coachID" IS NULL OR "tmID" IS NULL OR "lgID" IS NULL OR "won" IS NULL OR "lost" IS NULL OR "post_wins" IS NULL OR "post_losses" IS NULL;
UPDATE "coaches" SET "won" = "won" + 1, "lost" = "lost" + 1, "post_wins" = "post_wins" + 1, "post_losses" = "post_losses" + 1;
DELETE FROM "draft" WHERE "id" IS NULL OR "id" = 0;
DELETE FROM "draft" WHERE "draftYear" IS NULL OR "draftRound" IS NULL OR "draftSelection" IS NULL OR "draftOverall" IS NULL OR "tmID" IS NULL OR "firstName" IS NULL OR "lastName" IS NULL OR "draftFrom" IS NULL OR "lgID" IS NULL;
UPDATE "draft" SET "draftYear" = "draftYear" + 1, "draftRound" = "draftRound" + 1, "draftSelection" = "draftSelection" + 1, "draftOverall" = "draftOverall" + 1;
DELETE FROM "player_allstar" WHERE "season_id" IS NULL OR "season_id" = 0;
DELETE FROM "player_allstar" WHERE "playerID" IS NULL OR "last_name" IS NULL OR "first_name" IS NULL OR "conference" IS NULL OR "league_id" IS NULL OR "games_played" IS NULL OR "minutes" IS NULL OR "points" IS NULL OR "rebounds" IS NULL OR "assists" IS NULL OR "fg_attempted" IS NULL OR "fg_made" IS NULL OR "ft_attempted" IS NULL OR "ft_made" IS NULL;
UPDATE "player_allstar" SET "games_played" = "games_played" + 1, "minutes" = "minutes" + 1, "points" = "points" + 1, "rebounds" = "rebounds" + 1, "assists" = "assists" + 1, "fg_attempted" = "fg_attempted" + 1, "fg_made" = "fg_made" + 1, "ft_attempted" = "ft_attempted" + 1, "ft_made" = "ft_made" + 1;
DELETE
FROM "players"
WHERE "playerID" IS NULL
   OR "useFirst" IS NULL
   OR "firstName" IS NULL
   OR "lastName" IS NULL
   OR "pos" IS NULL
   OR "firstseason" IS NULL
   OR "lastseason" IS NULL
   OR "height" IS NULL
   OR "weight" IS NULL
   OR "college" IS NULL
   OR "birthDate" IS NULL
   OR "birthCity" IS NULL
   OR "birthState" IS NULL
   OR "birthCountry" IS NULL
   OR "highSchool" IS NULL
   OR "hsCity" IS NULL
   OR "hsState" IS NULL
   OR "hsCountry" IS NULL
   OR "deathDate" IS NULL
   OR "race" IS NULL;
UPDATE "players"
SET "firstseason" = "firstseason" + 1,
    "lastseason"  = "lastseason" + 1,
    "height"      = "height" + 1,
    "weight"      = "weight" + 1;
DELETE FROM "players_teams" WHERE "id" IS NULL OR "id" = 0;
DELETE FROM "players_teams" WHERE "playerID" IS NULL OR "year" IS NULL OR "stint" IS NULL OR "tmID" IS NULL OR "lgID" IS NULL OR "GP" IS NULL OR "GS" IS NULL OR "minutes" IS NULL OR "points" IS NULL OR "oRebounds" IS NULL OR "dRebounds" IS NULL OR "rebounds" IS NULL OR "assists" IS NULL OR "steals" IS NULL OR "blocks" IS NULL OR "turnovers" IS NULL OR "PF" IS NULL OR "fgAttempted" IS NULL OR "fgMade" IS NULL OR "ftAttempted" IS NULL OR "ftMade" IS NULL OR "threeAttempted" IS NULL OR "threeMade" IS NULL OR "PostGP" IS NULL OR "PostGS" IS NULL OR "PostMinutes" IS NULL OR "PostPoints" IS NULL OR "PostoRebounds" IS NULL OR "PostdRebounds" IS NULL OR "PostRebounds" IS NULL OR "PostAssists" IS NULL OR "PostSteals" IS NULL OR "PostBlocks" IS NULL OR "PostTurnovers" IS NULL OR "PostPF" IS NULL OR "PostfgAttempted" IS NULL OR "PostfgMade" IS NULL OR "PostftAttempted" IS NULL OR "PostftMade" IS NULL OR "PostthreeAttempted" IS NULL OR "PostthreeMade" IS NULL;
UPDATE "players_teams" SET "year" = "year" + 1, "stint" = "stint" + 1, "GP" = "GP" + 1, "GS" = "GS" + 1, "minutes" = "minutes" + 1, "points" = "points" + 1, "oRebounds" = "oRebounds" + 1, "dRebounds" = "dRebounds" + 1, "rebounds" = "rebounds" + 1, "assists" = "assists" + 1, "steals" = "steals" + 1, "blocks" = "blocks" + 1, "turnovers" = "turnovers" + 1, "PF" = "PF" + 1, "fgAttempted" = "fgAttempted" + 1, "fgMade" = "fgMade" + 1, "ftAttempted" = "ftAttempted" + 1, "ftMade" = "ftMade" + 1, "threeAttempted" = "threeAttempted" + 1, "threeMade" = "threeMade" + 1, "PostGP" = "PostGP" + 1, "PostGS" = "PostGS" + 1, "PostMinutes" = "PostMinutes" + 1, "PostPoints" = "PostPoints" + 1, "PostoRebounds" = "PostoRebounds" + 1, "PostdRebounds" = "PostdRebounds" + 1, "PostRebounds" = "PostRebounds" + 1, "PostAssists" = "PostAssists" + 1, "PostSteals" = "PostSteals" + 1, "PostBlocks" = "PostBlocks" + 1, "PostTurnovers" = "PostTurnovers" + 1, "PostPF" = "PostPF" + 1, "PostfgAttempted" = "PostfgAttempted" + 1, "PostfgMade" = "PostfgMade" + 1, "PostftAttempted" = "PostftAttempted" + 1, "PostftMade" = "PostftMade" + 1, "PostthreeAttempted" = "PostthreeAttempted" + 1, "PostthreeMade" = "PostthreeMade" + 1;
DELETE FROM "series_post" WHERE "id" IS NULL OR "id" = 0;
DELETE FROM "series_post" WHERE "year" IS NULL OR "round" IS NULL OR "series" IS NULL OR "tmIDWinner" IS NULL OR "lgIDWinner" IS NULL OR "tmIDLoser" IS NULL OR "lgIDLoser" IS NULL OR "W" IS NULL OR "L" IS NULL;
UPDATE "series_post" SET "year" = "year" + 1, "W" = "W" + 1, "L" = "L" + 1;
DELETE
FROM "teams"
WHERE "year" IS NULL
   OR "year" = 0;
DELETE
FROM "teams"
WHERE "lgID" IS NULL
   OR "tmID" IS NULL
   OR "franchID" IS NULL
   OR "divID" IS NULL
   OR "rank" IS NULL
   OR "confRank" IS NULL
   OR "name" IS NULL
   OR "o_fgm" IS NULL
   OR "o_fga" IS NULL
   OR "o_ftm" IS NULL
   OR "o_fta" IS NULL
   OR "o_3pm" IS NULL
   OR "o_3pa" IS NULL
   OR "o_oreb" IS NULL
   OR "o_dreb" IS NULL
   OR "o_reb" IS NULL
   OR "o_asts" IS NULL
   OR "o_pf" IS NULL
   OR "o_stl" IS NULL
   OR "o_to" IS NULL
   OR "o_blk" IS NULL
   OR "o_pts" IS NULL
   OR "d_fgm" IS NULL
   OR "d_fga" IS NULL
   OR "d_ftm" IS NULL
   OR "d_fta" IS NULL
   OR "d_3pm" IS NULL
   OR "d_3pa" IS NULL
   OR "d_oreb" IS NULL
   OR "d_dreb" IS NULL
   OR "d_reb" IS NULL
   OR "d_asts" IS NULL
   OR "d_pf" IS NULL
   OR "d_stl" IS NULL
   OR "d_to" IS NULL
   OR "d_blk" IS NULL
   OR "d_pts" IS NULL
   OR "o_tmRebound" IS NULL
   OR "d_tmRebound" IS NULL
   OR "homeWon" IS NULL
   OR "homeLost" IS NULL
   OR "awayWon" IS NULL
   OR "awayLost" IS NULL
   OR "neutWon" IS NULL
   OR "neutLoss" IS NULL
   OR "confWon" IS NULL
   OR "confLoss" IS NULL
   OR "divWon" IS NULL
   OR "divLoss" IS NULL
   OR "pace" IS NULL
   OR "won" IS NULL
   OR "lost" IS NULL
   OR "games" IS NULL
   OR "min" IS NULL
   OR "arena" IS NULL
   OR "attendance" IS NULL
   OR "bbtmID" IS NULL;
UPDATE "teams"
SET "rank"        = "rank" + 1,
    "confRank"    = "confRank" + 1,
    "o_fgm"       = "o_fgm" + 1,
    "o_fga"       = "o_fga" + 1,
    "o_ftm"       = "o_ftm" + 1,
    "o_fta"       = "o_fta" + 1,
    "o_3pm"       = "o_3pm" + 1,
    "o_3pa"       = "o_3pa" + 1,
    "o_oreb"      = "o_oreb" + 1,
    "o_dreb"      = "o_dreb" + 1,
    "o_reb"       = "o_reb" + 1,
    "o_asts"      = "o_asts" + 1,
    "o_pf"        = "o_pf" + 1,
    "o_stl"       = "o_stl" + 1,
    "o_to"        = "o_to" + 1,
    "o_blk"       = "o_blk" + 1,
    "o_pts"       = "o_pts" + 1,
    "d_fgm"       = "d_fgm" + 1,
    "d_fga"       = "d_fga" + 1,
    "d_ftm"       = "d_ftm" + 1,
    "d_fta"       = "d_fta" + 1,
    "d_3pm"       = "d_3pm" + 1,
    "d_3pa"       = "d_3pa" + 1,
    "d_oreb"      = "d_oreb" + 1,
    "d_dreb"      = "d_dreb" + 1,
    "d_reb"       = "d_reb" + 1,
    "d_asts"      = "d_asts" + 1,
    "d_pf"        = "d_pf" + 1,
    "d_stl"       = "d_stl" + 1,
    "d_to"        = "d_to" + 1,
    "d_blk"       = "d_blk" + 1,
    "d_pts"       = "d_pts" + 1,
    "o_tmRebound" = "o_tmRebound" + 1,
    "d_tmRebound" = "d_tmRebound" + 1,
    "homeWon"     = "homeWon" + 1,
    "homeLost"    = "homeLost" + 1,
    "awayWon"     = "awayWon" + 1,
    "awayLost"    = "awayLost" + 1,
    "neutWon"     = "neutWon" + 1,
    "neutLoss"    = "neutLoss" + 1,
    "confWon"     = "confWon" + 1,
    "confLoss"    = "confLoss" + 1,
    "divWon"      = "divWon" + 1,
    "divLoss"     = "divLoss" + 1,
    "pace"        = "pace" + 1,
    "won"         = "won" + 1,
    "lost"        = "lost" + 1,
    "games"       = "games" + 1,
    "min"         = "min" + 1,
    "attendance"  = "attendance" + 1;

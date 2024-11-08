DELETE FROM "allstarfull" WHERE "yearID" IS NULL OR "yearID" = 0 OR "gameNum" IS NULL OR "gameNum" = 0;
DELETE FROM "allstarfull" WHERE "playerID" IS NULL OR "gameID" IS NULL OR "teamID" IS NULL OR "lgID" IS NULL OR "GP" IS NULL;
UPDATE "allstarfull" SET "GP" = "GP" + 1;
DELETE FROM "appearances" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "appearances" WHERE "teamID" IS NULL OR "lgID" IS NULL OR "playerID" IS NULL OR "G_all" IS NULL OR "G_defense" IS NULL OR "G_p" IS NULL OR "G_c" IS NULL OR "G_1b" IS NULL OR "G_2b" IS NULL OR "G_3b" IS NULL OR "G_ss" IS NULL OR "G_lf" IS NULL OR "G_cf" IS NULL OR "G_rf" IS NULL OR "G_of" IS NULL OR "G_dh" IS NULL;
UPDATE "appearances" SET "G_all" = "G_all" + 1, "G_defense" = "G_defense" + 1, "G_p" = "G_p" + 1, "G_c" = "G_c" + 1, "G_1b" = "G_1b" + 1, "G_2b" = "G_2b" + 1, "G_3b" = "G_3b" + 1, "G_ss" = "G_ss" + 1, "G_lf" = "G_lf" + 1, "G_cf" = "G_cf" + 1, "G_rf" = "G_rf" + 1, "G_of" = "G_of" + 1, "G_dh" = "G_dh" + 1;
DELETE FROM "awardsmanagers" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "awardsmanagers" WHERE "managerID" IS NULL OR "awardID" IS NULL OR "lgID" IS NULL;
DELETE FROM "awardsplayers" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "awardsplayers" WHERE "playerID" IS NULL OR "awardID" IS NULL OR "lgID" IS NULL;
DELETE FROM "awardssharemanagers" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "awardssharemanagers" WHERE "awardID" IS NULL OR "lgID" IS NULL OR "managerID" IS NULL OR "pointsWon" IS NULL OR "pointsMax" IS NULL OR "votesFirst" IS NULL;
UPDATE "awardssharemanagers" SET "pointsWon" = "pointsWon" + 1, "pointsMax" = "pointsMax" + 1, "votesFirst" = "votesFirst" + 1;
DELETE FROM "awardsshareplayers" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "awardsshareplayers" WHERE "awardID" IS NULL OR "lgID" IS NULL OR "playerID" IS NULL OR "pointsWon" IS NULL OR "pointsMax" IS NULL OR "votesFirst" IS NULL;
UPDATE "awardsshareplayers" SET "pointsWon" = "pointsWon" + 1, "pointsMax" = "pointsMax" + 1, "votesFirst" = "votesFirst" + 1;
DELETE FROM "batting" WHERE "yearID" IS NULL OR "yearID" = 0 OR "stint" IS NULL OR "stint" = 0;
DELETE
FROM "batting"
WHERE "playerID" IS NULL
   OR "teamID" IS NULL
   OR "lgID" IS NULL
   OR "G" IS NULL
   OR "G_batting" IS NULL
   OR "AB" IS NULL
   OR "R" IS NULL
   OR "H" IS NULL
   OR "SecondB" IS NULL
   OR "ThirdB" IS NULL
   OR "HR" IS NULL
   OR "RBI" IS NULL
   OR "SB" IS NULL
   OR "CS" IS NULL
   OR "BB" IS NULL
   OR "SO" IS NULL
   OR "HBP" IS NULL
   OR "SH" IS NULL
   OR "GIDP" IS NULL
   OR "G_old" IS NULL;
UPDATE "batting"
SET "G"         = "G" + 1,
    "G_batting" = "G_batting" + 1,
    "AB"        = "AB" + 1,
    "R"         = "R" + 1,
    "H"         = "H" + 1,
    "SecondB"   = "SecondB" + 1,
    "ThirdB"    = "ThirdB" + 1,
    "HR"        = "HR" + 1,
    "RBI"       = "RBI" + 1,
    "SB"        = "SB" + 1,
    "CS"        = "CS" + 1,
    "BB"        = "BB" + 1,
    "SO"        = "SO" + 1,
    "HBP"       = "HBP" + 1,
    "SH"        = "SH" + 1,
    "GIDP"      = "GIDP" + 1,
    "G_old"     = "G_old" + 1;
DELETE
FROM "battingpost"
WHERE "yearID" IS NULL
   OR "yearID" = 0;
DELETE
FROM "battingpost"
WHERE "round" IS NULL
   OR "playerID" IS NULL
   OR "teamID" IS NULL
   OR "lgID" IS NULL
   OR "G" IS NULL
   OR "AB" IS NULL
   OR "R" IS NULL
   OR "H" IS NULL
   OR "SecondB" IS NULL
   OR "ThirdB" IS NULL
   OR "HR" IS NULL
   OR "RBI" IS NULL
   OR "SB" IS NULL
   OR "CS" IS NULL
   OR "BB" IS NULL
   OR "SO" IS NULL
   OR "IBB" IS NULL
   OR "HBP" IS NULL
   OR "SH" IS NULL
   OR "SF" IS NULL
   OR "GIDP" IS NULL;
UPDATE "battingpost"
SET "G"       = "G" + 1,
    "AB"      = "AB" + 1,
    "R"       = "R" + 1,
    "H"       = "H" + 1,
    "SecondB" = "SecondB" + 1,
    "ThirdB"  = "ThirdB" + 1,
    "HR"      = "HR" + 1,
    "RBI"     = "RBI" + 1,
    "SB"      = "SB" + 1,
    "CS"      = "CS" + 1,
    "BB"      = "BB" + 1,
    "SO"      = "SO" + 1,
    "IBB"     = "IBB" + 1,
    "HBP"     = "HBP" + 1,
    "SH"      = "SH" + 1,
    "SF"      = "SF" + 1,
    "GIDP"    = "GIDP" + 1;
DELETE FROM "els_teamnames" WHERE "id" IS NULL OR "lgid" IS NULL OR "teamid" IS NULL OR "franchid" IS NULL OR "name" IS NULL OR "park" IS NULL;
UPDATE "els_teamnames" SET "id" = "id" + 1;
DELETE FROM "fielding" WHERE "yearID" IS NULL OR "yearID" = 0 OR "stint" IS NULL OR "stint" = 0;
DELETE FROM "fielding" WHERE "playerID" IS NULL OR "teamID" IS NULL OR "lgID" IS NULL OR "POS" IS NULL OR "G" IS NULL OR "PO" IS NULL OR "A" IS NULL OR "E" IS NULL OR "DP" IS NULL;
UPDATE "fielding" SET "G" = "G" + 1, "PO" = "PO" + 1, "A" = "A" + 1, "E" = "E" + 1, "DP" = "DP" + 1;
DELETE FROM "fieldingof" WHERE "yearID" IS NULL OR "yearID" = 0 OR "stint" IS NULL OR "stint" = 0;
DELETE FROM "fieldingof" WHERE "playerID" IS NULL OR "Glf" IS NULL OR "Gcf" IS NULL OR "Grf" IS NULL;
UPDATE "fieldingof" SET "Glf" = "Glf" + 1, "Gcf" = "Gcf" + 1, "Grf" = "Grf" + 1;
DELETE
FROM "fieldingpost"
WHERE "yearID" IS NULL
   OR "yearID" = 0;
DELETE
FROM "fieldingpost"
WHERE "playerID" IS NULL
   OR "teamID" IS NULL
   OR "lgID" IS NULL
   OR "round" IS NULL
   OR "POS" IS NULL
   OR "G" IS NULL
   OR "GS" IS NULL
   OR "InnOuts" IS NULL
   OR "PO" IS NULL
   OR "A" IS NULL
   OR "E" IS NULL
   OR "DP" IS NULL
   OR "TP" IS NULL;
UPDATE "fieldingpost"
SET "G"       = "G" + 1,
    "GS"      = "GS" + 1,
    "InnOuts" = "InnOuts" + 1,
    "PO"      = "PO" + 1,
    "A"       = "A" + 1,
    "E"       = "E" + 1,
    "DP"      = "DP" + 1,
    "TP"      = "TP" + 1;
DELETE FROM "halloffame" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "halloffame" WHERE "hofID" IS NULL OR "votedBy" IS NULL OR "ballots" IS NULL OR "needed" IS NULL OR "votes" IS NULL OR "inducted" IS NULL OR "category" IS NULL;
UPDATE "halloffame" SET "ballots" = "ballots" + 1, "needed" = "needed" + 1, "votes" = "votes" + 1;
DELETE FROM "managers" WHERE "yearID" IS NULL OR "yearID" = 0 OR "inseason" IS NULL OR "inseason" = 0;
DELETE FROM "managers" WHERE "managerID" IS NULL OR "teamID" IS NULL OR "lgID" IS NULL OR "G" IS NULL OR "W" IS NULL OR "L" IS NULL OR "rank" IS NULL OR "plyrMgr" IS NULL;
UPDATE "managers" SET "G" = "G" + 1, "W" = "W" + 1, "L" = "L" + 1, "rank" = "rank" + 1;
DELETE FROM "managershalf" WHERE "yearID" IS NULL OR "yearID" = 0 OR "half" IS NULL OR "half" = 0;
DELETE FROM "managershalf" WHERE "managerID" IS NULL OR "teamID" IS NULL OR "lgID" IS NULL OR "inseason" IS NULL OR "G" IS NULL OR "W" IS NULL OR "L" IS NULL OR "rank" IS NULL;
UPDATE "managershalf" SET "inseason" = "inseason" + 1, "G" = "G" + 1, "W" = "W" + 1, "L" = "L" + 1, "rank" = "rank" + 1;
DELETE FROM "pitching" WHERE "yearID" IS NULL OR "yearID" = 0 OR "stint" IS NULL OR "stint" = 0;
DELETE
FROM "pitching"
WHERE "playerID" IS NULL
   OR "teamID" IS NULL
   OR "lgID" IS NULL
   OR "W" IS NULL
   OR "L" IS NULL
   OR "G" IS NULL
   OR "GS" IS NULL
   OR "CG" IS NULL
   OR "SHO" IS NULL
   OR "SV" IS NULL
   OR "IPouts" IS NULL
   OR "H" IS NULL
   OR "ER" IS NULL
   OR "HR" IS NULL
   OR "BB" IS NULL
   OR "SO" IS NULL
   OR "BAOpp" IS NULL
   OR "ERA" IS NULL
   OR "WP" IS NULL
   OR "HBP" IS NULL
   OR "BK" IS NULL
   OR "BFP" IS NULL
   OR "GF" IS NULL
   OR "R" IS NULL;
UPDATE "pitching"
SET "W"      = "W" + 1,
    "L"      = "L" + 1,
    "G"      = "G" + 1,
    "GS"     = "GS" + 1,
    "CG"     = "CG" + 1,
    "SHO"    = "SHO" + 1,
    "SV"     = "SV" + 1,
    "IPouts" = "IPouts" + 1,
    "H"      = "H" + 1,
    "ER"     = "ER" + 1,
    "HR"     = "HR" + 1,
    "BB"     = "BB" + 1,
    "SO"     = "SO" + 1,
    "BAOpp"  = "BAOpp" + 1,
    "ERA"    = "ERA" + 1,
    "WP"     = "WP" + 1,
    "HBP"    = "HBP" + 1,
    "BK"     = "BK" + 1,
    "BFP"    = "BFP" + 1,
    "GF"     = "GF" + 1,
    "R"      = "R" + 1;
DELETE
FROM "pitchingpost"
WHERE "yearID" IS NULL
   OR "yearID" = 0;
DELETE
FROM "pitchingpost"
WHERE "playerID" IS NULL
   OR "round" IS NULL
   OR "teamID" IS NULL
   OR "lgID" IS NULL
   OR "W" IS NULL
   OR "L" IS NULL
   OR "G" IS NULL
   OR "GS" IS NULL
   OR "CG" IS NULL
   OR "SHO" IS NULL
   OR "SV" IS NULL
   OR "IPouts" IS NULL
   OR "H" IS NULL
   OR "ER" IS NULL
   OR "HR" IS NULL
   OR "BB" IS NULL
   OR "SO" IS NULL
   OR "BAOpp" IS NULL
   OR "ERA" IS NULL
   OR "IBB" IS NULL
   OR "WP" IS NULL
   OR "HBP" IS NULL
   OR "BK" IS NULL
   OR "BFP" IS NULL
   OR "GF" IS NULL
   OR "R" IS NULL
   OR "SH" IS NULL
   OR "SF" IS NULL
   OR "GIDP" IS NULL;
UPDATE "pitchingpost"
SET "W"      = "W" + 1,
    "L"      = "L" + 1,
    "G"      = "G" + 1,
    "GS"     = "GS" + 1,
    "CG"     = "CG" + 1,
    "SHO"    = "SHO" + 1,
    "SV"     = "SV" + 1,
    "IPouts" = "IPouts" + 1,
    "H"      = "H" + 1,
    "ER"     = "ER" + 1,
    "HR"     = "HR" + 1,
    "BB"     = "BB" + 1,
    "SO"     = "SO" + 1,
    "BAOpp"  = "BAOpp" + 1,
    "ERA"    = "ERA" + 1,
    "IBB"    = "IBB" + 1,
    "WP"     = "WP" + 1,
    "HBP"    = "HBP" + 1,
    "BK"     = "BK" + 1,
    "BFP"    = "BFP" + 1,
    "GF"     = "GF" + 1,
    "R"      = "R" + 1,
    "SH"     = "SH" + 1,
    "SF"     = "SF" + 1,
    "GIDP"   = "GIDP" + 1;
DELETE FROM "players" WHERE "lahmanID" IS NULL OR "lahmanID" = 0;
DELETE FROM "players" WHERE "playerID" IS NULL OR "birthYear" IS NULL OR "birthMonth" IS NULL OR "birthDay" IS NULL OR "birthCountry" IS NULL OR "birthState" IS NULL OR "birthCity" IS NULL OR "nameFirst" IS NULL OR "nameLast" IS NULL OR "nameGiven" IS NULL OR "weight" IS NULL OR "height" IS NULL OR "bats" IS NULL OR "throws" IS NULL OR "debut" IS NULL OR "finalGame" IS NULL OR "lahman40ID" IS NULL OR "lahman45ID" IS NULL OR "retroID" IS NULL OR "holtzID" IS NULL OR "bbrefID" IS NULL;
UPDATE "players" SET "birthYear" = "birthYear" + 1, "birthMonth" = "birthMonth" + 1, "birthDay" = "birthDay" + 1, "weight" = "weight" + 1, "height" = "height" + 1;
DELETE FROM "salaries" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "salaries" WHERE "teamID" IS NULL OR "lgID" IS NULL OR "playerID" IS NULL OR "salary" IS NULL;
UPDATE "salaries" SET "salary" = "salary" + 1;
DELETE FROM "schools" WHERE "schoolID" IS NULL OR "schoolName" IS NULL OR "schoolCity" IS NULL OR "schoolState" IS NULL OR "schoolNick" IS NULL;
DELETE FROM "schoolsplayers" WHERE "playerID" IS NULL OR "schoolID" IS NULL OR "yearMin" IS NULL OR "yearMax" IS NULL;
UPDATE "schoolsplayers" SET "yearMin" = "yearMin" + 1, "yearMax" = "yearMax" + 1;
DELETE
FROM "seriespost"
WHERE "yearID" IS NULL
   OR "yearID" = 0;
DELETE
FROM "seriespost"
WHERE "round" IS NULL
   OR "teamIDwinner" IS NULL
   OR "lgIDwinner" IS NULL
   OR "teamIDloser" IS NULL
   OR "lgIDloser" IS NULL
   OR "wins" IS NULL
   OR "losses" IS NULL
   OR "ties" IS NULL;
UPDATE "seriespost"
SET "wins"   = "wins" + 1,
    "losses" = "losses" + 1,
    "ties"   = "ties" + 1;
DELETE
FROM "teams"
WHERE "yearID" IS NULL
   OR "yearID" = 0;
DELETE
FROM "teams"
WHERE "lgID" IS NULL
   OR "teamID" IS NULL
   OR "franchID" IS NULL
   OR "Rank" IS NULL
   OR "G" IS NULL
   OR "Ghome" IS NULL
   OR "W" IS NULL
   OR "L" IS NULL
   OR "LgWin" IS NULL
   OR "WSWin" IS NULL
   OR "R" IS NULL
   OR "AB" IS NULL
   OR "H" IS NULL
   OR "SecondB" IS NULL
   OR "ThirdB" IS NULL
   OR "HR" IS NULL
   OR "BB" IS NULL
   OR "SO" IS NULL
   OR "SB" IS NULL
   OR "CS" IS NULL
   OR "RA" IS NULL
   OR "ER" IS NULL
   OR "ERA" IS NULL
   OR "CG" IS NULL
   OR "SHO" IS NULL
   OR "SV" IS NULL
   OR "IPouts" IS NULL
   OR "HA" IS NULL
   OR "HRA" IS NULL
   OR "BBA" IS NULL
   OR "SOA" IS NULL
   OR "E" IS NULL
   OR "DP" IS NULL
   OR "FP" IS NULL
   OR "name" IS NULL
   OR "park" IS NULL
   OR "attendance" IS NULL
   OR "BPF" IS NULL
   OR "PPF" IS NULL
   OR "teamIDBR" IS NULL
   OR "teamIDlahman45" IS NULL
   OR "teamIDretro" IS NULL;
UPDATE "teams"
SET "Rank"       = "Rank" + 1,
    "G"          = "G" + 1,
    "Ghome"      = "Ghome" + 1,
    "W"          = "W" + 1,
    "L"          = "L" + 1,
    "R"          = "R" + 1,
    "AB"         = "AB" + 1,
    "H"          = "H" + 1,
    "SecondB"    = "SecondB" + 1,
    "ThirdB"     = "ThirdB" + 1,
    "HR"         = "HR" + 1,
    "BB"         = "BB" + 1,
    "SO"         = "SO" + 1,
    "SB"         = "SB" + 1,
    "CS"         = "CS" + 1,
    "RA"         = "RA" + 1,
    "ER"         = "ER" + 1,
    "ERA"        = "ERA" + 1,
    "CG"         = "CG" + 1,
    "SHO"        = "SHO" + 1,
    "SV"         = "SV" + 1,
    "IPouts"     = "IPouts" + 1,
    "HA"         = "HA" + 1,
    "HRA"        = "HRA" + 1,
    "BBA"        = "BBA" + 1,
    "SOA"        = "SOA" + 1,
    "E"          = "E" + 1,
    "DP"         = "DP" + 1,
    "FP"         = "FP" + 1,
    "attendance" = "attendance" + 1,
    "BPF"        = "BPF" + 1,
    "PPF"        = "PPF" + 1;
DELETE
FROM "teamsfranchises"
WHERE "franchID" IS NULL
   OR "franchName" IS NULL
   OR "active" IS NULL;
DELETE FROM "teamshalf" WHERE "yearID" IS NULL OR "yearID" = 0;
DELETE FROM "teamshalf" WHERE "lgID" IS NULL OR "teamID" IS NULL OR "Half" IS NULL OR "divID" IS NULL OR "DivWin" IS NULL OR "Rank" IS NULL OR "G" IS NULL OR "W" IS NULL OR "L" IS NULL;
UPDATE "teamshalf" SET "Rank" = "Rank" + 1, "G" = "G" + 1, "W" = "W" + 1, "L" = "L" + 1;

DELETE FROM "L_AIRLINE_ID" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_AIRLINE_ID" WHERE "Description" IS NULL;
DELETE FROM "L_AIRPORT" WHERE "Code" IS NULL OR "Description" IS NULL;
DELETE FROM "L_AIRPORT_ID" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_AIRPORT_ID" WHERE "Description" IS NULL;
DELETE FROM "L_AIRPORT_SEQ_ID" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_AIRPORT_SEQ_ID" WHERE "Description" IS NULL;
DELETE
FROM "L_CANCELLATION"
WHERE "Code" IS NULL
   OR "Description" IS NULL;
DELETE FROM "L_CITY_MARKET_ID" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_CITY_MARKET_ID" WHERE "Description" IS NULL;
DELETE
FROM "L_DEPARRBLK"
WHERE "Code" IS NULL
   OR "Description" IS NULL;
DELETE FROM "L_DISTANCE_GROUP_250" WHERE "Code" IS NULL OR "Code" = 0;
DELETE
FROM "L_DISTANCE_GROUP_250"
WHERE "Description" IS NULL;
DELETE FROM "L_DIVERSIONS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_DIVERSIONS" WHERE "Description" IS NULL;
DELETE FROM "L_MONTHS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_MONTHS" WHERE "Description" IS NULL;
DELETE FROM "L_ONTIME_DELAY_GROUPS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_ONTIME_DELAY_GROUPS" WHERE "Description" IS NULL;
DELETE FROM "L_QUARTERS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_QUARTERS" WHERE "Description" IS NULL;
DELETE FROM "L_STATE_ABR_AVIATION" WHERE "Code" IS NULL OR "Description" IS NULL;
DELETE FROM "L_STATE_FIPS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_STATE_FIPS" WHERE "Description" IS NULL;
DELETE FROM "L_UNIQUE_CARRIERS" WHERE "Code" IS NULL OR "Description" IS NULL;
DELETE FROM "L_WEEKDAYS" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_WEEKDAYS" WHERE "Description" IS NULL;
DELETE FROM "L_WORLD_AREA_CODES" WHERE "Code" IS NULL OR "Code" = 0;
DELETE FROM "L_WORLD_AREA_CODES" WHERE "Description" IS NULL;
DELETE
FROM "L_YESNO_RESP"
WHERE "Code" IS NULL
   OR "Code" = 0;
DELETE
FROM "L_YESNO_RESP"
WHERE "Description" IS NULL;
DELETE
FROM "On_Time_On_Time_Performance_2016_1"
WHERE "Year" IS NULL
   OR "Quarter" IS NULL
   OR "Month" IS NULL
   OR "DayofMonth" IS NULL
   OR "DayOfWeek" IS NULL
   OR "UniqueCarrier" IS NULL
   OR "AirlineID" IS NULL
   OR "Carrier" IS NULL
   OR "TailNum" IS NULL
   OR "FlightNum" IS NULL
   OR "OriginAirportID" IS NULL
   OR "OriginAirportSeqID" IS NULL
   OR "OriginCityMarketID" IS NULL
   OR "Origin" IS NULL
   OR "OriginCityName" IS NULL
   OR "OriginState" IS NULL
   OR "OriginStateFips" IS NULL
   OR "OriginStateName" IS NULL
   OR "OriginWac" IS NULL
   OR "DestAirportID" IS NULL
   OR "DestAirportSeqID" IS NULL
   OR "DestCityMarketID" IS NULL
   OR "Dest" IS NULL
   OR "DestCityName" IS NULL
   OR "DestState" IS NULL
   OR "DestStateFips" IS NULL
   OR "DestStateName" IS NULL
   OR "DestWac" IS NULL
   OR "CRSDepTime" IS NULL
   OR "DepTime" IS NULL
   OR "DepDelay" IS NULL
   OR "DepDelayMinutes" IS NULL
   OR "DepDel15" IS NULL
   OR "DepartureDelayGroups" IS NULL
   OR "DepTimeBlk" IS NULL
   OR "TaxiOut" IS NULL
   OR "WheelsOff" IS NULL
   OR "WheelsOn" IS NULL
   OR "TaxiIn" IS NULL
   OR "CRSArrTime" IS NULL
   OR "ArrTime" IS NULL
   OR "ArrDelay" IS NULL
   OR "ArrDelayMinutes" IS NULL
   OR "ArrDel15" IS NULL
   OR "ArrivalDelayGroups" IS NULL
   OR "ArrTimeBlk" IS NULL
   OR "Cancelled" IS NULL
   OR "Diverted" IS NULL
   OR "CRSElapsedTime" IS NULL
   OR "ActualElapsedTime" IS NULL
   OR "AirTime" IS NULL
   OR "Flights" IS NULL
   OR "Distance" IS NULL
   OR "DistanceGroup" IS NULL
   OR "DivAirportLandings" IS NULL;
UPDATE "On_Time_On_Time_Performance_2016_1"
SET "Year"                 = "Year" + 1,
    "Quarter"              = "Quarter" + 1,
    "Month"                = "Month" + 1,
    "DayofMonth"           = "DayofMonth" + 1,
    "DayOfWeek"            = "DayOfWeek" + 1,
    "AirlineID"            = "AirlineID" + 1,
    "FlightNum"            = "FlightNum" + 1,
    "OriginAirportID"      = "OriginAirportID" + 1,
    "OriginAirportSeqID"   = "OriginAirportSeqID" + 1,
    "OriginCityMarketID"   = "OriginCityMarketID" + 1,
    "OriginStateFips"      = "OriginStateFips" + 1,
    "OriginWac"            = "OriginWac" + 1,
    "DestAirportID"        = "DestAirportID" + 1,
    "DestAirportSeqID"     = "DestAirportSeqID" + 1,
    "DestCityMarketID"     = "DestCityMarketID" + 1,
    "DestStateFips"        = "DestStateFips" + 1,
    "DestWac"              = "DestWac" + 1,
    "CRSDepTime"           = "CRSDepTime" + 1,
    "DepTime"              = "DepTime" + 1,
    "DepDelay"             = "DepDelay" + 1,
    "DepDelayMinutes"      = "DepDelayMinutes" + 1,
    "DepDel15"             = "DepDel15" + 1,
    "DepartureDelayGroups" = "DepartureDelayGroups" + 1,
    "TaxiOut"              = "TaxiOut" + 1,
    "WheelsOff"            = "WheelsOff" + 1,
    "WheelsOn"             = "WheelsOn" + 1,
    "TaxiIn"               = "TaxiIn" + 1,
    "CRSArrTime"           = "CRSArrTime" + 1,
    "ArrTime"              = "ArrTime" + 1,
    "ArrDelay"             = "ArrDelay" + 1,
    "ArrDelayMinutes"      = "ArrDelayMinutes" + 1,
    "ArrDel15"             = "ArrDel15" + 1,
    "ArrivalDelayGroups"   = "ArrivalDelayGroups" + 1,
    "Cancelled"            = "Cancelled" + 1,
    "Diverted"             = "Diverted" + 1,
    "CRSElapsedTime"       = "CRSElapsedTime" + 1,
    "ActualElapsedTime"    = "ActualElapsedTime" + 1,
    "AirTime"              = "AirTime" + 1,
    "Flights"              = "Flights" + 1,
    "Distance"             = "Distance" + 1,
    "DistanceGroup"        = "DistanceGroup" + 1,
    "DivAirportLandings"   = "DivAirportLandings" + 1;

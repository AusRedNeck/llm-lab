@echo off
REM Gap-fill: cosmopedia + open-web-math, using the pattern form that works.
REM hf quirk: --include's value is taken as a positional filename pattern, and
REM '*' there does not cross '/'. So pass ONE flat pattern positionally:
REM   "*.parquet" (all parquet) or "*enwiki*" (subset by name).
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
set LOG=logs\download_cosmopedia_owm.log
echo === gap-fill downloads start %date% %time% === > %LOG%

echo --- cosmopedia (all configs) --- >> %LOG%
hf download HuggingFaceTB/cosmopedia --type dataset --local-dir data\cosmopedia "*.parquet" --max-workers 6 >> %LOG% 2>&1
echo === cosmopedia done %date% %time% === >> %LOG%

echo --- open-web-math --- >> %LOG%
hf download open-web-math/open-web-math --type dataset --local-dir data\open_web_math "*.parquet" --max-workers 6 >> %LOG% 2>&1
echo === open-web-math done %date% %time% === >> %LOG%
echo === GAP-FILL DONE %date% %time% === >> %LOG%

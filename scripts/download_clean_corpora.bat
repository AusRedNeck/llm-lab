@echo off
REM Download the clean/curated corpora: cosmopedia + finewiki(en) + open-web-math.
REM ~147 GiB total, ungated. Lands under llm-lab\data on D:.
REM NOTE: hf's --include takes ONE glob. Multiple patterns become positional
REM FILENAMES and are silently taken as specific files — one pattern per call.
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
set LOG=logs\download_clean_corpora.log
echo === clean corpus downloads start %date% %time% === > %LOG%

echo --- cosmopedia (86 GiB: textbook-style synthetic prose, all configs) --- >> %LOG%
hf download HuggingFaceTB/cosmopedia --type dataset --local-dir data\cosmopedia --include "data/*" --max-workers 8 >> %LOG% 2>&1
echo === cosmopedia done %date% %time% === >> %LOG%

echo --- finewiki en (35 GiB: Wikipedia, cleaned) --- >> %LOG%
hf download HuggingFaceFW/finewiki --type dataset --local-dir data\finewiki --include "data/enwiki/*" --max-workers 8 >> %LOG% 2>&1
echo === finewiki done %date% %time% === >> %LOG%

echo --- open-web-math (26 GiB: math text) --- >> %LOG%
hf download open-web-math/open-web-math --type dataset --local-dir data\open_web_math --include "data/*" --max-workers 8 >> %LOG% 2>&1
echo === open-web-math done %date% %time% === >> %LOG%

echo === ALL DOWNLOADS DONE %date% %time% === >> %LOG%

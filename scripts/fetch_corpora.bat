@echo off
REM Fetch the curated corpora (cosmopedia + finewiki-en + open-web-math) ~147 GiB.
REM Python API path — patterns validated first, resumable, no CLI quoting traps.
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === corpus fetch start %date% %time% === > logs\fetch_corpus.log
python -u scripts\fetch_corpus.py >> logs\fetch_corpus.log 2>&1
echo === corpus fetch exit %errorlevel% at %date% %time% === >> logs\fetch_corpus.log

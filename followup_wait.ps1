# Wait for the running 16k encode to finish, then run the gap-fill + merge + verify.
#
# Why: worker part03 died mid-run, so shards 33-42 are missing and the parent will
# deliberately REFUSE to merge. This waiter hands off to a second pass that reads
# the plan off the parts on disk, encodes only the gap, merges in corpus order and
# verifies — so the corpus finishes whether or not anyone is watching.
#
# Guards (both must hold before it touches anything):
#   1. the first run has printed its "encode exit" marker
#   2. no tokenize_owt_16k process is alive
# The process check matches python.exe only, so this waiter cannot match itself
# (the earlier pattern-kill incident: a filter that matches its own command line).

$root   = 'D:\Projects\llm-lab'
$log    = Join-Path $root 'logs\16k_tokenize_par.log'
$marker = Join-Path $root 'logs\followup.log'

function Write-Marker($text) {
    "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))  $text" | Add-Content -Path $marker
}

Write-Marker "waiter started; holding until the current encode finishes"

while ($true) {
    $finished = Select-String -Path $log -Pattern 'encode exit' -Quiet -ErrorAction SilentlyContinue
    $busy = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*tokenize_owt_16k*' })
    if ($finished -and $busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
}

Write-Marker "encode finished and no workers alive; starting gap-fill pass"
Set-Location $root

# Second pass: plan from parts on disk, fill the gap, merge, verify.
& python -u tokenize_owt_16k.py --workers 8 --status-every 60 *>> (Join-Path $root 'logs\16k_tokenize_followup.log')
$rc = $LASTEXITCODE
Write-Marker "encode pass exited $rc"

if ($rc -eq 0) {
    & python -u verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin `
        --vocab data/bpe_owt16k.json `
        --src data/openwebtext/shards/train-00000-of-00080.txt `
        --manifest *>> (Join-Path $root 'logs\16k_tokenize_followup.log')
    $vrc = $LASTEXITCODE
    if ($vrc -eq 0) {
        Write-Marker "VERIFIED OK - corpus complete and 7/7 checks green"
    } else {
        Write-Marker "VERIFY FAILED (exit $vrc) - see logs\16k_tokenize_followup.log"
    }
} else {
    Write-Marker "ENCODE INCOMPLETE (exit $rc) - rerun run_16k_full_par.bat; parts are kept"
}

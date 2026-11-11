# MODEL d UNDER 5-FOLD CV: 5 fine-tune runs, NO new encoders.
#
# d = contrastive-pretrained init + lambda_inv 0.4. It differs from model b ONLY
# by the invariance loss, so it REUSES b's five per-fold encoders. The encoder
# guard in dual_protocol_training.py compares (protocol, fold, split_seed,
# n_folds) - not the training objective - so the same encoder is correct for
# both arms, and that is scientifically right: the encoder is pretrained on that
# fold's TRAIN side regardless of what is fine-tuned on top.
#
# Consequence: this arm is runs only, ~5-8 h, not the ~18 h quoted on 9-12-26
# when no encoders existed.
#
# NO TIME LIMIT (Q, 2026-09-15). There is no stop-hour gate. The safety rails
# that remain are the ones that protect the machine and the data, not the clock:
#   - strictly sequential, with a <=70C cooldown gate before each run
#     (a laptop crash on 8-19-26 followed two concurrent jobs at 88C)
#   - a 25-min no-log-growth stall watchdog (the 2h34m hang of 9-1-26)
#   - a generous per-run hard timeout, so a wedged run cannot block the queue
#     forever
#   - sleep guard held for the whole session
#
# RUNTIME EXPECTATION: arm c (random init, lambda=0.4) ran 78-100 min/fold and
# arm b (pretrained, lambda=0) ran 22-48 min. d is pretrained AND lambda=0.4:
# the invariance term roughly doubles per-epoch cost while pretrained init cuts
# epochs, so expect ~60-90 min/fold. Do not trust that until fold 0 measures it
# - a one-fold extrapolation was wrong by 1.8x on 9-14-26.
#
# ASCII ONLY (harness bug #1, 2026-09-12). Parse-check before launching.

$ErrorActionPreference = "Continue"
$py      = "C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe"
$root    = "C:\Users\sakso\Downloads\NCKU\Proj\ECG_project"
$logdir  = "$root\runs\_logs"
$NFolds  = 5
$SplitSeed = 0
$Seed    = 42
$Lambda  = "0.4"
$StallMin = 25
$HardMin  = 240     # per-run ceiling; ~3x the worst plausible d fold
Set-Location $root
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$Power = Add-Type -MemberDefinition $sig -Name PowerUtilKFoldD -Namespace Win32 -PassThru
function Assert-Awake { [void]$Power::SetThreadExecutionState([uint32]2147483713) }
Assert-Awake

$t0All = Get-Date
Write-Output "=============================================================="
Write-Output " MODEL d, 5-FOLD CV - 5 runs, reusing b's per-fold encoders"
Write-Output " started : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " STOP AT : none (no time limit)"
Write-Output "=============================================================="

# ---- PRE-FLIGHT: every encoder must exist AND declare its own fold ----------
# A missing or mis-labelled encoder must stop the session BEFORE any GPU work,
# not fail run-by-run. Fold 0's train set contains 100% of every other fold's
# val, so a wrong-fold encoder is total validation leakage (9-12-26).
$missing = @()
foreach ($fold in 0..($NFolds - 1)) {
    $enc  = "$root\model_saved\pretrained_encoder_kfold_fold$fold.pth"
    $meta = "$root\model_saved\pretrained_encoder_kfold_fold$fold.meta.json"
    if (-not (Test-Path $enc))  { $missing += "fold ${fold}: .pth missing";  continue }
    if (-not (Test-Path $meta)) { $missing += "fold ${fold}: .meta.json missing"; continue }
    $m = Get-Content $meta -Raw | ConvertFrom-Json
    if ($m.protocol -ne "kfold" -or [int]$m.fold -ne $fold -or
        [int]$m.n_folds -ne $NFolds -or [int]$m.split_seed -ne $SplitSeed) {
        $missing += "fold ${fold}: sidecar mismatch (protocol=$($m.protocol) fold=$($m.fold) n_folds=$($m.n_folds) split_seed=$($m.split_seed))"
    }
}
if ($missing.Count -gt 0) {
    Write-Output ""
    Write-Output "PRE-FLIGHT FAILED - not starting:"
    $missing | ForEach-Object { Write-Output "    $_" }
    Write-Output "Build the missing encoder(s) with scripts/run_kfold_b.ps1 first."
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    exit 1
}
Write-Output "pre-flight: all $NFolds encoders present and fold-labelled correctly"

function Wait-Cool {
    param([int]$Target = 70, [int]$MaxMin = 30)
    $deadline = (Get-Date).AddMinutes($MaxMin)
    while ((Get-Date) -lt $deadline) {
        Assert-Awake
        $t = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
        if ($t -le $Target) { Write-Output "    GPU ${t}C - ok"; return }
        Write-Output "    GPU ${t}C - waiting"
        Start-Sleep -Seconds 60
    }
    Write-Output "    cooldown timed out; starting anyway"
}

function Invoke-Watched {
    param([string[]]$ArgList, [string]$Log)
    if (Test-Path $Log) { Remove-Item $Log -Force }
    $errLog = [System.IO.Path]::ChangeExtension($Log, ".err.log")
    $proc = Start-Process -FilePath $py -ArgumentList $ArgList -PassThru `
        -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError $errLog
    $hard = (Get-Date).AddMinutes($HardMin)
    $lastSize = -1; $lastGrew = Get-Date
    while ($true) {
        Start-Sleep -Seconds 60
        Assert-Awake
        if ($proc.HasExited) { return "exit$($proc.ExitCode)" }
        $sz = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
        if ($sz -ne $lastSize) { $lastSize = $sz; $lastGrew = Get-Date }
        if (((Get-Date) - $lastGrew).TotalMinutes -ge $StallMin) {
            Write-Output "    STALLED - no log growth for $StallMin min; killing."
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "stalled"
        }
        if ((Get-Date) -ge $hard) {
            Write-Output "    HARD TIMEOUT at $HardMin min - killing."
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "timeout"
        }
    }
}

$done = 0; $failed = 0
foreach ($fold in 0..($NFolds - 1)) {
    Assert-Awake
    $enc = "model_saved\pretrained_encoder_kfold_fold$fold.pth"
    $tag = "kfold_pretrained_inv04_sampler_fold$fold"
    $rundir = "$root\runs\${tag}_seed$Seed"

    Write-Output ""
    Write-Output "=== [fold $fold] MODEL d   $(Get-Date -Format 'HH:mm')"
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    already complete - skipping"; $done++; continue
    }

    Wait-Cool
    $t0 = Get-Date
    $log = "$logdir\${tag}_seed$Seed.log"
    $r = Invoke-Watched -ArgList @("-u", "code\dual_protocol_training.py",
        "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
        "--split-seed", "$SplitSeed", "--seed", "$Seed",
        "--lambda-inv", $Lambda, "--pretrained", $enc, "--tag", $tag) -Log $log
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    finished in $mins min ($r)"; $done++
        & $py "code\run_summary.py" --tag $tag --seed $Seed
    } else {
        Write-Output "    RUN FAILED after $mins min ($r). Tail:"
        Get-Content $log -Tail 12 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "      $_" }
        Get-Content ([System.IO.Path]::ChangeExtension($log, ".err.log")) -Tail 6 `
            -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "      ! $_" }
        $failed++
        if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force }
        # Stop rather than burning hours on the next fold with an unexplained
        # failure behind us (the fall-forward mistake of 2026-09-14).
        Write-Output "    Stopping - a failure needs a human, not more GPU time."
        break
    }
}

$elapsed = [math]::Round(((Get-Date) - $t0All).TotalMinutes, 1)
Write-Output ""
Write-Output "=============================================================="
Write-Output " MODEL d K-FOLD SESSION ENDED  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " folds completed $done   failed $failed   wall clock $elapsed min"
Write-Output "=============================================================="
$have = @()
foreach ($f in 0..($NFolds - 1)) {
    if (Test-Path "$root\runs\kfold_pretrained_inv04_sampler_fold${f}_seed$Seed\metrics.json") { $have += $f }
}
Write-Output (" d folds complete : {0}" -f ($have -join ","))
if ($have.Count -eq $NFolds) {
    Write-Output " ALL FOUR ARMS (a,b,c,d) ARE NOW AT 5 FOLDS."
    Write-Output " Analyse with: python code/kfold_ttests.py --arms a b c d"
    Write-Output "               python code/kfold_report.py --tag kfold_pretrained_inv04_sampler"
} else {
    Write-Output " Resume by re-running this script - it skips completed folds."
}
[void]$Power::SetThreadExecutionState([uint32]2147483648)
Write-Output " sleep guard released"

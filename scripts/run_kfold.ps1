# 5-FOLD CV on the pooled-beat (random 8:2) protocol - models a and c.
# NOT YET RUN as of 2026-09-11. Verify first with: python code/test_kfold.py
#
#   a. kfold_focal_sampler   lambda_inv 0.0
#   c. kfold_inv04_sampler   lambda_inv 0.4
#
# WHY: the co-advisor asked for 5-fold CV on the 8:2 random split. This is the
# literature-comparable presentation - LMUEBCNet (Xie & Lin 2023) used random
# 10-fold CV over pooled beats, explicitly "not subject-related". Reporting
# mean +/- std across folds makes this project's random-split arm directly
# comparable to that paper instead of comparable-in-spirit.
#
# FOLD-MAJOR, not arm-major: fold 0 for both arms, then fold 1, and so on. An
# interrupted session therefore leaves COMPLETE folds across both arms, which is
# what a paired fold-wise comparison needs. Arm-major would leave a ragged set -
# the same reasoning as the seed-major rule for the seed sweeps.
#
# SCOPE - a and c ONLY, deliberately. Models b and d each need a contrastive
# encoder pretrained on THAT FOLD's training side (CLAUDE.md 8-31-26: 78.4% of a
# pooled-split val set sits inside the inter-patient encoder's pretraining set).
# That is 5 x ~2 h = ~10 h of pretraining on top of the runs. a and c are
# random-init and need none.
#
# BUDGET: 10 runs. Measured random-split runs took 26-88 min (mean ~52) because
# early stopping rarely fires under leakage and most runs hit the 30-epoch cap.
# Expect ~9 h, worst case ~15 h.
#
# Harness as established: async launch + stall watchdog, thermal gate, sleep
# guard, per-run summary. Add a 09:00 stop gate if launching overnight.

param(
    # Which arms to run. Default both; pass -Arms a to do the baseline only.
    [string[]]$Arms = @("a", "c")
)

$ErrorActionPreference = "Continue"
$py     = "C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe"
$root   = "c:\Users\sakso\Downloads\NCKU\Proj\ECG_project"
$logdir = "$root\runs\_logs"
$NFolds = 5
$SplitSeed = 0
$Seed   = 42          # training seed held fixed; the FOLD is the varying factor
$StallMin = 20
Set-Location $root
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$Power = Add-Type -MemberDefinition $sig -Name PowerUtilKFold -Namespace Win32 -PassThru
function Assert-Awake { [void]$Power::SetThreadExecutionState([uint32]2147483713) }
Assert-Awake

Write-Output "=============================================================="
Write-Output " 5-FOLD CV - pooled-beat protocol, models a and c"
Write-Output " started : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " $NFolds folds x $($Arms.Count) arm(s), fold-major"
Write-Output "=============================================================="

function Wait-Cool {
    param([int]$Target = 70, [int]$MaxMin = 20)
    $deadline = (Get-Date).AddMinutes($MaxMin)
    while ((Get-Date) -lt $deadline) {
        Assert-Awake
        $t = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
        if ($t -le $Target) { Write-Output "    GPU ${t}C - ok"; return }
        Write-Output "    GPU ${t}C - waiting for <= ${Target}C"
        Start-Sleep -Seconds 60
    }
    Write-Output "    cooldown timed out; starting anyway"
}

function Invoke-Watched {
    param([string[]]$ArgList, [string]$Log, [int]$HardMin = 160)
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
            Write-Output "    HARD TIMEOUT - killing."
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "timeout"
        }
    }
}

$allArms = @(
    @{key = "a"; tag = "kfold_focal_sampler"; lam = "0"}
    @{key = "c"; tag = "kfold_inv04_sampler"; lam = "0.4"}
)
# NOTE: must NOT be named $arms - PowerShell variable names are case-INSENSITIVE,
# so $arms IS the [string[]]$Arms parameter, and assigning hashtables to a
# type-constrained string[] silently coerces each to "System.Collections.Hashtable".
$selected = @($allArms | Where-Object { $Arms -contains $_.key })
if ($selected.Count -eq 0) { Write-Output "No arms matched -Arms $Arms"; exit 1 }
Write-Output " arms this session: $(($selected | ForEach-Object { $_.key }) -join ', ')"

$done = 0; $failed = 0; $i = 0
foreach ($fold in 0..($NFolds - 1)) {
    foreach ($arm in $selected) {
        $i++
        Assert-Awake
        Write-Output ""
        Write-Output "=== [$i/$($NFolds * $selected.Count)] $($arm.key) fold $fold   $(Get-Date -Format 'HH:mm')"

        # NOTE: tag carries the fold, so runs land in runs/{tag}_fold{K}_seed{S}.
        $tag = "$($arm.tag)_fold$fold"
        $rundir = "$root\runs\${tag}_seed$Seed"
        if (Test-Path "$rundir\metrics.json") {
            Write-Output "    already complete - skipping"; continue
        }

        Wait-Cool
        $started = Get-Date
        $log = "$logdir\${tag}_seed$Seed.log"
        $r = Invoke-Watched -ArgList @("-u", "code\dual_protocol_training.py",
            "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
            "--split-seed", "$SplitSeed", "--seed", "$Seed",
            "--lambda-inv", "$($arm.lam)", "--tag", $tag) -Log $log
        $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

        if (Test-Path "$rundir\metrics.json") {
            Write-Output "    finished in $mins min ($r)"
            $done++
            & $py "code\run_summary.py" --tag $tag --seed $Seed
        } else {
            Write-Output "    FAILED after $mins min ($r). Tail:"
            Get-Content $log -Tail 12 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Output "      $_" }
            $failed++
            if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force }
        }
    }
}

Write-Output ""
Write-Output "=============================================================="
Write-Output " 5-FOLD CV ENDED  $(Get-Date -Format 'HH:mm')   done $done  failed $failed"
Write-Output "=============================================================="
foreach ($arm in $selected) {
    $have = @()
    foreach ($f in 0..($NFolds - 1)) {
        if (Test-Path "$root\runs\$($arm.tag)_fold${f}_seed$Seed\metrics.json") { $have += $f }
    }
    Write-Output (" {0} {1,-24} folds: {2}" -f $arm.key, $arm.tag, ($have -join ","))
}
[void]$Power::SetThreadExecutionState([uint32]2147483648)
Write-Output " sleep guard released"

# Option B - extend the four INTER-PATIENT arms from n=6 to n=10 (seeds 48-51).
#
#   a. focal_sampler             invariance_training.py            lambda 0.0
#   b. pretrained_focal_sampler  pretrained_training.py            lambda 0.0
#   c. focal_inv04_sampler       invariance_training.py            lambda 0.4
#   d. pretrained_inv04_sampler  pretrained_training.py            lambda 0.4
#
# SAME SCRIPTS that produced seeds 42-47 (verified from those runs' metrics.json:
# source_script, pretrained_path=pretrained_encoder_1D_aug.pth, sampler,
# lambda_inv all match), so the new seeds pool with the existing six rather than
# forming a separate arm. b and d take the DEFAULT --pretrained, which is
# pretrained_encoder_1D_aug.pth - NOT the cwtfix encoder, which lives in its own
# pretrained_cwtfix_* tags.
#
# WHY n=10: the target is d - b on INCART 3-class, currently +0.0189 at p=0.077.
# On the observed n=6 sd (0.0208) that projects to p=0.026 at n=9 and p=0.018 at
# n=10 (67% -> 72% power). d is the recommended model and d - b is the
# comparison that justifies recommending it over plain contrastive pretraining,
# so it is the single most valuable statistical upgrade left on these arms.
# NOT fixed by this: b - a (38-43% power even at n=10) and F1(F) (needs n~94).
#
# SEED-MAJOR: all four arms at seed 48, then 49, then 50, then 51. An interrupted
# run therefore leaves COMPLETE seeds across arms, which is what paired tests
# need. Mixing arms at unequal n is the error behind the 8-30-26 retraction.
#
# Carries the v2 harness: async launch + 20-min stall watchdog (a synchronous
# invoke let one hung process block the queue for 2 h 34 min on 2026-09-01),
# thermal cooldown gate, sleep guard, per-seed run_summary.

$ErrorActionPreference = "Continue"
$py     = "C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe"
$root   = "c:\Users\sakso\Downloads\NCKU\Proj\ECG_project"
$logdir = "$root\runs\_logs"
$StallMin = 20
Set-Location $root
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$Power = Add-Type -MemberDefinition $sig -Name PowerUtilN10 -Namespace Win32 -PassThru
$KEEP_AWAKE = [uint32]2147483713
function Assert-Awake { [void]$Power::SetThreadExecutionState($KEEP_AWAKE) }
Assert-Awake

Write-Output "=============================================================="
Write-Output " Option B - inter-patient arms n=6 -> n=10 (seeds 48-51)"
Write-Output " started : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " 16 runs, seed-major, ~25-30 min each -> ETA ~7-8 h"
Write-Output " watchdog: kill after $StallMin min of no log growth"
$RunBudget = 45
$StopAt = (Get-Date).Date.AddHours(9)
if ((Get-Date) -ge $StopAt) { $StopAt = $StopAt.AddDays(1) }
Write-Output " STOP AT : $($StopAt.ToString('yyyy-MM-dd HH:mm'))  (nothing starts after)"
Write-Output " runway  : $([math]::Round(($StopAt - (Get-Date)).TotalMinutes)) min, budget $RunBudget min/run"
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
    param([string[]]$ArgList, [string]$Log, [int]$HardMin = 120)
    if (Test-Path $Log) { Remove-Item $Log -Force }
    $errLog = [System.IO.Path]::ChangeExtension($Log, ".err.log")
    $proc = Start-Process -FilePath $py -ArgumentList $ArgList -PassThru `
        -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError $errLog
    $hardDeadline = (Get-Date).AddMinutes($HardMin)
    $lastSize = -1; $lastGrew = Get-Date
    while ($true) {
        Start-Sleep -Seconds 60
        Assert-Awake
        if ($proc.HasExited) { return "exit$($proc.ExitCode)" }
        $sz = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
        if ($sz -ne $lastSize) { $lastSize = $sz; $lastGrew = Get-Date }
        if (((Get-Date) - $lastGrew).TotalMinutes -ge $StallMin) {
            Write-Output "    STALLED - no log growth for $StallMin min."
            $dump = "$logdir\stall_pyspy_$(Get-Date -Format 'HHmmss').txt"
            try {
                & $py -m py_spy dump --pid $proc.Id 2>&1 | Out-File $dump -Encoding utf8
                Write-Output "    py-spy stack dumped -> $dump"
            } catch { Write-Output "    (py-spy unavailable; no stack captured)" }
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "stalled"
        }
        if ((Get-Date) -ge $hardDeadline) {
            Write-Output "    HARD TIMEOUT ($HardMin min) - killing."
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "timeout"
        }
    }
}

$arms = @(
    @{key="a"; script="code\invariance_training.py"; tag="focal_sampler";            lam="0"  }
    @{key="b"; script="code\pretrained_training.py"; tag="pretrained_focal_sampler"; lam="0"  }
    @{key="c"; script="code\invariance_training.py"; tag="focal_inv04_sampler";      lam="0.4"}
    @{key="d"; script="code\pretrained_training.py"; tag="pretrained_inv04_sampler"; lam="0.4"}
)

$jobs = @()
foreach ($seed in 48, 49, 50, 51) {
    foreach ($a in $arms) {
        $jobs += @{tag=$a.tag; script=$a.script; seed=$seed; lam=$a.lam; key=$a.key}
    }
}

$i = 0; $done = 0; $skipped = 0; $failed = 0
foreach ($j in $jobs) {
    $i++
    Assert-Awake
    Write-Output ""
    Write-Output "=== [$i/$($jobs.Count)] $($j.key) $($j.tag) seed $($j.seed)   $(Get-Date -Format 'HH:mm')"

    $rem = ($StopAt - (Get-Date)).TotalMinutes
    Write-Output ("    remaining {0} min" -f [math]::Round($rem))
    if ($rem -lt $RunBudget) {
        Write-Output "    STOP: $([math]::Round($rem)) min left < $RunBudget needed."
        Write-Output "    Deadline reached - not starting this or anything after it."
        break
    }

    $rundir = "$root\runs\$($j.tag)_seed$($j.seed)"
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    already complete - skipping"; $skipped++; continue
    }

    Wait-Cool
    $started = Get-Date
    $log = "$logdir\$($j.tag)_seed$($j.seed).log"
    $cmd = @("-u", $j.script, "--seed", "$($j.seed)", "--lambda-inv", "$($j.lam)",
             "--tag", "$($j.tag)")

    $r = Invoke-Watched -ArgList $cmd -Log $log
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    finished in $mins min ($r)"
        $done++
        & $py "code\run_summary.py" --tag $j.tag --seed $j.seed
    } else {
        Write-Output "    FAILED after $mins min ($r) - no metrics.json. Tail:"
        Get-Content $log -Tail 12 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "      $_" }
        $failed++
        if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force }
    }
}

Write-Output ""
Write-Output "=============================================================="
Write-Output " OPTION B ENDED  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " completed $done   skipped $skipped   failed $failed   queued $($jobs.Count)"
Write-Output "=============================================================="
foreach ($a in $arms) {
    $seeds = @()
    foreach ($s in 42, 43, 44, 45, 46, 47, 48, 49, 50, 51) {
        if (Test-Path "$root\runs\$($a.tag)_seed$s\metrics.json") { $seeds += $s }
    }
    Write-Output (" {0} {1,-30} n={2}" -f $a.key, $a.tag, $seeds.Count)
}
[void]$Power::SetThreadExecutionState([uint32]2147483648)
Write-Output " sleep guard released"

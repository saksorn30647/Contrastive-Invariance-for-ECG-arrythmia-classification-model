# MODEL b UNDER 5-FOLD CV: per-fold encoder + fine-tune, PAIRED fold by fold.
#
# Each fold needs its OWN contrastive encoder. Measured 2026-09-12: fold0's
# train side contains 100.0% of fold1.val and 100.0% of fold3.val, because
# fold0.train IS the other 80%. A single shared encoder would leak every other
# fold's validation set completely and nothing in the aggregate would show it.
# dual_protocol_training.py refuses an encoder whose sidecar fold != the run's.
#
# STRUCTURE - encoder K then run K, NOT all encoders then all runs. A pair is
# ~165 min (115 encoder + ~50 run) and is SELF-CONTAINED: whatever the 10:00
# gate cuts, what remains is COMPLETE folds of model b, not orphan encoders.
# Encoder-first ordering would spend the whole session producing zero runs.
#
# HONEST BUDGET: the full arm is 5 x 165 min = ~13.8 h. This session has ~8 h,
# so expect 2 complete folds (+ possibly a 3rd encoder). b at n=2 CANNOT be
# compared against arm a at n=5 - that is exactly the unequal-n error of
# 2026-08-30. This session is the first half of a two-session job.
#
# ASCII ONLY (harness bug #1, 2026-09-12). Parse-check before launching.

$ErrorActionPreference = "Continue"
$py      = "C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe"
$root    = "c:\Users\sakso\Downloads\NCKU\Proj\ECG_project"
$logdir  = "$root\runs\_logs"
$NFolds  = 5
$SplitSeed = 0
$Seed    = 42
$StallMin  = 25
$EncBudget = 140    # min of runway required to START an encoder
$RunBudget = 40     # min of runway required to START a fine-tune run.
                    # MEASURED 2026-09-14: fold 0 took 22 min (16 epochs,
                    # early-stopped - pretrained init converges faster than
                    # arm a, which mostly hit the 30-epoch cap). A capped
                    # 30-epoch b run is ~42 min at 1.4 min/epoch, so 45 is
                    # the honest gate. The old 75 was inherited from arm a
                    # and would have blocked a 22-min run with 60 min left,
                    # leaving an orphan encoder instead of a complete fold.
Set-Location $root
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$Power = Add-Type -MemberDefinition $sig -Name PowerUtilKFoldB -Namespace Win32 -PassThru
function Assert-Awake { [void]$Power::SetThreadExecutionState([uint32]2147483713) }
Assert-Awake

# 2026-09-14 session 2: finish the arm (folds 3-4). Remaining work is
# 2 encoders (~125 min each) + 2 runs (~40 min each) = ~5.5 h, so ~15:45.
# 17:00 leaves ~65 min of margin and still gates a runaway. NOTE the roll
# below: if this hour has already passed the gate jumps to TOMORROW, which
# would leave the session effectively ungated - set it ahead of 'now'.
# 17:00 was chosen assuming ~125 min encoders. They degraded across the
# day (125/124/124/141/164 min - ambient thermal, not fold-dependent), so
# fold 4's run was left 48 min for a ~48 min job and would have been
# killed mid-flight with its rundir deleted. 18:30 finishes the arm.
$StopAt = (Get-Date).Date.AddHours(18).AddMinutes(30)
if ((Get-Date) -ge $StopAt) { $StopAt = $StopAt.AddDays(1) }
function Remaining { return ($StopAt - (Get-Date)).TotalMinutes }

Write-Output "=============================================================="
Write-Output " MODEL b, 5-FOLD CV - per-fold encoder + fine-tune, paired"
Write-Output " started : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " STOP AT : $($StopAt.ToString('yyyy-MM-dd HH:mm'))"
Write-Output " runway  : $([math]::Round((Remaining))) min   (full arm needs ~830 min)"
Write-Output "=============================================================="

function Wait-Cool {
    param([int]$Target = 70, [int]$MaxMin = 25)
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
    param([string[]]$ArgList, [string]$Log, [int]$HardMin = 200)
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
        if ((Get-Date) -ge $hard -or (Get-Date) -ge $StopAt) {
            Write-Output "    TIMEOUT/DEADLINE - killing."
            try { taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
            return "deadline"
        }
    }
}

$encDone = 0; $runDone = 0; $failed = 0
foreach ($fold in 0..($NFolds - 1)) {
    $enc  = "model_saved\pretrained_encoder_kfold_fold$fold.pth"
    $meta = "model_saved\pretrained_encoder_kfold_fold$fold.meta.json"
    $tag  = "kfold_pretrained_focal_sampler_fold$fold"
    $rundir = "$root\runs\${tag}_seed$Seed"

    # ---- step 1: encoder for THIS fold --------------------------------------
    Assert-Awake
    Write-Output ""
    Write-Output "=== [fold $fold] ENCODER   $(Get-Date -Format 'HH:mm')  remaining $([math]::Round((Remaining))) min"
    if (Test-Path "$root\$enc") {
        Write-Output "    already exists - skipping"
    } elseif ((Remaining) -lt $EncBudget) {
        Write-Output "    STOP: $([math]::Round((Remaining))) min < $EncBudget needed. Ending session."
        break
    } else {
        Wait-Cool
        $t0 = Get-Date
        $log = "$logdir\contrastive_pretrain_kfold_fold$fold.log"
        $r = Invoke-Watched -ArgList @("-u", "code\contrastive_pretrain.py",
            "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
            "--split-seed", "$SplitSeed", "--seed", "$Seed", "--epochs", "50",
            "--save-path", $enc) -Log $log -HardMin 190
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ((Test-Path "$root\$enc") -and (Test-Path "$root\$meta")) {
            Write-Output "    encoder built in $mins min ($r)"; $encDone++
        } else {
            Write-Output "    ENCODER FAILED after $mins min ($r); removing partials."
            foreach ($f in $enc, $meta, "model_saved\pretrained_encoder_kfold_fold${fold}_loss.png") {
                if (Test-Path "$root\$f") { Remove-Item "$root\$f" -Force }
            }
            Get-Content $log -Tail 12 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Output "      $_" }
            $failed++; break
        }
    }

    # ---- step 2: fine-tune model b on THIS fold ------------------------------
    Assert-Awake
    Write-Output ""
    Write-Output "=== [fold $fold] MODEL b   $(Get-Date -Format 'HH:mm')  remaining $([math]::Round((Remaining))) min"
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    already complete - skipping"; continue
    }
    if ((Remaining) -lt $RunBudget) {
        Write-Output "    STOP: $([math]::Round((Remaining))) min < $RunBudget needed."
        Write-Output "    Encoder for fold $fold IS built and reusable next session."
        break
    }
    Wait-Cool
    $t0 = Get-Date
    $log = "$logdir\${tag}_seed$Seed.log"
    $r = Invoke-Watched -ArgList @("-u", "code\dual_protocol_training.py",
        "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
        "--split-seed", "$SplitSeed", "--seed", "$Seed",
        "--lambda-inv", "0", "--pretrained", $enc, "--tag", $tag) -Log $log -HardMin 160
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    finished in $mins min ($r)"; $runDone++
        & $py "code\run_summary.py" --tag $tag --seed $Seed
    } else {
        Write-Output "    RUN FAILED after $mins min ($r). Tail:"
        Get-Content $log -Tail 12 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "      $_" }
        $failed++
        if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force }
        # Always stop. Falling forward into the NEXT fold's 125-min encoder
        # while THIS fold has no run leaves an orphan encoder and costs a
        # complete fold - which is what happened at 04:01 on 2026-09-14 when a
        # stale encoder sidecar failed the guard. Complete folds are the whole
        # point of the pairing; a failure needs a human, not more GPU time.
        break
    }
}

Write-Output ""
Write-Output "=============================================================="
Write-Output " MODEL b K-FOLD SESSION ENDED  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " encoders built $encDone   b runs completed $runDone   failed $failed"
Write-Output "=============================================================="
$encs = @(); $runs = @()
foreach ($f in 0..($NFolds - 1)) {
    if (Test-Path "$root\model_saved\pretrained_encoder_kfold_fold$f.pth") { $encs += $f }
    if (Test-Path "$root\runs\kfold_pretrained_focal_sampler_fold${f}_seed$Seed\metrics.json") { $runs += $f }
}
Write-Output (" encoders present : {0}" -f ($encs -join ","))
Write-Output (" b folds complete : {0}" -f ($runs -join ","))
Write-Output " REMINDER: b at n<5 must NOT be compared against arm a at n=5."
Write-Output " Resume next session by re-running this script - it skips what exists."
[void]$Power::SetThreadExecutionState([uint32]2147483648)
Write-Output " sleep guard released"

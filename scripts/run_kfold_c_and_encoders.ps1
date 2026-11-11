# CHAINED: wait for the arm-a k-fold session, then run arm c, then build
# per-fold contrastive encoders until a hard 12:00 stop.
#
# PHASE 1  arm c (kfold_inv04_sampler), folds 0-4, lambda_inv 0.4   ~4.2 h
# PHASE 2  per-fold encoders for models b/d                         ~115 min each
#
# WHY PHASE 2 CANNOT FINISH TODAY, stated up front: the full b/d k-fold path is
# 5 encoders (~9.6 h) PLUS 10 fine-tuning runs (~8.3 h) = ~18 h. Phase 2 is a
# partial down-payment. Encoders are pure, reusable dependencies, so partial
# progress is not wasted - but b/d cannot be reported from this session.
#
# WHY EACH FOLD NEEDS ITS OWN ENCODER - measured, not assumed:
#   fold0.train contains 100.0% of fold1.val and 100.0% of fold3.val.
# With 5 folds the train sides overlap 75%, and every other fold's validation
# set sits entirely inside any given fold's train set. One shared encoder would
# therefore leak every other fold's val completely, and it would be invisible in
# the aggregate. dual_protocol_training.py now refuses an encoder whose sidecar
# fold does not match the run's fold.
#
# ASCII ONLY. PowerShell 5.1 reads a BOM-less file as ANSI, and a UTF-8 em-dash
# decodes to a trailing U+201D which PowerShell accepts as a closing quote -
# that silently broke scripts/run_kfold.ps1 on 2026-09-11.

$ErrorActionPreference = "Continue"
$py      = "C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe"
$root    = "c:\Users\sakso\Downloads\NCKU\Proj\ECG_project"
$logdir  = "$root\runs\_logs"
$NFolds  = 5
$SplitSeed = 0
$Seed    = 42
$StallMin = 25
$RunBudget = 90      # min of runway required to START an arm-c fold
$EncBudget = 140     # min of runway required to START an encoder
$waitLog = "$logdir\kfold_a_session.log"
Set-Location $root
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$Power = Add-Type -MemberDefinition $sig -Name PowerUtilKFoldC -Namespace Win32 -PassThru
function Assert-Awake { [void]$Power::SetThreadExecutionState([uint32]2147483713) }
Assert-Awake

$StopAt = (Get-Date).Date.AddHours(12)
if ((Get-Date) -ge $StopAt) { $StopAt = $StopAt.AddDays(1) }
function Remaining { return ($StopAt - (Get-Date)).TotalMinutes }

Write-Output "=============================================================="
Write-Output " PHASE 1 arm c (5 folds) + PHASE 2 per-fold encoders"
Write-Output " armed   : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " STOP AT : $($StopAt.ToString('yyyy-MM-dd HH:mm'))"
Write-Output " runway  : $([math]::Round((Remaining))) min"
Write-Output "=============================================================="

function Wait-Cool {
    param([int]$Target = 70, [int]$MaxMin = 20)
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

# ---- wait for the arm-a session to finish -----------------------------------
Write-Output ""
Write-Output "waiting for the arm-a k-fold session to end..."
while ($true) {
    Assert-Awake
    if ((Test-Path $waitLog) -and
        (Select-String -Path $waitLog -Pattern "5-FOLD CV ENDED" -Quiet)) {
        Write-Output "arm-a session finished at $(Get-Date -Format 'HH:mm')"
        break
    }
    if ((Get-Date) -ge $StopAt) { Write-Output "12:00 reached while waiting."; exit 0 }
    Start-Sleep -Seconds 120
}

# ---- PHASE 1: arm c ---------------------------------------------------------
Write-Output ""
Write-Output "############ PHASE 1 - arm c, 5 folds ############"
$cDone = 0; $cFail = 0
foreach ($fold in 0..($NFolds - 1)) {
    Assert-Awake
    $rem = Remaining
    Write-Output ""
    Write-Output "=== [c fold $fold] $(Get-Date -Format 'HH:mm')  remaining $([math]::Round($rem)) min"
    if ($rem -lt $RunBudget) {
        Write-Output "    STOP: $([math]::Round($rem)) min < $RunBudget needed. Ending phase 1."
        break
    }
    $tag = "kfold_inv04_sampler_fold$fold"
    $rundir = "$root\runs\${tag}_seed$Seed"
    if (Test-Path "$rundir\metrics.json") { Write-Output "    already complete"; continue }

    Wait-Cool
    $started = Get-Date
    $log = "$logdir\${tag}_seed$Seed.log"
    $r = Invoke-Watched -ArgList @("-u", "code\dual_protocol_training.py",
        "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
        "--split-seed", "$SplitSeed", "--seed", "$Seed",
        "--lambda-inv", "0.4", "--tag", $tag) -Log $log -HardMin 160
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if (Test-Path "$rundir\metrics.json") {
        Write-Output "    finished in $mins min ($r)"; $cDone++
        & $py "code\run_summary.py" --tag $tag --seed $Seed
    } else {
        Write-Output "    FAILED after $mins min ($r). Tail:"
        Get-Content $log -Tail 10 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "      $_" }
        $cFail++
        if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force }
        if ($r -eq "deadline") { break }
    }
}
Write-Output ""
Write-Output "PHASE 1 done: $cDone completed, $cFail failed"

# ---- PHASE 2: per-fold encoders --------------------------------------------
Write-Output ""
Write-Output "############ PHASE 2 - per-fold contrastive encoders ############"
$eDone = 0
foreach ($fold in 0..($NFolds - 1)) {
    Assert-Awake
    $rem = Remaining
    $enc = "model_saved\pretrained_encoder_kfold_fold$fold.pth"
    $meta = "model_saved\pretrained_encoder_kfold_fold$fold.meta.json"
    Write-Output ""
    Write-Output "=== [encoder fold $fold] $(Get-Date -Format 'HH:mm')  remaining $([math]::Round($rem)) min"
    if (Test-Path "$root\$enc") { Write-Output "    already exists"; continue }
    if ($rem -lt $EncBudget) {
        Write-Output "    STOP: $([math]::Round($rem)) min < $EncBudget needed."
        Write-Output "    Remaining encoders are a NEXT-SESSION job. Command:"
        Write-Output "      python code/contrastive_pretrain.py --protocol kfold --fold K \"
        Write-Output "          --n-folds 5 --split-seed 0 --seed 42 --epochs 50 \"
        Write-Output "          --save-path model_saved/pretrained_encoder_kfold_foldK.pth"
        break
    }

    Wait-Cool
    $started = Get-Date
    $log = "$logdir\contrastive_pretrain_kfold_fold$fold.log"
    $r = Invoke-Watched -ArgList @("-u", "code\contrastive_pretrain.py",
        "--protocol", "kfold", "--fold", "$fold", "--n-folds", "$NFolds",
        "--split-seed", "$SplitSeed", "--seed", "$Seed", "--epochs", "50",
        "--save-path", $enc) -Log $log -HardMin 190
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if ((Test-Path "$root\$enc") -and (Test-Path "$root\$meta")) {
        Write-Output "    encoder built in $mins min ($r)"; $eDone++
    } else {
        Write-Output "    ENCODER FAILED after $mins min ($r); removing partials."
        foreach ($f in $enc, $meta, "model_saved\pretrained_encoder_kfold_fold${fold}_loss.png") {
            if (Test-Path "$root\$f") { Remove-Item "$root\$f" -Force }
        }
        Get-Content $log -Tail 10 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "      $_" }
        break
    }
}

Write-Output ""
Write-Output "=============================================================="
Write-Output " SESSION ENDED $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output " arm c folds completed : $cDone   encoders built : $eDone"
Write-Output "=============================================================="
foreach ($tag in "kfold_focal_sampler", "kfold_inv04_sampler") {
    $have = @()
    foreach ($f in 0..($NFolds - 1)) {
        if (Test-Path "$root\runs\${tag}_fold${f}_seed$Seed\metrics.json") { $have += $f }
    }
    Write-Output (" {0,-24} folds: {1}" -f $tag, ($have -join ","))
}
$encs = @()
foreach ($f in 0..($NFolds - 1)) {
    if (Test-Path "$root\model_saved\pretrained_encoder_kfold_fold$f.pth") { $encs += $f }
}
Write-Output (" per-fold encoders built  : {0}" -f ($encs -join ","))
Write-Output " b/d k-fold still needs: $(5 - $encs.Count) encoder(s) + 10 runs (~$([math]::Round((5-$encs.Count)*1.9 + 8.3,1)) h)"
[void]$Power::SetThreadExecutionState([uint32]2147483648)
Write-Output " sleep guard released"

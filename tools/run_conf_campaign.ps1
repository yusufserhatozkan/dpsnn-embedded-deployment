# ICECS conf-branch latency campaign — fully automated on-device measurement.
#
# For each variant: stedgeai generate -> headless CubeIDE build -> start serial
# listener -> flash + reset -> collect the firmware's DWT TIMING line.
# Requires the B-U585I-IOT02A board connected via USB (ST-LINK + VCP).
#
# Usage:  powershell -File tools\run_conf_campaign.ps1 [-Port COM3] [-Variants conf_nobin_n64,...]
param(
    [string]$Port = "COM3",
    [string[]]$Variants = @(
        "conf_baseline_n64",            # control: expect ~2.715 ms/frame
        "conf_nobin_n64",
        "conf_noptwise_n64",
        "conf_plifro_n64",
        "conf_combined_n64",
        "conf_decmatvec_n64",
        "conf_decmatvec_combined_n64",
        "conf_decmatvec_n128"
    )
)

$ErrorActionPreference = "Stop"
# Tool and project locations — edit these for your machine.
$STEDGEAI = "$env:USERPROFILE\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0\Utilities\windows\stedgeai.exe"
$CUBEIDE  = "C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\stm32cubeidec.exe"
$PROGCLI  = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
$PYTHON   = "python"                                  # or full path to your env's python.exe
$REPO     = (Resolve-Path "$PSScriptRoot\..").Path    # repo root (this script lives in tools/)
$FWPROJ   = "$env:USERPROFILE\Desktop\Stm_deployment" # STM32CubeIDE firmware project
$HEADLESS_WS = "$env:USERPROFILE\.cubeide_headless_ws"
$ELF      = "$FWPROJ\Debug\Stm_deployment.elf"
$RESULTS  = "$REPO\results\conf_timing_results.txt"

Set-Location $REPO

# Sanity: board present?
$probe = & $PROGCLI -l 2>&1 | Out-String
if ($probe -match "No ST-Link detected") {
    Write-Error "No ST-LINK detected - connect the B-U585I-IOT02A board first."
}

Add-Content $RESULTS "`n=== Campaign run $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="

foreach ($v in $Variants) {
    Write-Host "`n########## $v ##########" -ForegroundColor Cyan
    $onnx = "export\$v`_xcubeai.onnx"
    if (-not (Test-Path $onnx)) { Write-Error "missing $onnx" }

    # 1. Generate C model under the fixed firmware network name
    Write-Host "[1/4] stedgeai generate ..."
    & $STEDGEAI generate --model $onnx --target stm32u5 --optimization balanced `
        --compression none --name dpsnn_streaming_n64 --workspace st_ai_ws `
        --output "$FWPROJ\X-CUBE-AI\App" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "stedgeai generate failed for $v" }

    # 2. Headless build
    Write-Host "[2/4] CubeIDE headless build ..."
    $before = (Get-Item $ELF).LastWriteTime
    & $CUBEIDE --launcher.suppressErrors -nosplash `
        -application org.eclipse.cdt.managedbuilder.core.headlessbuild `
        -data $HEADLESS_WS -build "Stm_deployment/Debug" 2>&1 |
        Select-String "Build Finished|error" | ForEach-Object { Write-Host "  $_" }
    if ((Get-Item $ELF).LastWriteTime -le $before) { Write-Error "build produced no new ELF for $v" }

    # 3. Start serial listener BEFORE flashing (flash ends with a reset that
    #    starts the run immediately)
    Write-Host "[3/4] starting serial listener on $Port ..."
    $log = "results\conf_timing_$v.log"
    $listener = Start-Process -FilePath $PYTHON `
        -ArgumentList "tools/read_timing.py","--port",$Port,"--label",$v,"--timeout","300" `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru
    Start-Sleep -Seconds 3

    # 4. Flash + reset
    Write-Host "[4/4] flashing ..."
    & $PROGCLI -c port=SWD mode=UR -d $ELF -v -rst | Select-String "File download complete|Error" |
        ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) { Stop-Process -Id $listener.Id -Force -ErrorAction SilentlyContinue; Write-Error "flash failed for $v" }

    $listener.WaitForExit()
    $timing = (Select-String -Path $log -Pattern "TIMING:").Line
    if (-not $timing) { Write-Error "no TIMING line captured for $v (see $log)" }
    Write-Host "  => $timing" -ForegroundColor Green
    Add-Content $RESULTS "$v  $timing"
}

Write-Host "`nAll done. Results:" -ForegroundColor Cyan
Get-Content $RESULTS | Select-Object -Last ($Variants.Count + 1)

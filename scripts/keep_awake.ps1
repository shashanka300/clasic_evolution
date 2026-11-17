# Keep Windows from idle-sleeping while the WSL job queue runs; exits (and releases) when the scheduler is gone.
# Uses SetThreadExecutionState (what "Caffeine"-style tools do): no power settings are changed.
# Run in its own terminal:  powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k = Add-Type -MemberDefinition $sig -Name Power -Namespace KeepAwake -PassThru
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
$null = $k::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Write-Host "$(Get-Date -Format s) keeping the PC awake while scripts/pqueue.py runs in WSL"
while ($true) {
    Start-Sleep -Seconds 300
    $n = (wsl -d Ubuntu-24.04 -- pgrep -fc "python3 scripts/pqueue[.]py") 2>$null
    if (-not $n -or [int]$n -eq 0) { break }
    $null = $k::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
}
$null = $k::SetThreadExecutionState($ES_CONTINUOUS)
Write-Host "$(Get-Date -Format s) queue finished; sleep allowed again"

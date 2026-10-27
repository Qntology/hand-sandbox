# Called by test_runner.bat (test_cuda / test_rocm / test_vulkan / test_cpu) at the end of a run.
#  - Collects the check lines of the step logs into highlights.log: [GPU] [NN] [PRESET] [DYN] and
#    the reference benchmarks [AX] [FX] [CAM] [VX] [CX] [TX] [KX] [MX] [UX] (each with -SUM, -HELD,
#    -WHY or -CELL), plus the source-split count [SRC].
#  - Appends failed tests, panic locations, warning count, GPU parity lines, the N:N composite
#    summary, the scenario-line count of every benchmark area and the cross-area deviation
#    report ([UX] / [UX-SUM]) to summary.log.
param([Parameter(Mandatory = $true)][string]$LogDir, [string]$Backend = '')
$ErrorActionPreference = 'Continue'
$enc = New-Object System.Text.UTF8Encoding($false)
$summary = Join-Path $LogDir 'summary.log'
$highlights = Join-Path $LogDir 'highlights.log'
$tag = '^\[(GPU|PRESET|DYN|SRC|(NN|AX|FX|CAM|VX|CX|TX|KX|MX|UX)(-SUM|-HELD|-WHY|-CELL)?)\]'
$hl = New-Object System.Collections.Generic.List[string]
$failed = New-Object System.Collections.Generic.List[string]
$panics = New-Object System.Collections.Generic.List[string]
$gpu = New-Object System.Collections.Generic.List[string]
$nnsum = New-Object System.Collections.Generic.List[string]
$ux = New-Object System.Collections.Generic.List[string]
$areas = [ordered]@{}
$warnings = 0
$logs = Get-ChildItem -LiteralPath $LogDir -Filter '*.log' | Where-Object { $_.Name -match '^\d\d_' } | Sort-Object Name
foreach ($f in $logs) {
    $lines = [System.IO.File]::ReadAllLines($f.FullName, [System.Text.Encoding]::UTF8)
    $picked = New-Object System.Collections.Generic.List[string]
    foreach ($raw in $lines) {
        if ($raw -match '^test (.+) \.\.\. FAILED') { $failed.Add("$($f.BaseName): $($Matches[1])") }
        if ($raw -match 'panicked at (.+)$') { $panics.Add("$($f.BaseName): $($Matches[1])") }
        if ($raw -match '^warning: ' -and $raw -notmatch 'generated \d+ warning') { $warnings++ }
        # With --nocapture the first line a test prints follows cargo's 'test <name> ... '.
        $l = $raw -replace '^test \S+ \.\.\. ', ''
        if ($l -match $tag) { $picked.Add($l) }
        if ($l -match '^\[GPU\]') { $gpu.Add($l) }
        if ($l -match '^\[(NN-SUM|NN-HELD)\]') { $nnsum.Add($l) }
        if ($l -match '^\[UX(-SUM)?\]') { $ux.Add("$($f.BaseName): $l") }
        if ($l -match '^\[(AX|FX|CAM|VX|CX|TX|KX|MX)-SUM\]') {
            $a = $Matches[1]
            if ($areas.Contains($a)) { $areas[$a]++ } else { $areas[$a] = 1 }
        }
    }
    if ($picked.Count -gt 0) {
        $hl.Add("===== $($f.Name) ($($picked.Count) lines) =====")
        $hl.AddRange($picked)
        $hl.Add('')
    }
}
[System.IO.File]::WriteAllLines($highlights, $hl, $enc)

$s = New-Object System.Collections.Generic.List[string]
$s.Add('')
$s.Add('======== check summary (full lines in highlights.log) ========')
if ($Backend) { $s.Add("backend: $Backend (physics and game logic run on the CPU for every backend; steps 02-06 must match across backends)") }
if ($nnsum.Count -gt 0) { $s.Add('N:N composite scenarios:'); foreach ($n in $nnsum) { $s.Add("  $n") } }
if ($areas.Count -gt 0) { $s.Add('reference benchmarks (scenario lines per area; cells in highlights.log):'); foreach ($k in $areas.Keys) { $s.Add(("  {0}: {1}" -f $k, $areas[$k])) } }
if ($ux.Count -gt 0) { $s.Add('cross-area deviation (median |z| per area; the farthest area is the next work item):'); foreach ($u in $ux) { $s.Add("  $u") } }
if ($gpu.Count -gt 0) { $s.Add(("GPU parity: {0} lines" -f $gpu.Count)); foreach ($g in ($gpu | Select-Object -First 30)) { $s.Add("  $g") } }
$s.Add(("compiler warnings: {0}" -f $warnings))
if ($failed.Count -gt 0) { $s.Add('failed tests:'); foreach ($t in $failed) { $s.Add("  $t") } }
if ($panics.Count -gt 0) { $s.Add('panics:'); foreach ($p in ($panics | Select-Object -First 20)) { $s.Add("  $p") } }
[System.IO.File]::AppendAllLines($summary, $s, $enc)

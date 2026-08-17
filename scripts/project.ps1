[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet("doctor", "bootstrap", "test", "run", "rebuild")]
    [string]$Command = "doctor",

    [ValidateSet("base", "ml", "domain")]
    [string]$Profile = "base",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExpectedPython = (Get-Content -LiteralPath (Join-Path $ProjectRoot ".python-version") -Raw).Trim()
$EnvironmentRoot = Join-Path $ProjectRoot ".venv-profiles"
$EnvironmentPath = Join-Path $EnvironmentRoot $Profile
$PythonPath = Join-Path $EnvironmentPath "Scripts\python.exe"
$MarkerPath = Join-Path $EnvironmentPath ".mlevolve-profile.json"

function Resolve-Uv {
    foreach ($candidate in @("E:\codex-tools\bin\uv.cmd", "uv")) {
        if ($candidate -eq "uv") {
            $command = Get-Command uv -ErrorAction SilentlyContinue
            if ($command) { return $command.Source }
        }
        elseif (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "ENV_BLOCKED: uv was not found. Restore E:\codex-tools\bin\uv.cmd."
}

function Get-RequirementFiles {
    $files = [System.Collections.Generic.List[string]]::new()
    $files.Add((Join-Path $ProjectRoot "requirements_entry.txt"))
    $files.Add((Join-Path $ProjectRoot "requirements_base.txt"))
    if ($Profile -in @("ml", "domain")) {
        $files.Add((Join-Path $ProjectRoot "requirements_ml.txt"))
    }
    if ($Profile -eq "domain") {
        $files.Add((Join-Path $ProjectRoot "requirements_domain.txt"))
    }
    return $files.ToArray()
}

function Get-RequirementsDigest {
    $lines = foreach ($file in (Get-RequirementFiles)) {
        "$(Split-Path $file -Leaf):$((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant())"
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([Convert]::ToHexString($sha.ComputeHash($bytes))).ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Assert-ProfileSupported {
    if ($IsWindows -and $Profile -ne "base") {
        throw "ENV_BLOCKED: profile '$Profile' includes Linux CUDA wheels (nvidia-*, triton). Use WSL/Linux or prepare a separately reviewed Windows/CUDA requirements set; project.ps1 will not silently drop pinned packages."
    }
}

function Invoke-Uv([Parameter(ValueFromRemainingArguments = $true)][string[]]$UvArguments) {
    & $script:UvPath @UvArguments
    if ($LASTEXITCODE -ne 0) { throw "ENV_BLOCKED: uv failed with exit code $LASTEXITCODE." }
}

function Invoke-Bootstrap {
    Assert-ProfileSupported
    New-Item -ItemType Directory -Path $EnvironmentRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        Invoke-Uv venv --python $ExpectedPython $EnvironmentPath
    }
    foreach ($file in (Get-RequirementFiles)) {
        Invoke-Uv pip install --python $PythonPath --no-deps --requirement $file
    }
    $marker = [ordered]@{
        profile = $Profile
        python = $ExpectedPython
        requirements_digest = Get-RequirementsDigest
    } | ConvertTo-Json
    Set-Content -LiteralPath $MarkerPath -Value $marker -Encoding utf8
}

function Invoke-Doctor {
    Assert-ProfileSupported
    $failures = [System.Collections.Generic.List[string]]::new()
    Write-Host "PASS project root: $ProjectRoot"
    Write-Host "PASS profile isolation: $EnvironmentPath"
    Write-Host "PASS uv: $(& $script:UvPath --version)"
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        $failures.Add("profile environment is missing; run bootstrap -Profile $Profile")
    }
    else {
        $actual = (& $PythonPath -c "import platform; print(platform.python_version())").Trim()
        if ($actual -ne $ExpectedPython) { $failures.Add("Python $actual does not match $ExpectedPython; run rebuild -Profile $Profile") }
        else { Write-Host "PASS Python $actual" }
        if (-not (Test-Path -LiteralPath $MarkerPath)) { $failures.Add("profile marker is missing; run bootstrap -Profile $Profile") }
        else {
            $marker = Get-Content -LiteralPath $MarkerPath -Raw | ConvertFrom-Json
            if ($marker.profile -ne $Profile -or $marker.requirements_digest -ne (Get-RequirementsDigest)) {
                $failures.Add("profile inputs changed; run bootstrap -Profile $Profile")
            }
            else { Write-Host "PASS pinned requirement inputs match the profile marker" }
        }
        & $PythonPath -c "import omegaconf, rich, yaml; print('PASS base imports')"
        if ($LASTEXITCODE -ne 0) { $failures.Add("base imports failed; run bootstrap -Profile $Profile") }
    }
    if ($failures.Count) {
        foreach ($failure in $failures) { Write-Error "FAIL $failure" -ErrorAction Continue }
        throw "ENV_BLOCKED: doctor found $($failures.Count) problem(s)."
    }
}

function Remove-ProfileEnvironment {
    if (-not (Test-Path -LiteralPath $EnvironmentPath)) { return }
    $item = Get-Item -LiteralPath $EnvironmentPath -Force
    if ($null -ne $item.LinkType) { throw "REFUSED: profile environment is a reparse point ($($item.LinkType))." }
    $resolved = (Resolve-Path -LiteralPath $EnvironmentPath).Path
    $expected = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv-profiles\$Profile"))
    if (-not [string]::Equals($resolved, $expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "REFUSED: environment resolved outside the expected profile path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$UvPath = Resolve-Uv
switch ($Command) {
    "doctor" { Invoke-Doctor }
    "bootstrap" { Invoke-Bootstrap; Invoke-Doctor }
    "test" {
        Invoke-Bootstrap
        & $PythonPath -m compileall -q (Join-Path $ProjectRoot "config") (Join-Path $ProjectRoot "engine") (Join-Path $ProjectRoot "utils")
        if ($LASTEXITCODE -ne 0) { throw "TEST_FAILED: compileall exited with $LASTEXITCODE." }
        Invoke-Doctor
    }
    "run" {
        if ($Profile -eq "base") { throw "ENV_BLOCKED: run.py imports PyTorch; use -Profile ml or domain in WSL/Linux." }
        Invoke-Bootstrap
        Push-Location $ProjectRoot
        try { & $PythonPath run.py @Arguments; if ($LASTEXITCODE -ne 0) { throw "RUN_FAILED: run.py exited with $LASTEXITCODE." } }
        finally { Pop-Location }
    }
    "rebuild" { Remove-ProfileEnvironment; Invoke-Bootstrap; Invoke-Doctor }
}

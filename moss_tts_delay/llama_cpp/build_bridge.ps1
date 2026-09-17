param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaCppDir
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$llamaRoot = (Resolve-Path -LiteralPath $LlamaCppDir).Path
$buildDir = Join-Path $scriptDir "build-windows"

cmake -S $scriptDir -B $buildDir `
    -DLLAMA_CPP_DIR="$llamaRoot"
cmake --build $buildDir --config Release -j

$bridgeCandidates = @(
    (Join-Path $buildDir "Release\backbone_bridge.dll"),
    (Join-Path $buildDir "backbone_bridge.dll")
)
$bridge = $bridgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $bridge) {
    throw "backbone_bridge.dll was not produced under $buildDir"
}
Copy-Item -LiteralPath $bridge -Destination (Join-Path $scriptDir "backbone_bridge.dll") -Force

$runtimeDirs = @(
    (Join-Path $llamaRoot "build\bin\Release"),
    (Join-Path $llamaRoot "build\bin")
)
$runtimeDir = $runtimeDirs | Where-Object { Test-Path -LiteralPath (Join-Path $_ "llama.dll") } | Select-Object -First 1
if (-not $runtimeDir) {
    throw "Cannot find llama.dll. Rebuild llama.cpp with -DBUILD_SHARED_LIBS=ON."
}
Get-ChildItem -LiteralPath $runtimeDir -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $scriptDir -Force
}

Write-Host "Built: $(Join-Path $scriptDir 'backbone_bridge.dll')"
Write-Host "Copied llama.cpp runtime DLLs from: $runtimeDir"

param(
    [string]$ModelDir = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    [string]$CodecDir = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
    [string]$TtsDevice = "cuda:0",
    [string]$CodecDevice = "cuda:0",
    [string]$Port = "7861"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$env:MODEL_DIR = $ModelDir
$env:CODEC_DIR = $CodecDir
$env:TTS_DEVICE = $TtsDevice
$env:CODEC_DEVICE = $CodecDevice
$env:DEVICE = "cuda"
$env:TTS_DTYPE = "bfloat16"
$env:ATTN_IMPLEMENTATION = "flash_attention_2"
$env:CODEC_WEIGHT_DTYPE = "fp32"
$env:CODEC_COMPUTE_DTYPE = "bf16"
$env:OUTPUT_DIR = Join-Path $RepoRoot "outputs\moss_tts_local_v1_5_streaming"
$env:UPLOAD_DIR = Join-Path $RepoRoot "outputs\moss_tts_local_v1_5_uploads"
$env:HOST = "0.0.0.0"
$env:PORT = $Port

Write-Host "Starting MOSS-TTS Local v1.5 WebUI..." -ForegroundColor Cyan
Write-Host "Model: $env:MODEL_DIR"
Write-Host "Codec: $env:CODEC_DIR"
Write-Host "TTS device: $env:TTS_DEVICE"
Write-Host "Codec device: $env:CODEC_DEVICE"
Write-Host "Open http://localhost:$env:PORT in your browser." -ForegroundColor Green

Push-Location $RepoRoot
try {
    python ".\clis\moss_tts_local_v1.5_app.py"
}
finally {
    Pop-Location
}

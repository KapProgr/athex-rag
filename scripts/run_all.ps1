<#
  Runs the whole project end to end on Windows:
  venv -> install -> pull LLMs -> experiments (writes reports/ and updates README) -> web UI.

  Usage (from the repo root, in PowerShell):
      .\scripts\run_all.ps1                 # everything
      .\scripts\run_all.ps1 -Quick          # one embedder, one LLM, no reranker (much faster)
      .\scripts\run_all.ps1 -SkipLLM        # retrieval experiments only
      .\scripts\run_all.ps1 -ServeOnly      # just start the web UI on the best index
      .\scripts\run_all.ps1 -Quick -Llm groq:llama-3.3-70b-versatile
                                            # use a hosted LLM instead of Ollama
                                            # (needs:  setx LLM_API_KEY "<key>"  once)

  If script execution is blocked:  Set-ExecutionPolicy -Scope Process Bypass
#>
param(
    [switch]$Quick,
    [switch]$SkipLLM,
    [switch]$ServeOnly,
    [string[]]$Llm      # e.g. groq:llama-3.3-70b-versatile ; overrides the Ollama models
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
pip install -q -e ".[embeddings,dev]"

$llms = @("ollama:qwen2.5:7b", "ollama:llama3.1:8b")
$embedders = @("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "BAAI/bge-m3")
$reranker = "BAAI/bge-reranker-v2-m3"
if ($Quick) {
    $llms = @("ollama:qwen2.5:7b")
    $embedders = @("intfloat/multilingual-e5-small")
    $reranker = "none"
}
if ($Llm) { $llms = $Llm }
if (($llms -notmatch "^ollama:") -and -not $env:LLM_API_KEY -and -not $SkipLLM) {
    throw "LLM_API_KEY is not set. Run:  setx LLM_API_KEY `"<your key>`"  and open a new PowerShell."
}

if (-not $ServeOnly) {
    if (-not (Test-Path "eval\questions.jsonl")) { throw "eval\questions.jsonl not found" }
    if (-not (Get-ChildItem "data\raw\*.pdf" -ErrorAction SilentlyContinue)) {
        throw "No PDFs in data\raw (see data\README.md)"
    }
    if (-not $SkipLLM) {
        foreach ($m in ($llms | Where-Object { $_ -like "ollama:*" })) {
            $name = $m -replace "^ollama:", ""
            Write-Host "Pulling $name ..." -ForegroundColor Cyan
            ollama pull $name
        }
    }

    $runArgs = @("experiments", "eval\questions.jsonl", "--update-readme",
              "--embedders") + $embedders + @("--llms") + $llms + @("--reranker", $reranker)
    if ($SkipLLM) { $runArgs += "--skip-llm" }
    Write-Host "Running experiments (this can take a while; progress is printed per run) ..." -ForegroundColor Cyan
    athex-rag @runArgs
}

# Serve the best configuration found by the experiments
$best = Get-Content "reports\experiments.json" -Raw | ConvertFrom-Json
Write-Host "Best configuration: $($best.best)" -ForegroundColor Green
$c = $best.best_config
$env:ATHEX_INDEX_DIR = $c.index_dir
$env:ATHEX_RETRIEVAL = $c.mode
if ($c.reranker) { $env:ATHEX_RERANKER = $c.reranker }
$env:ATHEX_LLM = $llms[0]

Write-Host "Web UI: http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
uvicorn athex_rag.api:app --port 8000

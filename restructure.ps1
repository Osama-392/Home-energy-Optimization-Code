# Restructure script for HEMS project
# Run from project root: powershell -ExecutionPolicy Bypass -File .\restructure.ps1

$cwd = (Get-Location).Path
$dirs = 'src\hems','tests','notebooks','data','models','artifacts'
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# Move .py files from repo root to src\hems
Get-ChildItem -Path $cwd -File -Filter *.py | Where-Object { $_.DirectoryName -eq $cwd } | Move-Item -Destination src\hems -Force

# Move notebooks
Get-ChildItem -Path $cwd -File -Filter *.ipynb | Where-Object { $_.DirectoryName -eq $cwd } | Move-Item -Destination notebooks -Force

# Move CSV and Excel data files from repo root to data
Get-ChildItem -Path $cwd -File | Where-Object { $_.Extension -in '.csv','.xlsx' -and $_.DirectoryName -eq $cwd } | Move-Item -Destination data -Force

# Move model/artifact files to models
Get-ChildItem -Path $cwd -File | Where-Object { $_.Extension -in '.zip','.pkl' -and $_.DirectoryName -eq $cwd } | Move-Item -Destination models -Force

# Move html visualizations to artifacts
Get-ChildItem -Path $cwd -File | Where-Object { $_.Extension -eq '.html' -and $_.DirectoryName -eq $cwd } | Move-Item -Destination artifacts -Force

# Create package __init__.py
$initPath = Join-Path 'src\hems' '__init__.py'
if (-not (Test-Path $initPath)) {
    @" 
__version__ = '0.1.0'
"@ | Out-File -Encoding utf8 $initPath
}

# Create requirements.txt if missing
$reqPath = Join-Path $cwd 'requirements.txt'
if (-not (Test-Path $reqPath)) {
    @"
numpy
pandas
matplotlib
scipy
scikit-learn
stable-baselines3
torch
gym
jupyter
"@ | Out-File -Encoding utf8 $reqPath
}

# Create minimal pyproject.toml if missing
$projectPath = Join-Path $cwd 'pyproject.toml'
if (-not (Test-Path $projectPath)) {
    @"
[project]
name = "hems_project"
version = "0.1.0"
description = "HEMS project package"
"@ | Out-File -Encoding utf8 $projectPath
}

# Create a basic test to ensure package imports as `hems`
$testPath = Join-Path 'tests' 'test_import.py'
if (-not (Test-Path $testPath)) {
    @"
def test_import():
    import importlib
    importlib.import_module('hems')
"@ | Out-File -Encoding utf8 $testPath
}

Write-Host "Restructure completed. Created directories and moved files. Review changes." -ForegroundColor Green
Write-Host "If you want to run tests: python -m pytest -q" -ForegroundColor Yellow

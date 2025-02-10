<#
            _                      
   _____   (_)  _____  _____  ____ 
  / ___/  / /  / ___/ / ___/ / __ \
 / /     / /  / /__  / /__  / /_/ /
/_/     /_/   \___/  \___/  \____/ 
                                   
© r1cco.com

Environment Setup Module

This module is responsible for configuring the necessary Python virtual environment
to run the project scripts. It performs the following operations:

1. Verifies if Python is installed on the system
2. Creates a virtual environment if it doesn't exist
3. Activates the virtual environment in PowerShell
4. Installs the project dependencies via pip

The script is designed to work on Windows through PowerShell and manage
project dependencies via pip.
#>

$pythonCommand = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } 
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { "py -3" }
    else { $null }

if (-not $pythonCommand) {
    Write-Host "Python not found. Install Python first."
    exit 1
}

$venvDir = "venv"
$activateScript = Join-Path (Join-Path $venvDir "Scripts") "Activate.ps1"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..."
    & $pythonCommand -m venv $venvDir
}

if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    . $activateScript
} else {
    Write-Host "Error: Activation script not found in $activateScript"
    exit 1
}

if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..."
    pip install numpy==2.1.3 --only-binary :all:
    
    $requirements = Get-Content requirements.txt | Where-Object { $_ -notmatch "numpy" }
    $requirements | Set-Content requirements.txt.temp
    
    pip install --ignore-installed -r requirements.txt.temp
    
    Remove-Item requirements.txt.temp

    Write-Host "`nInstalling PyTorch..."
    pip install torch torchvision torchaudio
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Attempting to upgrade to CUDA version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    }
} else {
    Write-Host "Warning: requirements.txt not found"
}

function Test-FFmpeg {
    try {
        $null = & ffmpeg -version
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-FFmpeg)) {
    Write-Host "`nFFmpeg not found. Installing..."
    
    $ffmpegDir = Join-Path $PSScriptRoot "ffmpeg"
    $ffmpegZip = Join-Path $PSScriptRoot "ffmpeg.zip"
    $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    if (-not (Test-Path $ffmpegDir)) {
        New-Item -ItemType Directory -Path $ffmpegDir -Force | Out-Null
    }
    
    Write-Host "Downloading FFmpeg..."
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip
    
    Write-Host "Extracting FFmpeg..."
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force
    
    $ffmpegBinDir = Get-ChildItem -Path $ffmpegDir -Recurse -Directory | 
                    Where-Object { $_.Name -eq "bin" } | 
                    Select-Object -First 1 -ExpandProperty FullName
    
    $env:Path = "$ffmpegBinDir;$env:Path"
    
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$ffmpegBinDir*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$userPath;$ffmpegBinDir",
            "User"
        )
    }
    
    Remove-Item $ffmpegZip -Force
    
    Write-Host "FFmpeg installed successfully!"
    
    if (-not (Test-FFmpeg)) {
        Write-Host "Error: FFmpeg installation failed. Please install manually."
        exit 1
    }
}

Write-Host "`nEnvironment ready! To reactivate manually use:"
Write-Host "  .\$venvDir\Scripts\Activate.ps1`n"
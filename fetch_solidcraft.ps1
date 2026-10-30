param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Sha256,
    [Parameter(Mandatory = $true)][string]$Dest,
    [Parameter(Mandatory = $true)][string]$Folder
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$zip = Join-Path ([IO.Path]::GetTempPath()) "$Folder.zip"
$unpack = Join-Path ([IO.Path]::GetTempPath()) "$Folder-unpack"
Write-Host "Downloading $Url"
Invoke-WebRequest -Uri $Url -OutFile $zip
$got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
if ($got -ne $Sha256.ToLower()) {
    Remove-Item $zip -Force
    throw "SHA-256 mismatch: expected $Sha256, got $got"
}
if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $unpack
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
Move-Item -Path (Join-Path $unpack $Folder) -Destination $Dest
Remove-Item $unpack -Recurse -Force
Remove-Item $zip -Force
Write-Host "SolidCraft web package is in $Dest"

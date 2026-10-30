@echo off
rem =====================================================================
rem  Fetches the SolidCraft web package this app embeds: the official
rem  release, pinned by version and SHA-256, unpacked to ui/vendor/solidcraft.
rem  SolidCraft is MIT OR Apache-2.0; its licences and NOTICE come with it.
rem =====================================================================
chcp 65001 > nul
setlocal
set "VERSION=0.27.1"
set "SHA256=3bd142af6fbc5b209738f7ddc69d58d9baf649085dc5bd718b61900f5c572a1f"
set "URL=https://github.com/tangency-labs/solidcraft/releases/download/v%VERSION%/solidcraft-web-%VERSION%.zip"
set "DEST=%~dp0ui\vendor\solidcraft"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch_solidcraft.ps1" -Url "%URL%" -Sha256 "%SHA256%" -Dest "%DEST%" -Folder "solidcraft-web-%VERSION%"
exit /b %errorlevel%

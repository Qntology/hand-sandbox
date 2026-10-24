@echo off
rem check for the cuda backend. Extra arguments go to cargo. See _cargo.bat.
call "%~dp0_cargo.bat" cuda check %*

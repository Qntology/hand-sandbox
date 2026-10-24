@echo off
rem dev for the cuda backend. Extra arguments go to cargo. See _cargo.bat.
call "%~dp0_cargo.bat" cuda dev %*

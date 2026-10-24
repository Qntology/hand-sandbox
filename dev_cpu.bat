@echo off
rem dev for the cpu backend. Extra arguments go to cargo. See _cargo.bat.
call "%~dp0_cargo.bat" cpu dev %*

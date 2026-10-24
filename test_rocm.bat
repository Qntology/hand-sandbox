@echo off
rem test for the rocm backend. Extra arguments go to cargo. See _cargo.bat.
call "%~dp0_cargo.bat" rocm test %*

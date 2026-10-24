@echo off
rem test for the vulkan backend. Extra arguments go to cargo. See _cargo.bat.
call "%~dp0_cargo.bat" vulkan test %*

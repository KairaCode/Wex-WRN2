@echo off
rem nvcc with an MSVC host compiler: VCVARS = path of vcvars64.bat (default: the VS 18 BuildTools location)
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
nvcc %*

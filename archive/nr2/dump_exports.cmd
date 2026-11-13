@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem dump_exports.cmd file.dll : list a DLL's exports with dumpbin
rem Override (environment): VCVARS (vcvars64.bat).
setlocal
if "%~1"=="" (echo usage: dump_exports.cmd file.dll & exit /b 1)
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
dumpbin /nologo /exports "%~1"
exit /b %errorlevel%

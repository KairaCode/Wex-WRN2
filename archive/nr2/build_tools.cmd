@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem wex_nr2_tools.dll: WNR2 GPU counts/encoder + CPU reference wrappers (lab only), with line info and the
rem ptxas report. integration\build\build_wex_windows.cmd builds the same DLL without those two.
rem Overrides (environment): VCVARS (vcvars64.bat), NVCC (default nvcc on PATH), ARCH (default sm_120 = RTX 50xx).
setlocal
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined NVCC set "NVCC=nvcc"
if not defined ARCH set "ARCH=sm_120"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
"%NVCC%" -O3 -std=c++17 -arch=%ARCH% -lineinfo -Xptxas -v -Xcompiler /O2,/EHsc,/W3 -I"%~dp0..\..\c" -shared "%~dp0encode_nr2.cu" -o "%~dp0wex_nr2_tools.dll"
exit /b %errorlevel%

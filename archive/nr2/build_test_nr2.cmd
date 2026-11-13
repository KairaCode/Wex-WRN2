@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem test_nr2_cuda.exe: WNR2 GPU decoder test for kaira_wex_cuda.dll (its own CUDA
rem runtime; the helper is loaded with LoadLibrary, as the engine does).
rem Build the helper first with integration\build\build_wex_windows.cmd; run the test with run_test_nr2.cmd.
rem Overrides (environment): VCVARS (vcvars64.bat), NVCC (default nvcc on PATH), ARCH (default sm_120 = RTX 50xx).
setlocal
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined NVCC set "NVCC=nvcc"
if not defined ARCH set "ARCH=sm_120"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
if not exist "%~dp0build" mkdir "%~dp0build"
"%NVCC%" -O2 -std=c++17 -arch=%ARCH% -Xcompiler /O2,/EHsc "%~dp0test_nr2_cuda.cu" -o "%~dp0build\test_nr2_cuda.exe"
exit /b %errorlevel%

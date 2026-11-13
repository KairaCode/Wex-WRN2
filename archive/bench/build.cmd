@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem Builds one benchmark of this folder: bench\NAME.cu -> bench\build\NAME.exe (compile only; build\ is git-ignored).
rem usage: bench\build.cmd NAME
rem   bench_prio, h2d_bw, bench_nr2_batch, bench_nr2_rechunk, wddm_graph   decode-contention micro-benchmarks
rem   bench_nr2_kernel                                                      WNR2 kernel-only throughput
rem   test_decode_batch                                                     WNR2 GPU batch check (also built by
rem                                                                         integration\build\build_wex_windows.cmd)
rem Overrides (environment): VCVARS (vcvars64.bat), NVCC (default nvcc on PATH), ARCH (default sm_120 = RTX 50xx).
setlocal
if "%~1"=="" (echo usage: bench\build.cmd NAME, to build bench\NAME.cu & exit /b 1)
if not exist "%~dp0%~1.cu" (echo no such source: "%~dp0%~1.cu" & exit /b 1)
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined NVCC set "NVCC=nvcc"
if not defined ARCH set "ARCH=sm_120"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
if not exist "%~dp0build" mkdir "%~dp0build"
"%NVCC%" -O2 -std=c++17 -arch=%ARCH% -Xcompiler /O2,/EHsc -I"%~dp0..\..\c" "%~dp0%~1.cu" -o "%~dp0build\%~1.exe"
exit /b %errorlevel%

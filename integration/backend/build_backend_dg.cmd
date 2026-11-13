@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem coli_cuda_dsv4_dg.dll (the DeepGEMM backend with the WEX direct-upload exports) from an engine tree.
rem The backend source is engine code and is not in this repo, so the two source locations are required:
rem   COLIBRI_C      the engine's c\ folder, the one that holds backend_cuda_dsv4.cu and dsv4.def
rem   DEEPGEMM_ROOT  a DeepGEMM checkout (deep_gemm\include, cutlass\include, third-party\cutlass\include)
rem   CUDA_PATH      the CUDA toolkit root (the CUDA installer sets it); KAIRA builds with v13.4
rem usage: integration\backend\build_backend_dg.cmd [OUT_DIR]      (default: [repo]\out, git-ignored)
rem Override (environment): VCVARS (vcvars64.bat; found with vswhere when unset). The -gencode target is sm_120a (RTX 50xx), as in KAIRA.
setlocal
if not defined COLIBRI_C (echo set COLIBRI_C to the engine c\ folder that holds backend_cuda_dsv4.cu and dsv4.def & exit /b 1)
if not exist "%COLIBRI_C%\backend_cuda_dsv4.cu" (echo not found: "%COLIBRI_C%\backend_cuda_dsv4.cu" & exit /b 1)
if not defined DEEPGEMM_ROOT (echo set DEEPGEMM_ROOT to your DeepGEMM checkout & exit /b 1)
if not exist "%DEEPGEMM_ROOT%\deep_gemm\include" (echo not found: "%DEEPGEMM_ROOT%\deep_gemm\include" & exit /b 1)
if not defined CUDA_PATH (echo set CUDA_PATH to the CUDA toolkit root & exit /b 1)
if "%~1"=="" (set "OUT=%~dp0..\..\out") else (set "OUT=%~1")
rem VCVARS unset: the newest Visual Studio with the C++ x64 tools that vswhere finds, else the VS 18 BuildTools path
if not defined VCVARS for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
if not exist "%OUT%" mkdir "%OUT%"
for %%O in ("%OUT%") do set "OUT=%%~fO"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
cd /d "%COLIBRI_C%"
nvcc -O3 -std=c++20 -ftz=false --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=/Zc:preprocessor -gencode arch=compute_120a,code=sm_120a -DCOLI_DSV4_DEEPGEMM -I"%DEEPGEMM_ROOT%\deep_gemm\include" -I"%DEEPGEMM_ROOT%\cutlass\include" -I"%DEEPGEMM_ROOT%\third-party\cutlass\include" -shared -Xlinker /DEF:dsv4.def -L"%CUDA_PATH%\lib\x64" -lcublasLt -lcudart -lcuda backend_cuda_dsv4.cu -o "%OUT%\coli_cuda_dsv4_dg.dll"
exit /b %ERRORLEVEL%

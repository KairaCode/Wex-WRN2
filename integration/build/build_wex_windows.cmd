@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem Builds the WEX/WNR2 Windows binaries from this repo alone (no engine or colibri tree needed):
rem   OUT\kaira_wex_cuda.dll     engine helper (c\tools\wex_cuda_bridge.cu, ABI 1, features 1|2|4|8|16|32);
rem                              deploy it next to deepseek_v4.exe or point COLI_WEX_CUDA_LIB at it
rem   archive\nr2\wex_nr2_tools.dll  WNR2 trainer/encoder/verifier used by archive\nr2\*.py (nr2lib.py loads it from there)
rem   OUT\test_decode_batch.exe  WNR2 GPU batch check (archive\bench\test_decode_batch.cu)
rem   OUT\check_container.exe    independent WNR2-vs-WEX1 checker (archive\nr2\check_container.c, MSVC C11)
rem usage: integration\build\build_wex_windows.cmd [OUT_DIR]      (default: <repo>\out, git-ignored)
rem Overrides (environment): VCVARS (vcvars64.bat; found with vswhere when unset), NVCC (default nvcc on PATH),
rem   ARCH (default sm_120 = RTX 50xx; sm_89 RTX 40xx, sm_86 RTX 30xx; below sm_80 a plain load replaces cp.async, untested on such a GPU).
setlocal
set "ROOT=%~dp0..\.."
if "%~1"=="" (set "OUT=%ROOT%\out") else (set "OUT=%~1")
rem VCVARS unset: the newest Visual Studio with the C++ x64 tools that vswhere finds, else the VS 18 BuildTools path
if not defined VCVARS for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined NVCC set "NVCC=nvcc"
if not defined ARCH set "ARCH=sm_120"
call "%VCVARS%" >nul
rem no %VCVARS% inside the parentheses: its "(x86)" would close the block
if errorlevel 1 (echo vcvars64.bat failed: set VCVARS to your vcvars64.bat & exit /b 1)
if not exist "%OUT%" mkdir "%OUT%"
rem c\ has the colibri c\ layout, so wex_cuda_bridge.cu finds its "../wex_cuda.cuh" where it is
"%NVCC%" -O3 -std=c++17 -arch=%ARCH% -Xcompiler /O2,/EHsc,/arch:AVX2 -shared "%ROOT%\c\tools\wex_cuda_bridge.cu" -o "%OUT%\kaira_wex_cuda.dll"
if errorlevel 1 exit /b 1
"%NVCC%" -O3 -std=c++17 -arch=%ARCH% -Xcompiler /O2,/EHsc,/W3 -I"%ROOT%\c" -shared "%ROOT%\archive\nr2\encode_nr2.cu" -o "%ROOT%\archive\nr2\wex_nr2_tools.dll"
if errorlevel 1 exit /b 1
"%NVCC%" -O2 -std=c++17 -arch=%ARCH% -Xcompiler /O2,/EHsc -I"%ROOT%\c" "%ROOT%\archive\bench\test_decode_batch.cu" -o "%OUT%\test_decode_batch.exe"
if errorlevel 1 exit /b 1
cl /nologo /O2 /std:c11 /I"%ROOT%\c" "%ROOT%\archive\nr2\check_container.c" /Fo"%OUT%\\" /Fe"%OUT%\check_container.exe" >nul
if errorlevel 1 (echo check_container.c did not build with MSVC; build it with MSYS2 gcc: gcc -O2 -std=c11 -I c archive\nr2\check_container.c -o check_container.exe & exit /b 1)
echo built: "%OUT%\kaira_wex_cuda.dll" "%ROOT%\archive\nr2\wex_nr2_tools.dll" "%OUT%\test_decode_batch.exe" "%OUT%\check_container.exe"
exit /b 0

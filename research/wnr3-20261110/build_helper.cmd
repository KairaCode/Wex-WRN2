@echo off
rem WNR3: kaira_wex_cuda.dll + test_decode_batch.exe + test_wex_nr3_gpu.exe from a FRESH copy of the
rem worktree src (arg 1 = that copy's src\colibri\c). Needs vswhere on PATH (Visual Studio\Installer).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set "SRC=%~1"
set "OUT=%KAIRA%\lab\wnr3-impl-20261110\build"
if not "%~2"=="" set "OUT=%~2"
set "LAB=%KAIRA%\lab\wex-prod-20261105"
"%NVCC%" -O3 -std=c++17 -arch=sm_120 -Xptxas -v -Xcompiler /O2,/EHsc,/arch:AVX2 -shared "%SRC%\tools\wex_cuda_bridge.cu" -o "%OUT%\kaira_wex_cuda.dll" > "%OUT%\ptxas_helper.log" 2>&1
if errorlevel 1 exit /b 1
"%NVCC%" -O2 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -I"%SRC%" "%LAB%\test_decode_batch.cu" -o "%OUT%\test_decode_batch.exe"
if errorlevel 1 exit /b 1
"%NVCC%" -O2 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc "%SRC%\tests\test_wex_nr3_gpu.cu" -o "%OUT%\test_wex_nr3_gpu.exe"
exit /b %errorlevel%

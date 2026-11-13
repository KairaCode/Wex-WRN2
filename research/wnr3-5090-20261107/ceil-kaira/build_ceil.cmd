@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set "SRC=<KAIRA>\src\colibri\c"
set "HERE=<KAIRA>\lab\wnr3-5090-20261107\ceil-kaira"
"%NVCC%" -O3 -std=c++17 -arch=sm_120 -Xptxas -v -Xcompiler /O2,/EHsc,/arch:AVX2 -I"%SRC%" -I"%HERE%" -shared "%HERE%\wex_cuda_bridge_ceil.cu" -o "%HERE%\kaira_wex_cuda_ceil.dll"
exit /b %errorlevel%

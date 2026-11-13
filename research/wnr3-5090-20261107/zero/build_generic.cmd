@echo off
rem zt_generic.dll: zt_harness.cu (#includes the worktree's backend_cuda_dsv4.cu verbatim), generic flavour:
rem shipped coli_cuda_dsv4.dll flags (Makefile cuda-dsv4-dll, NVCCFLAGS -O3 -std=c++17 -ftz=false) for sm_120, cuBLASLt TC path compiled in.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "SRC=<KAIRA-WT>\src\colibri\c"
set "HERE=%~dp0"
"%CUDA%\bin\nvcc.exe" -O3 -std=c++17 -ftz=false -gencode arch=compute_120,code=sm_120 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=/Zc:preprocessor -I"%SRC%" -shared -L"%CUDA%\lib\x64" -lcublasLt -lcudart -lcuda "%HERE%src\zt_harness.cu" -o "%HERE%build\zt_generic.dll"
exit /b %ERRORLEVEL%

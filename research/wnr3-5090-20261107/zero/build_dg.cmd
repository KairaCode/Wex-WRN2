@echo off
rem zt_dg.dll: zt_harness.cu (#includes the worktree's backend_cuda_dsv4.cu verbatim), DeepGEMM flavour:
rem shipped coli_cuda_dsv4_dg.dll flags (Makefile cuda-dsv4-dg-dll) with the pinned DeepGEMM checkout (39fb444 + MSVC patch).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "SRC=<KAIRA-WT>\src\colibri\c"
set "DG=<KAIRA>\src\colibri\c\third_party\deepgemm"
set "HERE=%~dp0"
"%CUDA%\bin\nvcc.exe" -O3 -std=c++20 -ftz=false --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=/Zc:preprocessor -gencode arch=compute_120a,code=sm_120a -DCOLI_DSV4_DEEPGEMM -I"%DG%\deep_gemm\include" -I"%DG%\cutlass\include" -I"%DG%\third-party\cutlass\include" -I"%SRC%" -shared -L"%CUDA%\lib\x64" -lcublasLt -lcudart -lcuda "%HERE%src\zt_harness.cu" -o "%HERE%build\zt_dg.dll"
exit /b %ERRORLEVEL%

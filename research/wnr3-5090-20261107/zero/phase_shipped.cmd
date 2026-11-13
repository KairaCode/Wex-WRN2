@echo off
rem the shipped app DLLs (read-only load from app\colibri-optimized): dsv4_cuda_expert_group + dsv4_cuda_moe via their own exports
setlocal
set "DSV4_CUDA_PIN_HOST=0"
set "PY=<KAIRA>\lab\wnr3-5090-20261107\venv\Scripts\python.exe"
set "Z=<KAIRA>\lab\wnr3-5090-20261107\zero"
set "APP=<KAIRA>\app\colibri-optimized"
"%PY%" -I "%Z%\src\zt_run.py" shipped "%APP%\coli_cuda_dsv4.dll" "%Z%\out\shipped_generic_default.json"
if errorlevel 1 exit /b 1
set DSV4_CUDA_TC=1
"%PY%" -I "%Z%\src\zt_run.py" shipped "%APP%\coli_cuda_dsv4.dll" "%Z%\out\shipped_generic_tc.json"
if errorlevel 1 exit /b 1
set DSV4_CUDA_TC=
set DSV4_CUDA_BATCHED=0
set DSV4_CUDA_MOE_FUSED6=0
"%PY%" -I "%Z%\src\zt_run.py" shipped "%APP%\coli_cuda_dsv4.dll" "%Z%\out\shipped_generic_unbatched_nofused.json"
if errorlevel 1 exit /b 1
set DSV4_CUDA_BATCHED=
set DSV4_CUDA_MOE_FUSED6=
"%PY%" -I "%Z%\src\zt_run.py" shipped "%APP%\coli_cuda_dsv4_dg.dll" "%Z%\out\shipped_dg_default.json"
exit /b %ERRORLEVEL%

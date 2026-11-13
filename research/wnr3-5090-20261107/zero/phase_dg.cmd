@echo off
rem DeepGEMM flavour: mma micro-test, DeepGEMM routed-expert paths (smoke then 40 layers), generic kernels in the sm_120a build
setlocal
set "DSV4_CUDA_PIN_HOST=0"
set "PY=<KAIRA>\lab\wnr3-5090-20261107\venv\Scripts\python.exe"
set "Z=<KAIRA>\lab\wnr3-5090-20261107\zero"
"%PY%" -I "%Z%\src\zt_run.py" mma "%Z%\build\zt_dg.dll" "%Z%\out\dg_mma.json"
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" dg "%Z%\build\zt_dg.dll" "%Z%\out\smoke_dg_moe.json" --layers 1
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" dg "%Z%\build\zt_dg.dll" "%Z%\out\dg_moe.json"
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" kernels "%Z%\build\zt_dg.dll" "%Z%\out\dg_kernels.json" --per-layer 1
exit /b %ERRORLEVEL%

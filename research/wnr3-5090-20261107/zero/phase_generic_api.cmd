@echo off
rem public API end-to-end on the harness build (worktree source): default, DSV4_CUDA_TC=1, DSV4_CUDA_BATCHED=0
setlocal
set "DSV4_CUDA_PIN_HOST=0"
set "PY=<KAIRA>\lab\wnr3-5090-20261107\venv\Scripts\python.exe"
set "Z=<KAIRA>\lab\wnr3-5090-20261107\zero"
"%PY%" -I "%Z%\src\zt_run.py" api "%Z%\build\zt_generic.dll" "%Z%\out\generic_api_default.json"
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" api "%Z%\build\zt_generic.dll" "%Z%\out\generic_api_tc.json" --env DSV4_CUDA_TC=1
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" api "%Z%\build\zt_generic.dll" "%Z%\out\generic_api_unbatched.json" --env DSV4_CUDA_BATCHED=0
exit /b %ERRORLEVEL%

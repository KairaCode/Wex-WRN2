@echo off
rem DeepGEMM routed paths again with controlled bank-slot spacing: masked dg_moe<256> showed run-to-run (A vs A) differences
rem in layers whose routed slots lie within 3 of each other; spaced = 32 apart, adjacent = consecutive slots.
setlocal
set "DSV4_CUDA_PIN_HOST=0"
set "PY=<KAIRA>\lab\wnr3-5090-20261107\venv\Scripts\python.exe"
set "Z=<KAIRA>\lab\wnr3-5090-20261107\zero"
"%PY%" -I "%Z%\src\zt_run.py" dg "%Z%\build\zt_dg.dll" "%Z%\out\dg_moe_spaced.json" --slots spaced
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" dg "%Z%\build\zt_dg.dll" "%Z%\out\dg_moe_adjacent.json" --slots adjacent
exit /b %ERRORLEVEL%

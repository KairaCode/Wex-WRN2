@echo off
rem smoke (1 expert) then every FP4 kernel at V4.1 shapes on 160 experts (4/layer) + synthetic zero rows
setlocal
set "PY=<KAIRA>\lab\wnr3-5090-20261107\venv\Scripts\python.exe"
set "Z=<KAIRA>\lab\wnr3-5090-20261107\zero"
"%PY%" -I "%Z%\src\zt_run.py" kernels "%Z%\build\zt_generic.dll" "%Z%\out\smoke_generic_kernels.json" --layers 1 --per-layer 1
if errorlevel 1 exit /b 1
"%PY%" -I "%Z%\src\zt_run.py" kernels "%Z%\build\zt_generic.dll" "%Z%\out\generic_kernels.json"
exit /b %ERRORLEVEL%

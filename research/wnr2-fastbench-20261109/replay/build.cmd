@echo off
rem replay.exe (host code only: Win32 overlapped I/O + cudart; the helper DLL is loaded at run time)
rem WEX headers from the integrated build W3 uses (ws-ab).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
if not exist "%~dp0build" mkdir "%~dp0build"
"%NVCC%" -O2 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc,/W3 ^
  -I"%~dp0..\..\v41-port-20261106\ws-ab\c" "%~dp0replay.cu" -o "%~dp0build\replay.exe"
exit /b %errorlevel%

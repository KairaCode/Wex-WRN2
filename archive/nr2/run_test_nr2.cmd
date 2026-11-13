@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Ludovico Zanichelli
rem Runs the WNR2 GPU test (build_test_nr2.cmd) against a helper DLL on the pilot containers. Needs a GPU.
rem Environment:
rem   WEX_PILOT_DIR  required: the folder with pilot-v1.wex and pilot-v3-neural512.wex (written by nr2\build_nr2.py);
rem                  the real WNR2 containers pilot-v3-nr2.wex and pilot-v3-nr2z.wex are checked too when present
rem   WEX_HELPER     default [repo]\out\kaira_wex_cuda.dll (integration\build\build_wex_windows.cmd); a pre-NR2
rem                  helper kept next to it as kaira_wex_cuda.pre-nr2.dll is the WNR1 timing baseline
setlocal
if not defined WEX_PILOT_DIR (echo set WEX_PILOT_DIR to the folder with pilot-v1.wex and pilot-v3-neural512.wex & exit /b 1)
if not defined WEX_HELPER set "WEX_HELPER=%~dp0..\..\out\kaira_wex_cuda.dll"
set "DATA=%WEX_PILOT_DIR%"
if not exist "%DATA%\pilot-v1.wex" (echo not found: "%DATA%\pilot-v1.wex" & exit /b 1)
if not exist "%WEX_HELPER%" (echo not found: "%WEX_HELPER%", build it with integration\build\build_wex_windows.cmd or set WEX_HELPER & exit /b 1)
if not exist "%~dp0build\test_nr2_cuda.exe" (echo not found: "%~dp0build\test_nr2_cuda.exe", run build_test_nr2.cmd first & exit /b 1)
set "EXTRA="
for %%H in ("%WEX_HELPER%") do set "BASE=%%~dpHkaira_wex_cuda.pre-nr2.dll"
if exist "%BASE%" set EXTRA=%EXTRA% --baseline "%BASE%"
if exist "%DATA%\pilot-v3-nr2.wex" set EXTRA=%EXTRA% --wnr2 "%DATA%\pilot-v3-nr2.wex"
if exist "%DATA%\pilot-v3-nr2z.wex" set EXTRA=%EXTRA% --wnr2 "%DATA%\pilot-v3-nr2z.wex"
"%~dp0build\test_nr2_cuda.exe" "%WEX_HELPER%" "%DATA%\pilot-v1.wex" "%DATA%\pilot-v3-neural512.wex" %EXTRA% --json "%~dp0build\test_nr2_cuda.json"
exit /b %errorlevel%

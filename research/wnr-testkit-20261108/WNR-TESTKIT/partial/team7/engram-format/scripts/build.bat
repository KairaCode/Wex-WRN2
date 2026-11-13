@echo off
rem team7 / engram-format: build erow.dll (kit version: no paths of the original PC).
rem Run from a "x64 Native Tools Command Prompt for VS" (cl/link on PATH). Output goes to ..\work\ .
set HERE=%~dp0
set WORK=%HERE%..\work
if not exist "%WORK%" mkdir "%WORK%"
set INCLUDE=%INCLUDE%;%WORK%\stubinc
cd /d "%WORK%"
cl /nologo /O2 /GS- /Oi- /c "%HERE%erow.c" /Fo:erow.obj
link /nologo /DLL /NOENTRY /NODEFAULTLIB /OUT:erow.dll erow.obj

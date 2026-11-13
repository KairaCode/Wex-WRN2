@echo off
set "PATH=%SystemRoot%\System32;%SystemRoot%;C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
set "VCT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
set "PATH=%VCT%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VCT%\include"
set "LIB="
cd /d "<RESEARCH-PC>\team2\zero-safety\build"
cl /nologo /c /O2 /arch:AVX2 /fp:precise /GS- /Gs1000000 /TC /W3 /I "<RESEARCH-PC>\team2\zero-safety\scripts\stubinc" "<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c" /Fo:zs_kern.obj
if errorlevel 1 exit /b 1
link /nologo /DLL /NOENTRY /NODEFAULTLIB /OUT:zs_kern.dll zs_kern.obj

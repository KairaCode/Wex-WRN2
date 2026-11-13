@echo off
rem audit_gpu.exe from the fresh-fix3 workspace headers (= worktree 8ef9645)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -I"%KAIRA%\lab\wnr3-impl-20261110\fresh-fix3\src\colibri\c" "%~dp0audit_gpu.cu" -o "%~dp0audit_gpu.exe"
exit /b %errorlevel%

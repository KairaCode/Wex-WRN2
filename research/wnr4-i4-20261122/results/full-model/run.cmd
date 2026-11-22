@echo off
cd /d %~dp0
set D=%~dp0wnr4_i4_tools.dll
echo START %DATE% %TIME%
if not exist q38_i4.wnm4 python -I wnr4i4.py train q38_int4g64.wex q38_i4.wnm4 --dll %D% --every 8 || goto fail
echo TRAINED %DATE% %TIME%
if not exist q38_int4g64_wnr4i4.wex python -I wnr4i4.py pack q38_int4g64.wex q38_int4g64_wnr4i4.wex --model q38_i4.wnm4 --dll %D% || goto fail
echo PACKED %DATE% %TIME%
python -I wnr4i4.py verify q38_int4g64_wnr4i4.wex --source q38_int4g64.wex --dll %D% || goto fail
echo VERIFIED %DATE% %TIME%
dir q38_int4g64.wex q38_int4g64_wnr4i4.wex
echo DONE
exit /b 0
:fail
echo FAILED %DATE% %TIME%
exit /b 1

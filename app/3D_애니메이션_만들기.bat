@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY ( if exist "%USERPROFILE%\myenv\Scripts\python.exe" set "PY=%USERPROFILE%\myenv\Scripts\python.exe" )
if not defined PY ( echo [오류] 파이썬을 찾지 못했습니다. & pause & exit /b 1 )
echo 앱과 같은 3D 애니메이션 HTML을 만듭니다. (몇 분 걸립니다)
%PY% runs\export_3d_html.py runs\cmp\B44_o2z_3d.pkl.gz runs\fig\B44_o2z_3d
%PY% runs\export_3d_html.py runs\cmp\B44_o2z_refill_3d.pkl.gz runs\fig\B44_o2z_refill_3d
echo.
echo 완료: runs\fig\B44_o2z_3d 폴더의 .html 파일을 더블클릭하면 브라우저에서 열립니다.
echo 오류가 보이면 화면 내용을 그대로 알려주세요.
pause

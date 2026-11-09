@echo off
setlocal
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY ( if exist "%USERPROFILE%\myenv\Scripts\python.exe" set "PY=%USERPROFILE%\myenv\Scripts\python.exe" )
if not defined PY ( echo Python not found. & pause & exit /b 1 )
echo Making 3D HTML files. This takes a few minutes.
%PY% runs\export_3d_html.py runs\cmp\B44_o2z_3d.pkl.gz runs\fig\B44_o2z_3d
%PY% runs\export_3d_html.py runs\cmp\B44_o2z_refill_3d.pkl.gz runs\fig\B44_o2z_refill_3d
echo.
echo Opening the results in your browser.
for %%f in (runs\fig\B44_o2z_3d\*.html) do start "" "%%~ff"
start "" explorer "%CD%\runs\fig\B44_o2z_3d"
echo If an error appeared above, send me the text of this window.
pause

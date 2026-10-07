@echo off
rem ===========================================================================
rem  Zebrafish Free Swim Analyzer - one-time setup for the lab laptop.
rem
rem  Double-click this file. It builds one conda environment holding both
rem  idtracker.ai and the analyzer, checks it, and puts a shortcut on the
rem  desktop. Safe to run again: it reuses the environment if it exists.
rem
rem  Needs: Miniconda or Anaconda, an NVIDIA GPU with a current driver, and
rem  an internet connection (about 5 GB is downloaded).
rem
rem  For testing: set FREESWIM_ENV to use another environment name,
rem  FREESWIM_NO_SHORTCUT=1 to skip the shortcut, FREESWIM_NO_PAUSE=1 to
rem  skip the final pause.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
if not defined FREESWIM_ENV set "FREESWIM_ENV=freeswim"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

echo.
echo  Zebrafish Free Swim Analyzer - setup
echo  ------------------------------------
echo.

rem --- 1. Find conda --------------------------------------------------------
set "CONDA="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA set "CONDA=%%I"
for %%P in ("%ProgramData%\miniconda3" "%UserProfile%\miniconda3" "%LocalAppData%\miniconda3" "%ProgramData%\anaconda3" "%UserProfile%\anaconda3") do (
    if not defined CONDA if exist "%%~P\condabin\conda.bat" set "CONDA=%%~P\condabin\conda.bat"
)
if not defined CONDA (
    echo  [STOP] Conda was not found.
    echo         Install Miniconda from https://www.anaconda.com/download/success
    echo         then run this file again.
    goto :fail
)
echo  [1/6] Found conda: %CONDA%

rem --- 2. Check for an NVIDIA GPU -------------------------------------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [STOP] No NVIDIA driver was found ^(nvidia-smi is missing^).
    echo         Tracking needs an NVIDIA GPU. Install the current driver from
    echo         https://www.nvidia.com/drivers then run this file again.
    goto :fail
)
echo  [2/6] Found an NVIDIA driver.

rem --- 3. Create or reuse the environment -----------------------------------
call "%CONDA%" env list | findstr /R /C:"^%FREESWIM_ENV% " >nul
if errorlevel 1 (
    echo  [3/6] Creating the "%FREESWIM_ENV%" environment...
    call "%CONDA%" create -n %FREESWIM_ENV% python=3.12 pip -y -q
    if errorlevel 1 goto :fail
) else (
    echo  [3/6] Reusing the existing "%FREESWIM_ENV%" environment.
)
set "PY="
for /f "delims=" %%I in ('call "%CONDA%" run -n %FREESWIM_ENV% python -c "import sys; print(sys.executable)"') do set "PY=%%I"
if not defined PY (
    echo  [STOP] Could not find Python inside the "%FREESWIM_ENV%" environment.
    goto :fail
)

rem --- 4. GPU PyTorch -------------------------------------------------------
echo  [4/6] Installing PyTorch for the GPU ^(the large download^)...
"%PY%" -m pip install -q --no-warn-script-location torch torchvision --index-url %TORCH_INDEX% -c constraints-win-cu128.txt
if errorlevel 1 goto :fail

rem --- 5. idtracker.ai and the analyzer -------------------------------------
echo  [5/6] Installing idtracker.ai and the analyzer...
"%PY%" -m pip install -q --no-warn-script-location -e ".[tracking]" -c constraints-win-cu128.txt
if errorlevel 1 goto :fail

rem --- 6. Check, then shortcut ----------------------------------------------
echo  [6/6] Checking the installation...
echo.
"%PY%" -m fish_analyzer --check
if errorlevel 1 goto :fail
echo.

if "%FREESWIM_NO_SHORTCUT%"=="1" goto :done
for %%I in ("%PY%") do set "PYW=%%~dpIpythonw.exe"
set "LAUNCHER=%~dp0launch.pyw"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Free Swim Analyzer.lnk'));" ^
  "$s.TargetPath = $env:PYW; $s.Arguments = [char]34 + $env:LAUNCHER + [char]34;" ^
  "$s.WorkingDirectory = [Environment]::GetFolderPath('MyDocuments');" ^
  "$s.Description = 'Zebrafish Free Swim Analyzer'; $s.Save()"
if errorlevel 1 (
    echo  [WARN] The desktop shortcut could not be created. The install itself is fine.
) else (
    echo  A "Free Swim Analyzer" shortcut is on the desktop.
)

:done
echo.
echo  Setup finished.
if not "%FREESWIM_NO_PAUSE%"=="1" pause
exit /b 0

:fail
echo.
echo  Setup did NOT finish. Read the message above, fix it, and run this again.
if not "%FREESWIM_NO_PAUSE%"=="1" pause
exit /b 1

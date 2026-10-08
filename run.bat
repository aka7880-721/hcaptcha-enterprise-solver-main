@echo off
title hCaptcha Pose Classifier
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

:menu
cls
call :ensure_python
if errorlevel 1 pause & exit /b 1
call :resolve_python

echo.
echo  ============================================
echo   hCaptcha Pose Classifier
echo  ============================================
if exist "%ROOT%\.venv\Scripts\python.exe" (echo   Venv:  ready) else (echo   Venv:  run option 1 first)
if exist "%ROOT%\models\pose_detection_model.h5" (echo   Model: ready) else (echo   Model: not trained)
echo.
echo   1  Setup     Install dependencies ^(first time^)
echo   2  Test      Check Python / GPU / folders
echo   3  Run all   Preprocess - Train - Validate
echo.
echo   4  Preprocess only
echo   5  Train only
echo   6  Validate only
echo.
echo   7  Predict one image
echo   8  Predict tile folder
echo.
echo   0  Exit
echo.
set "C="
set /p C=Choose [0-8]: 
if "%C%"=="1" call :do_setup & goto :menu
if "%C%"=="2" call :do_test & goto :menu
if "%C%"=="3" call :do_pipeline & goto :menu
if "%C%"=="4" call :do_convert & goto :menu
if "%C%"=="5" call :do_train & goto :menu
if "%C%"=="6" call :do_validate & goto :menu
if "%C%"=="7" call :do_predict_image & goto :menu
if "%C%"=="8" call :do_predict_tiles & goto :menu
if "%C%"=="0" exit /b 0
echo Invalid choice.
timeout /t 2 >nul
goto :menu

:ensure_python
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found. Install Python 3.10+ with "Add to PATH".
    exit /b 1
)
exit /b 0

:resolve_python
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PY=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
exit /b 0

:do_setup
echo.
echo === Setup ===
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo Creating .venv ...
    python -m venv "%ROOT%\.venv"
    if errorlevel 1 goto :err
)
set "PY=%ROOT%\.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :err
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :err
echo Setup complete.
pause
exit /b 0

:do_test
call :resolve_python
echo.
echo === Test ===
"%PY%" "%ROOT%\test.py"
pause
exit /b 0

:do_convert
call :resolve_python
echo.
echo === Preprocess ===
"%PY%" -m src.convert
if errorlevel 1 goto :err
pause
exit /b 0

:do_train
call :resolve_python
echo.
echo === Train ===
set "EPOCHS="
set /p EPOCHS=Epochs [50, use 150 for best accuracy]: 
if "!EPOCHS!"=="" set "EPOCHS=50"
if exist "%ROOT%\processed\training" (
    "%PY%" "%ROOT%\train.py" --epochs !EPOCHS!
) else (
    echo processed/ missing - using raw training/ ^(run Preprocess first next time^)
    "%PY%" "%ROOT%\train.py" --epochs !EPOCHS! --raw
)
if errorlevel 1 goto :err
echo Saved: models\pose_detection_model.h5
pause
exit /b 0

:do_validate
call :resolve_python
call :need_model
if errorlevel 1 pause & exit /b 1
echo.
echo === Validate ===
"%PY%" "%ROOT%\validate.py"
if errorlevel 1 goto :err
pause
exit /b 0

:do_pipeline
call :resolve_python
if not exist "%ROOT%\.venv\Scripts\python.exe" call :do_setup
call :resolve_python
echo.
echo === Full pipeline ===
echo.
echo [1/3] Preprocess...
"%PY%" -m src.convert
if errorlevel 1 goto :err
set "EPOCHS="
set /p EPOCHS=Training epochs [50]: 
if "!EPOCHS!"=="" set "EPOCHS=50"
echo.
echo [2/3] Train...
"%PY%" "%ROOT%\train.py" --epochs !EPOCHS!
if errorlevel 1 goto :err
echo.
echo [3/3] Validate...
"%PY%" "%ROOT%\validate.py"
if errorlevel 1 goto :err
echo.
echo Done.
pause
exit /b 0

:do_predict_image
call :resolve_python
call :need_model
if errorlevel 1 pause & exit /b 1
echo.
echo === Predict image ===
echo Drag a file here or paste path:
set "IMG="
set /p IMG=Path: 
if "!IMG!"=="" pause & exit /b 1
set "IMG=!IMG:"=!"
"%PY%" "%ROOT%\predict.py" --image "!IMG!"
pause
exit /b 0

:do_predict_tiles
call :resolve_python
call :need_model
if errorlevel 1 pause & exit /b 1
echo.
echo === Predict tiles ===
echo Example folder: validation\jumping
set "TDIR="
set /p TDIR=Folder: 
if "!TDIR!"=="" pause & exit /b 1
set "TDIR=!TDIR:"=!"
set "TARGET="
set /p TARGET=Target sitting or jumping: 
if /I not "!TARGET!"=="sitting" if /I not "!TARGET!"=="jumping" (
    echo Invalid target.
    pause
    exit /b 1
)
set "THR=0.5"
set /p THR=Threshold [0.5]: 
if "!THR!"=="" set "THR=0.5"
"%PY%" "%ROOT%\predict.py" --tiles-dir "!TDIR!" --target "!TARGET!" --threshold !THR!
pause
exit /b 0

:need_model
if exist "%ROOT%\models\pose_detection_model.h5" exit /b 0
echo.
echo [ERROR] Train first ^(menu 3 or 5^).
exit /b 1

:err
echo.
echo Failed. See errors above.
pause
exit /b 1

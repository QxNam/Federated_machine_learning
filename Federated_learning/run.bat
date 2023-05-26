@echo off

echo Starting server
start /B python server.py
timeout /T 3 >nul

REM Define the client files to run
set "clientFiles=client_inbreast.py client_mias.py"

for %%F in (%clientFiles%) do (
    echo Starting %%F
    start /B python %%F
)

REM This will allow you to use CTRL+C to stop all background processes
set "stop=false"
for /F "usebackq tokens=2 delims=,=" %%A in (`tasklist /FI "WINDOWTITLE eq server.py" /FO csv`) do (
    if "%%~A" NEQ "" (
        set "stop=true"
        set "PID=%%~A"
    )
)
if %stop%==true (
    echo Press CTRL+C to stop all processes
    :waitForExit
    timeout /T 1 >nul
    tasklist /FI "PID eq %PID%" | findstr /C:"%PID%" >nul
    if errorlevel 1 (
        exit /b
    )
    goto :waitForExit
)

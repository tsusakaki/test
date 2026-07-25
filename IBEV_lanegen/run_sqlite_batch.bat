@echo off
setlocal
if "%~2"=="" (
  echo Usage: %~nx0 ^<input.sqlite^> ^<output-root^> [additional LaneGen options]
  exit /b 2
)
set "DB=%~1"
set "OUT=%~2"
shift
shift
python "%~dp0lanegen_run.py" --input "%DB%" --output "%OUT%" %*
exit /b %ERRORLEVEL%

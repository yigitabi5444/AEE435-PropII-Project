@echo off
setlocal

set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%\..
python -m pip install -e . --upgrade
popd

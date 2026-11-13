@echo off
rem qwencoin on this Windows machine, inside WSL2: 4 nodes of the chain, then every model size merged,
rem checked and published (see README.txt).
rem   go.cmd            everything, resumable
rem   go.cmd status ^| watch ^| down ^| up N ^| chat ^| publish ^| seed [status ^| down]
rem   go.cmd ask "How many coins does this node have?"
setlocal
cd /d "%~dp0"
where wsl.exe >nul 2>nul || goto nowsl
rem the probe is tried twice: a key pressed at the wrong moment must not read as "no WSL"
wsl.exe -e bash -c "echo ok" >nul 2>nul || wsl.exe -e bash -c "echo ok" >nul 2>nul || goto nowsl
wsl.exe -e bash kit/go.sh %*
set RC=%errorlevel%
echo.
if not "%RC%"=="0" echo It stopped with an error: see logs\pipeline.log in this folder.
pause
exit /b %RC%
:nowsl
echo.
echo WSL2 with a Linux distro is needed: the node and ord are Linux programs, and so is the tested pipeline.
echo In an administrator PowerShell:   wsl --install -d Ubuntu-24.04
echo Reboot if it asks, open "Ubuntu" once to create a user, then run go.cmd again.
pause
exit /b 1

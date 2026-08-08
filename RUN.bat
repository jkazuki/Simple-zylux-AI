@echo off
:menu
cls
echo ======================================================
echo    MENU AI
echo ======================================================
echo 1. training (train.py)
echo 2. chat (chat_client.py)
echo 3. exit
echo ======================================================
set /p choice="chon (1-3): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto chat
if "%choice%"=="3" exit

:train
echo Dang bat dau huan luyen...
py -3.12 train.py
pause
goto menu

:chat
echo Dang khoi dong che do Chat...
py -3.12 chat_client.py
pause
goto menu

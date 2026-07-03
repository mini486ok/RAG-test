@echo off
chcp 65001 >nul
title RAIL-RAG Auth Proxy
cd /d "%~dp0"

if exist accounts.json goto run
echo [안내] 등록된 계정이 없습니다. 먼저 계정을 만드세요:
echo    python auth_proxy.py add-user 아이디
pause
exit /b 1

:run
echo ─────────────────────────────────────────
echo  RAIL-RAG LAB 인증 프록시를 시작합니다. (이 창을 닫으면 중단됩니다)
echo ─────────────────────────────────────────
python auth_proxy.py serve
echo.
echo [알림] 프록시가 종료되었습니다.
pause

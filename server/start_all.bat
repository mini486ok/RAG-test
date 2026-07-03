@echo off
chcp 65001 >nul
title RAIL-RAG Launcher
cd /d "%~dp0"

if exist accounts.json goto ok
echo [안내] 등록된 계정이 없습니다. 먼저 이 창에서 계정을 만드세요:
echo    python auth_proxy.py add-user 아이디
echo 계정 생성 후 이 파일을 다시 실행하세요.
cmd /k
exit /b 1

:ok
echo 인증 프록시와 터널을 각각 새 창으로 시작합니다...
start "" "%~dp0start_server.bat"
ping -n 4 127.0.0.1 >nul
start "" "%~dp0start_tunnel.bat"
echo 완료. 터널 창에 표시되는 trycloudflare.com 주소를 공유하세요.
ping -n 6 127.0.0.1 >nul

@echo off
chcp 65001 >nul
title RAIL-RAG Tunnel
setlocal

rem ── cloudflared 실행 파일 탐색 (PATH → 표준 설치 경로) ──
set "CFD=cloudflared"
where cloudflared >nul 2>nul
if not errorlevel 1 goto run

set "CFD=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if exist "%CFD%" goto run

set "CFD=C:\Program Files\cloudflared\cloudflared.exe"
if exist "%CFD%" goto run

echo cloudflared가 설치되어 있지 않습니다. 설치를 진행합니다...
winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
set "CFD=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if exist "%CFD%" goto run
echo.
echo [오류] 설치 후에도 cloudflared를 찾을 수 없습니다. PC를 재시작한 뒤 다시 실행해 주세요.
pause
exit /b 1

:run
echo ─────────────────────────────────────────
echo  Cloudflare Tunnel을 시작합니다. (이 창을 닫으면 중단됩니다)
echo  아래에 표시되는 https://xxxx.trycloudflare.com 주소를
echo  다음 형식으로 공유하세요:
echo  https://mini486ok.github.io/RAG-test/?server=https://xxxx.trycloudflare.com
echo ─────────────────────────────────────────
"%CFD%" tunnel --url http://localhost:8790
echo.
echo [알림] 터널이 종료되었습니다.
pause

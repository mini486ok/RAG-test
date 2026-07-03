@echo off
chcp 65001 >nul
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared가 설치되어 있지 않습니다. 설치를 진행합니다...
  winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
  echo 설치 후 이 창을 닫고 다시 실행해 주세요.
  pause
  exit /b
)
echo ─────────────────────────────────────────
echo  Cloudflare Tunnel을 시작합니다.
echo  아래에 표시되는 https://xxxx.trycloudflare.com 주소를
echo  다음 형식으로 공유하세요:
echo  https://mini486ok.github.io/RAG-test/?server=https://xxxx.trycloudflare.com
echo ─────────────────────────────────────────
cloudflared tunnel --url http://localhost:8790
pause

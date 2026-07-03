@echo off
chcp 65001 >nul
setlocal

rem cloudflared 실행 파일 탐색 (PATH → 표준 설치 경로 순)
set "CFD=cloudflared"
where cloudflared >nul 2>nul
if errorlevel 1 (
  if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" (
    set "CFD=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
  ) else if exist "%ProgramFiles%\cloudflared\cloudflared.exe" (
    set "CFD=%ProgramFiles%\cloudflared\cloudflared.exe"
  ) else (
    echo cloudflared가 설치되어 있지 않습니다. 설치를 진행합니다...
    winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
    if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" (
      set "CFD=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
    ) else (
      echo 설치 후에도 찾을 수 없습니다. 새 터미널에서 다시 실행해 주세요.
      pause
      exit /b 1
    )
  )
)

echo ─────────────────────────────────────────
echo  Cloudflare Tunnel을 시작합니다.
echo  아래에 표시되는 https://xxxx.trycloudflare.com 주소를
echo  다음 형식으로 공유하세요:
echo  https://mini486ok.github.io/RAG-test/?server=https://xxxx.trycloudflare.com
echo ─────────────────────────────────────────
"%CFD%" tunnel --url http://localhost:8790
pause

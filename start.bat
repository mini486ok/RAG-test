@echo off
chcp 65001 >nul
echo ─────────────────────────────────────────
echo  RAIL·RAG LAB 로컬 서버를 시작합니다.
echo  브라우저에서 http://localhost:8000 접속
echo ─────────────────────────────────────────
where python >nul 2>nul
if %errorlevel%==0 (
  start http://localhost:8000
  python -m http.server 8000
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    start http://localhost:8000
    py -m http.server 8000
  ) else (
    echo Python이 설치되어 있지 않습니다. https://python.org 에서 설치하세요.
    pause
  )
)

@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ─────────────────────────────────────────
echo  RAIL·RAG LAB 인증 프록시를 시작합니다.
echo  (계정이 없다면 먼저: python auth_proxy.py add-user 아이디)
echo ─────────────────────────────────────────
python auth_proxy.py serve
pause

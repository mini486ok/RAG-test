# -*- coding: utf-8 -*-
"""
RAIL·RAG LAB 인증 프록시
========================
로컬 Ollama(기본 http://localhost:11434) 앞단에서 동작하는 경량 인증·쿼터 프록시.
외부(다른 PC) 사용자가 이 PC의 LLM을 쓰려면 반드시 이 프록시를 거쳐야 하며,
아이디/비밀번호(Basic Auth)와 계정별 일일 호출 한도를 강제한다.

사용법
------
  계정 추가:      python auth_proxy.py add-user <아이디> [--limit 200]
  계정 삭제:      python auth_proxy.py remove-user <아이디>
  계정 목록:      python auth_proxy.py list
  서버 실행:      python auth_proxy.py serve [--port 8790] [--ollama http://localhost:11434]

외부 공개(예: Cloudflare Tunnel):
  cloudflared tunnel --url http://localhost:8790
  → 발급된 https URL을 웹앱 접속 링크에 붙여 공유:
    https://mini486ok.github.io/RAG-test/?server=https://<발급된주소>

의존성: Python 3.8+ 표준 라이브러리만 사용.
계정 파일(accounts.json)에는 비밀번호가 아닌 salt+SHA-256 해시만 저장된다.
"""

import argparse
import base64
import getpass
import hashlib
import http.client
import json
import os
import secrets
import sys
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS_PATH = os.path.join(BASE_DIR, 'accounts.json')
USAGE_PATH = os.path.join(BASE_DIR, 'usage.json')

# 프록시를 통과시킬 Ollama API 경로 (그 외는 전부 차단)
ALLOWED_PATHS = {'/api/tags', '/api/chat', '/api/embed', '/api/version'}
# 일일 한도에 집계되는 경로 (GPU를 실제로 사용하는 호출)
COUNTED_PATHS = {'/api/chat', '/api/embed'}

DEFAULT_DAILY_LIMIT = 200
MAX_BODY_BYTES = 4 * 1024 * 1024  # 요청 본문 4MB 상한
FAIL_LIMIT = 10                   # IP당 인증 실패 허용 횟수
FAIL_WINDOW_SEC = 600             # 실패 카운터/차단 유지 시간

_lock = threading.Lock()
_fail_by_ip = {}   # ip -> {count, until}
_usage_cache = None


# ──────────────────────── 계정/사용량 파일 ────────────────────────

def load_accounts():
    if not os.path.exists(ACCOUNTS_PATH):
        return {'users': {}}
    with open(ACCOUNTS_PATH, encoding='utf-8') as f:
        return json.load(f)


def save_accounts(data):
    with open(ACCOUNTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hash_pw(salt: str, password: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()


def load_usage():
    global _usage_cache
    if _usage_cache is None:
        if os.path.exists(USAGE_PATH):
            try:
                with open(USAGE_PATH, encoding='utf-8') as f:
                    _usage_cache = json.load(f)
            except Exception:
                _usage_cache = {}
        else:
            _usage_cache = {}
    return _usage_cache


def bump_usage(user: str) -> int:
    """오늘 사용량 +1 후 값 반환 (파일에도 반영)."""
    today = date.today().isoformat()
    with _lock:
        usage = load_usage()
        day = usage.setdefault(today, {})
        day[user] = day.get(user, 0) + 1
        # 이전 날짜는 7일치만 유지
        for k in sorted(usage.keys())[:-7]:
            del usage[k]
        try:
            with open(USAGE_PATH, 'w', encoding='utf-8') as f:
                json.dump(usage, f)
        except Exception:
            pass
        return day[user]


def today_usage(user: str) -> int:
    usage = load_usage()
    return usage.get(date.today().isoformat(), {}).get(user, 0)


# ──────────────────────── HTTP 핸들러 ────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    server_version = 'RailRagAuthProxy/1.0'
    protocol_version = 'HTTP/1.1'  # chunked 스트리밍에 필수
    ollama_url = 'http://localhost:11434'
    accounts = {'users': {}}

    # ---- 유틸 ----

    def client_ip(self):
        # Cloudflare/프록시 뒤에서는 헤더 우선
        return (self.headers.get('CF-Connecting-IP')
                or self.headers.get('X-Forwarded-For', '').split(',')[0].strip()
                or self.client_address[0])

    def cors_origin(self):
        origin = self.headers.get('Origin', '')
        if not origin:
            return None
        host = urlparse(origin).hostname or ''
        if origin == 'https://mini486ok.github.io' or host in ('localhost', '127.0.0.1'):
            return origin
        return None

    def send_cors(self):
        origin = self.cors_origin()
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
            self.send_header('Access-Control-Max-Age', '86400')

    def reply_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_cors()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        if status == 401:
            # 브라우저 기본 인증 팝업을 띄우지 않도록 Basic 스킴 헤더는 생략
            self.send_header('X-Auth-Required', 'true')
        self.end_headers()
        self.wfile.write(body)

    # ---- 인증 ----

    def check_auth(self):
        """인증 성공 시 사용자ID 반환, 실패 시 None (응답도 전송)."""
        ip = self.client_ip()
        now = time.time()
        with _lock:
            rec = _fail_by_ip.get(ip)
            if rec and rec['count'] >= FAIL_LIMIT and now < rec['until']:
                self.reply_json(429, {'error': '인증 실패가 많아 잠시 차단되었습니다. 10분 후 다시 시도하세요.'})
                return None
            if rec and now >= rec['until']:
                del _fail_by_ip[ip]

        header = self.headers.get('Authorization', '')
        if header.startswith('Basic '):
            try:
                decoded = base64.b64decode(header[6:]).decode('utf-8')
                user, _, pw = decoded.partition(':')
            except Exception:
                user, pw = '', ''
            acc = self.accounts['users'].get(user)
            if acc and hash_pw(acc['salt'], pw) == acc['hash']:
                with _lock:
                    _fail_by_ip.pop(ip, None)
                return user

        # 실패 기록
        with _lock:
            rec = _fail_by_ip.setdefault(ip, {'count': 0, 'until': 0})
            rec['count'] += 1
            rec['until'] = now + FAIL_WINDOW_SEC
        self.reply_json(401, {'error': '로그인이 필요합니다. 아이디와 비밀번호를 확인하세요.'})
        return None

    # ---- 요청 처리 ----

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/healthz':
            self.reply_json(200, {'ok': True})
            return
        self.handle_proxy('GET')

    def do_POST(self):
        self.handle_proxy('POST')

    def handle_proxy(self, method):
        path = self.path.split('?')[0]
        if path not in ALLOWED_PATHS:
            self.reply_json(404, {'error': '허용되지 않은 경로입니다.'})
            return

        user = self.check_auth()
        if not user:
            return

        # 일일 한도 검사 (GPU 사용 호출만)
        if path in COUNTED_PATHS:
            limit = self.accounts['users'][user].get('daily_limit', DEFAULT_DAILY_LIMIT)
            used = today_usage(user)
            if used >= limit:
                self.reply_json(429, {'error': f'일일 호출 한도({limit}회)를 초과했습니다. 내일 다시 이용하세요.'})
                return
            bump_usage(user)

        # 본문 읽기
        body = b''
        if method == 'POST':
            length = int(self.headers.get('Content-Length', 0) or 0)
            if length > MAX_BODY_BYTES:
                self.reply_json(413, {'error': '요청 본문이 너무 큽니다.'})
                return
            body = self.rfile.read(length)

        # 업스트림(Ollama)으로 전달
        target = urlparse(self.ollama_url)
        try:
            conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=600)
            conn.request(method, path, body=body if body else None,
                         headers={'Content-Type': 'application/json'})
            upstream = conn.getresponse()
        except Exception as e:
            self.reply_json(502, {'error': f'Ollama 서버에 연결할 수 없습니다: {e}'})
            return

        try:
            self.send_response(upstream.status)
            self.send_cors()
            self.send_header('Content-Type', upstream.getheader('Content-Type', 'application/json'))
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            # 스트리밍 릴레이 (NDJSON 토큰 스트림 대응)
            while True:
                chunk = upstream.read(8192)
                if not chunk:
                    break
                self.wfile.write(f'{len(chunk):X}\r\n'.encode('ascii'))
                self.wfile.write(chunk)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 클라이언트 중단(사용자 Stop 등) — 정상 상황
        finally:
            conn.close()

    def log_message(self, fmt, *args):
        sys.stderr.write('[%s] %s %s\n' % (self.client_ip(), self.log_date_time_string(), fmt % args))


# ──────────────────────── CLI ────────────────────────

def cmd_add_user(args):
    data = load_accounts()
    if args.username in data['users'] and not args.force:
        print(f'이미 존재하는 계정입니다: {args.username} (덮어쓰려면 --force)')
        return 1
    pw = getpass.getpass(f'{args.username} 비밀번호: ')
    pw2 = getpass.getpass('비밀번호 확인: ')
    if pw != pw2:
        print('비밀번호가 일치하지 않습니다.')
        return 1
    if len(pw) < 6:
        print('비밀번호는 6자 이상으로 하세요.')
        return 1
    salt = secrets.token_hex(16)
    data['users'][args.username] = {
        'salt': salt,
        'hash': hash_pw(salt, pw),
        'daily_limit': args.limit,
        'created_at': date.today().isoformat(),
    }
    save_accounts(data)
    print(f'계정 생성 완료: {args.username} (일일 한도 {args.limit}회)')
    return 0


def cmd_remove_user(args):
    data = load_accounts()
    if args.username not in data['users']:
        print(f'계정이 없습니다: {args.username}')
        return 1
    del data['users'][args.username]
    save_accounts(data)
    print(f'계정 삭제 완료: {args.username}')
    return 0


def cmd_list(_args):
    data = load_accounts()
    if not data['users']:
        print('등록된 계정이 없습니다. add-user로 추가하세요.')
        return 0
    print(f'{"아이디":<20} {"일일한도":>8} {"오늘사용":>8}')
    for name, acc in data['users'].items():
        print(f'{name:<20} {acc.get("daily_limit", DEFAULT_DAILY_LIMIT):>8} {today_usage(name):>8}')
    return 0


def cmd_serve(args):
    accounts = load_accounts()
    if not accounts['users']:
        print('⚠ 등록된 계정이 없습니다. 먼저 계정을 추가하세요:')
        print('   python auth_proxy.py add-user <아이디>')
        return 1
    ProxyHandler.ollama_url = args.ollama
    ProxyHandler.accounts = accounts
    server = ThreadingHTTPServer(('0.0.0.0', args.port), ProxyHandler)
    print('─' * 52)
    print(f' RAIL·RAG LAB 인증 프록시 시작')
    print(f'  수신 포트  : http://localhost:{args.port}')
    print(f'  Ollama     : {args.ollama}')
    print(f'  계정 수    : {len(accounts["users"])}개')
    print()
    print(' 외부 공개(Cloudflare Tunnel):')
    print(f'   cloudflared tunnel --url http://localhost:{args.port}')
    print(' 공유 링크 형식:')
    print('   https://mini486ok.github.io/RAG-test/?server=<터널주소>')
    print('─' * 52)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n종료합니다.')
    return 0


def main():
    ap = argparse.ArgumentParser(description='RAIL·RAG LAB 인증 프록시')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('add-user', help='계정 추가')
    p.add_argument('username')
    p.add_argument('--limit', type=int, default=DEFAULT_DAILY_LIMIT, help='일일 LLM 호출 한도 (기본 200)')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_add_user)

    p = sub.add_parser('remove-user', help='계정 삭제')
    p.add_argument('username')
    p.set_defaults(func=cmd_remove_user)

    p = sub.add_parser('list', help='계정 목록/사용량')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('serve', help='프록시 서버 실행')
    p.add_argument('--port', type=int, default=8790)
    p.add_argument('--ollama', default='http://localhost:11434')
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()

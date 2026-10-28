"""情商外挂：基于《非暴力沟通》的消息改写工具。

运行：python3 app.py            （默认 http://127.0.0.1:8787）

模型接口优先在页面右上角的"设置"里配置，保存在同目录的 config.json（只有本机用户可读）。
没有配置时，回退到环境变量：
  LLM_BASE_URL  默认 https://api.deepseek.com/v1（任意 OpenAI 兼容接口）
  LLM_API_KEY   接口密钥
  LLM_MODEL     默认 deepseek-flash
其他：
  PORT          默认 8787
  MOCK=1        不调用模型，返回固定示例，用于本地自测

设置口令在第一次启动时生成，打印在终端里；修改配置必须带上它，分享链接的使用者拿不到。

接口：POST /api/rewrite
  {"text": "...", "mode": "send|reply", "who": "client|boss|colleague|report|partner|family|friend",
   "tone": "gentle|firm|direct", "goal": "auto|push|decline|apologize|ask|complain|comfort", "model": "..."}
"""

import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.error
from collections import deque
from datetime import date
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"
PORT = int(os.environ.get("PORT", "8787"))
MOCK = os.environ.get("MOCK") == "1"
MAX_INPUT = 800
LOCK = threading.Lock()


def env_key():
    return os.environ.get("LLM_API_KEY", "").strip()


def save_config(cfg):
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(CONFIG_PATH)


def load_config():
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text("utf-8"))
        except (ValueError, OSError):
            cfg = {}
    if not cfg.get("admin_token"):
        cfg["admin_token"] = secrets.token_urlsafe(9)
        save_config(cfg)
    return cfg


CONFIG = load_config()


def current():
    """当前生效的接口地址、密钥、允许的模型和默认模型。"""
    base_url = (CONFIG.get("base_url") or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    api_key = CONFIG.get("api_key") or env_key()
    models = [m for m in CONFIG.get("models", []) if m] or [os.environ.get("LLM_MODEL", DEFAULT_MODEL)]
    default = CONFIG.get("default_model") if CONFIG.get("default_model") in models else models[0]
    return base_url, api_key, models, default


DEFAULT_LIMITS = {"limit_per_minute": 10, "limit_per_day": 200}
USAGE = {"minute": deque(), "day": "", "count": 0}


def limits():
    return {k: int(CONFIG.get(k, v)) for k, v in DEFAULT_LIMITS.items()}


def take_quota():
    """所有使用者共用的次数上限；返回 None 表示放行。每天的计数只存在内存里，重启后清零。"""
    lim = limits()
    now, today = time.time(), date.today().isoformat()
    with LOCK:
        q = USAGE["minute"]
        while q and now - q[0] > 60:
            q.popleft()
        if USAGE["day"] != today:
            USAGE["day"], USAGE["count"] = today, 0
        if lim["limit_per_minute"] and len(q) >= lim["limit_per_minute"]:
            return "用的人有点多，请一分钟后再试"
        if lim["limit_per_day"] and USAGE["count"] >= lim["limit_per_day"]:
            return "今天的使用次数已经用完了，明天再来"
        q.append(now)
        USAGE["count"] += 1
    return None


def mask(key):
    if not key:
        return ""
    return key[:4] + "••••" + key[-4:] if len(key) > 12 else "••••"

WHO = {
    "client": ("客户", "专业、不卑不亢。先确认对方关心的事，再说自己的情况；不过度道歉，不承诺做不到的事；涉及钱、时间、范围的地方说清楚。"),
    "boss": ("上级", "先说结论和需要对方决定的事，再说原因；简短，不诉苦，不绕弯；有困难就给出可选方案让对方选。"),
    "colleague": ("同事", "平级商量的口吻，对事不对人；说清自己卡在哪、需要对方做什么、什么时候。"),
    "report": ("下属", "尊重对方，说清期望和标准，不贬低、不翻旧账；给对方说明情况的空间，最后落到具体的下一步。"),
    "partner": ("伴侣", "情绪先于道理。先接住对方的情绪，说出自己真实的感受，不讲大道理，不用职场腔和书面语，可以亲昵、口语，像平时聊天那样。"),
    "family": ("家人", "说软话、说具体的事，少说教；表达关心和自己的需要，别用命令口吻。"),
    "friend": ("朋友", "轻松、真诚，不端着；有不满就直接但友好地说出来，别阴阳怪气。"),
}

TONE = {
    "gentle": ("温和", "语气柔和，多给对方台阶。"),
    "firm": ("坚定", "态度清楚、立场不让步，但不攻击人；该拒绝就拒绝，该催就催。"),
    "direct": ("直接", "最短的话说清事实和请求，不铺垫、不客套。"),
}

GOAL = {
    "auto": "自动判断",
    "push": "催进度",
    "decline": "拒绝",
    "apologize": "道歉",
    "ask": "提要求",
    "complain": "表达不满",
    "comfort": "安慰对方",
}

SYSTEM_PROMPT = """你是「情商外挂」，帮人在消息发出去之前改一遍，方法来自马歇尔·卢森堡《非暴力沟通》。

用户会告诉你：这段话是自己想发出去的（send），还是别人发来、需要回复的（reply）；对方是谁；想要的语气；这次的目的。

先找出原话里容易让对方不舒服的地方：道德评判和贴标签（「你太不靠谱了」）、一概而论（「总是」「从不」「每次」）、推卸责任（「我不得不」「你让我」）、命令和威胁、「应该」「活该」、阴阳怪气和反问。reply 场景下，找的是对方话里的情绪和真正在意的事。

再用观察、感受、需要、请求重新组织：
- 观察：具体发生了什么，不加评判；
- 感受：真实的情绪词（着急、担心、委屈、为难），不用「我感到被忽视」这类伪感受；
- 需要：自己真正在意的是什么，为自己的感受负责；
- 请求：正向、具体、对方此刻能回应。
reply 场景下，回复里先接住对方，再说自己的情况和请求。

改写要求：
- 写成可以直接在微信、飞书、钉钉里发的消息，符合对方身份和指定语气；
- 不要套「当我看到……我感到……因为我需要……」的固定句式，四要素融进口语；
- 非暴力沟通不是讨好，语气是坚定或直接时，立场不能被改软；
- 不过度客套和道歉，不加 emoji，不编造原话里没有的事实；缺关键信息就用【】标出来让用户补；
- 长度和原话相近，最多不超过原话的两倍。
- 所有文字内容里如果要用引号，一律用「」，不要用英文双引号，否则 JSON 会解析失败。

只输出一个 JSON 对象，不要输出任何其他文字：
{
  "risk": "原话发出去的风险，low、medium 或 high",
  "triggers": [{"text": "原话里的原词或原句", "why": "为什么容易惹毛对方，一句话"}],
  "message": "改写后可以直接发的消息",
  "short": "更短的一句话版本",
  "changes": ["改了哪里，每条一句话，最多三条"],
  "ofnr": {"observation": "观察", "feeling": "感受", "need": "需要", "request": "请求"}
}
triggers 最多三条，必须是原话里真实出现的文字；原话没有问题就返回空数组，risk 为 low。"""

MOCK_RESULT = {
    "risk": "high",
    "triggers": [
        {"text": "你怎么又拖到现在", "why": "“又”是在翻旧账，对方会先忙着辩解"},
        {"text": "每次都要我催", "why": "一概而论，等于给对方贴了拖延的标签"},
    ],
    "message": "周报说好中午交，我这边现在还没收到你那份，有点着急，因为【后面要用周报做的事】。你看能不能【时间】前发我？要是哪里卡住了也跟我说一声，我们一起看看。",
    "short": "周报我这边还没收到，能【时间】前发我吗？",
    "changes": [
        "把“你怎么又拖”“每次都要我催”改成具体的事实：说好中午交、现在还没收到",
        "说出自己着急的原因，而不是指责对方",
        "把命令改成商量，并给对方留了说明困难的口子",
    ],
    "ofnr": {
        "observation": "说好中午交周报，现在还没收到",
        "feeling": "着急",
        "need": "进度可预期",
        "request": "约定一个时间发过来，有困难提前说",
    },
}


def build_user_message(text, mode, who, tone, goal):
    who_name, who_rule = WHO[who]
    tone_name, tone_rule = TONE[tone]
    kind = "send（我想发出去的话）" if mode == "send" else "reply（别人发给我的话，帮我回）"
    return (
        f"类型：{kind}\n"
        f"对方是：{who_name}。{who_rule}\n"
        f"语气：{tone_name}。{tone_rule}\n"
        f"目的：{GOAL[goal]}\n"
        f"原话：\n{text}"
    )


def http_json(url, api_key, body=None, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = {400: "请求不被接受，常见原因是模型名在这个平台上不存在", 401: "密钥不对或已过期", 403: "没有权限使用这个模型",
                404: "接口地址或模型名不对", 429: "调用太频繁或余额不足"}.get(e.code, "")
        detail = ""
        try:
            err = json.loads(e.read().decode("utf-8", "replace"))
            err = err.get("error", err)
            detail = str(err.get("message") if isinstance(err, dict) else err)[:200]
        except Exception:
            pass
        msg = f"模型接口返回 {e.code}" + (f"：{hint}" if hint else "") + (f"（平台原话：{detail}）" if detail else "")
        raise RuntimeError(msg) from None
    except urllib.error.URLError:
        raise RuntimeError("连不上模型接口，请检查接口地址和网络") from None


NO_JSON_MODE = set()


def repair_json(raw):
    """把字符串里没转义的英文双引号转义掉：后面紧跟 , } ] : 的才算字符串结束。"""
    out, in_str, i = [], False, 0
    while i < len(raw):
        ch = raw[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
        elif ch == "\\":
            out.append(raw[i:i + 2])
            i += 1
        elif ch == '"':
            rest = raw[i + 1:].lstrip()
            if not rest or rest[0] in ",}]:":
                out.append(ch)
                in_str = False
            else:
                out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def parse_result(content):
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        raise RuntimeError("模型没有按格式返回，请再点一次，或换一个模型")
    raw = match.group(0)
    for candidate in (raw, repair_json(raw)):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise RuntimeError("模型返回的格式有点问题，请再点一次；经常出现的话换一个模型")


def call_model(text, mode, who, tone, goal, model):
    if MOCK:
        return {**MOCK_RESULT, "mock": True}
    base_url, api_key, _, _ = current()
    if not api_key:
        raise RuntimeError("还没有配置模型接口，请点右上角的设置填写 API Key")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(text, mode, who, tone, goal)},
        ],
        "temperature": 0.6,
        "max_tokens": 1000,
    }
    url = f"{base_url}/chat/completions"
    if base_url not in NO_JSON_MODE:
        try:
            data = http_json(url, api_key, {**body, "response_format": {"type": "json_object"}})
            return parse_result(data["choices"][0]["message"]["content"])
        except RuntimeError as e:
            if not str(e).startswith("模型接口返回 400"):
                raise
        data = http_json(url, api_key, body)
        NO_JSON_MODE.add(base_url)
    else:
        data = http_json(url, api_key, body)
    return parse_result(data["choices"][0]["message"]["content"])


def list_remote_models(base_url, api_key):
    data = http_json(f"{base_url.rstrip('/')}/models", api_key, timeout=20)
    ids = sorted({str(m.get("id")) for m in data.get("data", []) if m.get("id")})
    return ids[:1000]


def pick(value, options, default):
    return value if value in options else default


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def send_json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(min(length, 20000)).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def is_admin(self):
        token = self.headers.get("X-Admin-Token", "")
        return bool(token) and hmac.compare_digest(token, CONFIG["admin_token"])

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            raw = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/health":
            _, api_key, models, default = current()
            self.send_json(200, {"ok": True, "mock": MOCK, "configured": bool(api_key), "models": models, "default_model": default})
        elif path == "/api/admin/config":
            if not self.is_admin():
                self.send_json(401, {"error": "设置口令不对"})
                return
            base_url, api_key, models, default = current()
            self.send_json(200, {
                "base_url": base_url,
                "api_key_masked": mask(api_key),
                "api_key_source": "config" if CONFIG.get("api_key") else ("env" if api_key else ""),
                "models": models,
                "default_model": default,
                **limits(),
                "used_today": USAGE["count"] if USAGE["day"] == date.today().isoformat() else 0,
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlsplit(self.path).path
        payload = self.read_json()
        if payload is None:
            self.send_json(400, {"error": "请求格式不对"})
            return
        if path == "/api/rewrite":
            self.handle_rewrite(payload)
        elif path == "/api/admin/login":
            ok = hmac.compare_digest(str(payload.get("token", "")), CONFIG["admin_token"])
            self.send_json(200 if ok else 401, {"ok": ok} if ok else {"error": "设置口令不对，口令在启动工具的终端里"})
        elif path == "/api/admin/config":
            if not self.is_admin():
                self.send_json(401, {"error": "设置口令不对"})
                return
            self.handle_save(payload)
        elif path == "/api/admin/remote-models":
            if not self.is_admin():
                self.send_json(401, {"error": "设置口令不对"})
                return
            base_url = str(payload.get("base_url") or current()[0]).strip()
            api_key = str(payload.get("api_key") or "").strip() or current()[1]
            if not api_key:
                self.send_json(400, {"error": "先填 API Key"})
                return
            try:
                self.send_json(200, {"models": list_remote_models(base_url, api_key)})
            except Exception as e:
                self.send_json(502, {"error": f"{e}。这个平台可能不提供模型列表，可以手动填写模型名"})
        else:
            self.send_json(404, {"error": "not found"})

    def handle_save(self, payload):
        base_url = str(payload.get("base_url", "")).strip().rstrip("/")
        if base_url and not re.match(r"^https?://", base_url):
            self.send_json(400, {"error": "接口地址要以 http:// 或 https:// 开头"})
            return
        models = [str(m).strip() for m in payload.get("models", []) if str(m).strip()][:50]
        if not models:
            self.send_json(400, {"error": "至少选一个模型"})
            return
        default = str(payload.get("default_model", "")).strip()
        check_base = base_url or current()[0]
        check_key = str(payload.get("api_key", "")).strip() or current()[1]
        if check_key:
            try:
                remote = set(list_remote_models(check_base, check_key))
            except Exception:
                remote = set()
            missing = [m for m in models if remote and m not in remote]
            if missing:
                self.send_json(400, {"error": f"这些模型在 {check_base} 上不存在：{'、'.join(missing)}。点“从接口拉取模型列表”重新勾选"})
                return
        with LOCK:
            if base_url:
                CONFIG["base_url"] = base_url
            new_key = str(payload.get("api_key", "")).strip()
            if new_key:
                CONFIG["api_key"] = new_key
            if payload.get("clear_key"):
                CONFIG.pop("api_key", None)
            CONFIG["models"] = models
            CONFIG["default_model"] = default if default in models else models[0]
            for k in DEFAULT_LIMITS:
                if k in payload:
                    try:
                        CONFIG[k] = max(0, min(100000, int(payload[k])))
                    except (TypeError, ValueError):
                        pass
            save_config(CONFIG)
        self.send_json(200, {"ok": True})

    def handle_rewrite(self, payload):
        text = str(payload.get("text", "")).strip()
        if not text:
            self.send_json(400, {"error": "先输入一段话"})
            return
        if len(text) > MAX_INPUT:
            self.send_json(400, {"error": f"一次最多 {MAX_INPUT} 字"})
            return
        _, _, models, default = current()
        mode = "reply" if payload.get("mode") == "reply" else "send"
        who = pick(payload.get("who"), WHO, "colleague")
        tone = pick(payload.get("tone"), TONE, "gentle")
        goal = pick(payload.get("goal"), GOAL, "auto")
        model = pick(payload.get("model"), models, default)
        if not MOCK:
            blocked = take_quota()
            if blocked:
                self.send_json(429, {"error": blocked})
                return
        try:
            self.send_json(200, call_model(text, mode, who, tone, goal, model))
        except Exception as e:
            self.send_json(502, {"error": str(e)})


if __name__ == "__main__":
    _, _, models, default = current()
    print(f"情商外挂 已启动：http://127.0.0.1:{PORT}  模型：{'MOCK' if MOCK else default}")
    print(f"设置口令：{CONFIG['admin_token']}（点页面右上角的设置时输入，只在这台电脑的终端里显示，别发给别人）")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

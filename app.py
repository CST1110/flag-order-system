"""旗標訂餐系統 - 後端 API

啟動方式：
    pip install -r requirements.txt
    python app.py
然後打開 http://localhost:5000

API：
    POST /api/analyze-menu   {"url": "..."}            -> {"items": [...], "source": "claude" | "rule"}
    POST /api/order          {"name": "...", "items": [...]} -> {"ok": true, "order": {...}}
    GET  /api/orders                                    -> {"orders": [...], "summary": [...]}

菜單分析：
    有設定 ANTHROPIC_API_KEY 時，用 Claude 從網頁文字中擷取品項（含客製化選項）；
    沒有設定時，改用簡單的規則（找「品名 + 價格」的行）來解析。
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "orders.json"
CLAUDE_MODEL = "claude-opus-5-5"
MAX_PAGE_BYTES = 5 * 1024 * 1024  # 菜單網頁最大 5MB

app = Flask(__name__, static_folder=None)
app.json.ensure_ascii = False  # JSON 直接輸出中文
orders_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 前端頁面
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ---------------------------------------------------------------------------
# 菜單分析
# ---------------------------------------------------------------------------

class MenuItem(BaseModel):
    name: str = Field(description="品名")
    price: int = Field(description="價格（新台幣，整數）；有多種尺寸時填最低價")
    options: List[str] = Field(description="客製化選項，例如：大杯、少冰、半糖、小辣；沒有就給空陣列")


class Menu(BaseModel):
    items: List[MenuItem]


class AnalyzeError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def fetch_page_text(url: str) -> str:
    """下載網頁並轉成純文字（去掉 script/style 等雜訊）。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AnalyzeError("網址格式不正確，請輸入 http:// 或 https:// 開頭的網址")

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MenuOrderBot/1.0)"},
            stream=True,
        )
        resp.raise_for_status()
        content = resp.raw.read(MAX_PAGE_BYTES + 1, decode_content=True)
    except requests.RequestException as e:
        raise AnalyzeError(f"無法讀取菜單網址：{e}", 502)

    if len(content) > MAX_PAGE_BYTES:
        raise AnalyzeError("網頁太大，無法分析", 413)

    encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else None
    soup = BeautifulSoup(content, "html.parser", from_encoding=encoding)
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text:
        raise AnalyzeError("網頁中沒有可讀的文字（菜單可能是圖片或由 JavaScript 產生）", 422)
    return text


def has_claude_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def analyze_with_claude(page_text: str) -> List[dict]:
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        "以下是一間餐廳菜單網頁的文字內容。請擷取出所有可以點的餐點與飲料品項。\n"
        "- name：品名（保留原文）\n"
        "- price：新台幣價格，整數；若有多種尺寸，填最低價並把尺寸放進 options\n"
        "- options：客製化選項（尺寸、甜度、冰塊、辣度、加料等），沒有就給空陣列\n"
        "忽略導覽列、店家介紹、營業時間等非品項內容。\n\n"
        f"<menu_page>\n{page_text}\n</menu_page>"
    )

    try:
        response = client.beta.messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
            output_format=Menu,
        )
    except anthropic.AuthenticationError:
        raise AnalyzeError("Claude API 金鑰無效，請檢查 ANTHROPIC_API_KEY", 500)
    except anthropic.RateLimitError:
        raise AnalyzeError("Claude API 使用量已達上限，請稍後再試", 429)
    except anthropic.APIStatusError as e:
        raise AnalyzeError(f"Claude API 錯誤：{e.message}", 502)
    except anthropic.APIConnectionError:
        raise AnalyzeError("無法連線到 Claude API", 502)

    if response.stop_reason == "refusal":
        raise AnalyzeError("Claude 無法分析這個網頁", 422)
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise AnalyzeError("菜單內容太多，分析未完成", 422)

    return [item.model_dump() for item in response.parsed_output.items]


# 規則解析：找「品名 … 價格」的行，例如「珍珠奶茶 55」「雞腿便當 $110」「牛肉麵 NT$150 元」
PRICE_LINE = re.compile(
    r"^(?P<name>[^\d$＄]{1,30}?)[\s:：.\-…|/]*(?:NT\$|NTD|\$|＄)?\s*(?P<price>\d{2,4})\s*元?$"
)
PRICE_ONLY = re.compile(r"^(?:NT\$|NTD|\$|＄)?\s*(?P<price>\d{2,4})\s*元?$")


def analyze_with_rules(page_text: str) -> List[dict]:
    items, seen = [], set()
    lines = page_text.splitlines()
    for i, line in enumerate(lines):
        m = PRICE_LINE.match(line)
        if m:
            name, price = m.group("name").strip(" :：.-…|/"), int(m.group("price"))
        else:
            # 品名與價格分在兩行的情況
            m = PRICE_ONLY.match(line)
            if not m or i == 0:
                continue
            name, price = lines[i - 1].strip(), int(m.group("price"))
            if not name or len(name) > 30 or re.search(r"\d", name):
                continue
        if not name or name in seen:
            continue
        seen.add(name)
        items.append({"name": name, "price": price, "options": []})
    return items


@app.post("/api/analyze-menu")
def analyze_menu():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify(error="請提供菜單網址"), 400

    try:
        text = fetch_page_text(url)
        if has_claude_credentials():
            items, source = analyze_with_claude(text), "claude"
        else:
            items, source = analyze_with_rules(text), "rule"
    except AnalyzeError as e:
        return jsonify(error=e.message), e.status

    if not items:
        return jsonify(error="沒有在網頁中找到任何品項"), 422
    return jsonify(items=items, source=source)


# ---------------------------------------------------------------------------
# 訂單
# ---------------------------------------------------------------------------

def load_orders() -> list:
    if not ORDERS_FILE.exists():
        return []
    try:
        return json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_orders(orders: list) -> None:
    tmp = ORDERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ORDERS_FILE)


@app.post("/api/order")
def create_order():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    raw_items = data.get("items")

    if not name:
        return jsonify(error="請輸入姓名"), 400
    if not isinstance(raw_items, list) or not raw_items:
        return jsonify(error="訂單沒有任何品項"), 400

    items = []
    for it in raw_items:
        if not isinstance(it, dict):
            return jsonify(error="品項格式錯誤"), 400
        try:
            quantity = int(it.get("quantity", 0))
            price = int(it.get("price", 0))
        except (TypeError, ValueError):
            return jsonify(error="數量或價格格式錯誤"), 400
        if quantity <= 0:
            continue
        if quantity > 99 or price < 0:
            return jsonify(error="數量或價格超出範圍"), 400
        items.append({
            "name": str(it.get("name", "")).strip()[:100],
            "price": price,
            "option": str(it.get("option", "")).strip()[:100],
            "quantity": quantity,
            "note": str(it.get("note", "")).strip()[:200],
        })

    if not items:
        return jsonify(error="請至少選擇一個品項"), 400

    order = {
        "id": uuid.uuid4().hex[:8],
        "name": name[:50],
        "menuUrl": str(data.get("menuUrl", ""))[:500],
        "items": items,
        "total": sum(i["price"] * i["quantity"] for i in items),  # 由後端重新計算
        "createdAt": datetime.now().isoformat(timespec="seconds"),
    }

    with orders_lock:
        orders = load_orders()
        orders.append(order)
        save_orders(orders)

    return jsonify(ok=True, order=order), 201


@app.get("/api/orders")
def list_orders():
    with orders_lock:
        orders = load_orders()

    # 依「品名 + 選項」彙總，方便統一跟店家訂
    summary = {}
    for order in orders:
        for it in order["items"]:
            key = (it["name"], it["option"])
            s = summary.setdefault(key, {"name": it["name"], "option": it["option"], "quantity": 0, "subtotal": 0})
            s["quantity"] += it["quantity"]
            s["subtotal"] += it["price"] * it["quantity"]

    return jsonify(
        orders=orders,
        summary=list(summary.values()),
        total=sum(o["total"] for o in orders),
    )


if __name__ == "__main__":
    mode = "Claude" if has_claude_credentials() else "規則解析（未設定 ANTHROPIC_API_KEY）"
    print(f"菜單分析模式：{mode}")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)

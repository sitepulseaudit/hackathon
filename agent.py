"""
LLMCite — Strands agent for Site Pulse Audit
Probe → score → draft → approve → re-check

Live path: submitted brand + product_url → Bedrock queries/answers for THAT site.
Mock path: SleepFix fixtures (offline demo).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from strands import tool
except ImportError:  # Lambda zip ships without strands; pipeline uses boto3 + stdlib only
    def tool(fn=None, **_kwargs):  # type: ignore[misc]
        """No-op stand-in for strands.tool when the SDK is not installed."""
        if fn is not None:
            return fn

        def _decorator(f):
            return f

        return _decorator

ROOT = Path(__file__).resolve().parent
QUERIES_PATH = ROOT / "queries" / "sleepfix.json"
FIXTURES_PATH = ROOT / "fixtures" / "probe_responses.json"
# AgentCore runtimes are often read-only except /tmp
_output = os.environ.get("LLMCITE_OUTPUT_DIR", str(ROOT / "output"))
OUTPUT_DIR = Path(_output)
ACTIVE_PACK_PATH = OUTPUT_DIR / "active_pack.json"

DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"
FETCH_TIMEOUT_SEC = 8
USER_AGENT = "Mozilla/5.0 (compatible; LLMCite/1.0; +https://sitepulseaudit.com)"

TEMPLATE_QUERIES = [
    "How do I raise Fitbit deep sleep?",
    "Best fix for waking tired after 8 hours",
    "What helps 3am waking?",
    "Deep sleep habit that actually works",
    "{brand} review",
    "30-day deep sleep program",
    "{brand}",
    "Find a deep sleep habit that sticks",
]
FALLBACK_COMPETITORS = ["competitor", "alternative"]
CATEGORY_COMPETITOR_SEEDS = [
    "BetterSleep", "Headspace", "Calm", "Sleepio", "Sleep Reset", "Somryst",
    "Oura", "Whoop", "CBT-I", "magnesium glycinate",
    "Sleep Cycle", "Pzizz", "Endel",
]
COMPETITOR_URLS = {
    "BetterSleep": "https://www.bettersleep.com",
    "Headspace": "https://www.headspace.com",
    "Calm": "https://www.calm.com",
    "Sleepio": "https://www.sleepio.com",
    "Sleep Reset": "https://www.sleepreset.com",
    "Somryst": "https://www.somryst.com",
    "Oura": "https://ouraring.com",
    "Whoop": "https://www.whoop.com",
    "Sleep Cycle": "https://www.sleepcycle.com",
    "Pzizz": "https://pzizz.com",
    "Endel": "https://endel.io",
    "CBT-I": "https://www.sleepfoundation.org/sleep-news/cbt-i",
}
# Device/tracker words that often appear in shopper questions as symptoms, not rivals.
QUERY_CONTEXT_NOT_COMPETITORS = {
    "fitbit", "apple watch", "garmin", "whoop", "oura", "pixel watch", "samsung watch",
}

MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "google_ai": "Google AI",
    "perplexity": "Perplexity",
    "gemini": "Gemini",
    "copilot": "Copilot",
    "claude": "Claude",
    "grok": "Grok",
}
DEFAULT_REPORT_MODELS = ["chatgpt", "google_ai", "perplexity", "gemini", "copilot", "claude", "grok"]
MODEL_PERSONAS = {
    "chatgpt": "Respond as ChatGPT (OpenAI): clear, structured tips, conversational.",
    "google_ai": "Respond as Google AI Overviews: concise bullets, practical web-style advice.",
    "perplexity": "Respond as Perplexity: direct, citation-minded, name specific products.",
    "gemini": "Respond as Google Gemini: helpful, stepwise, product-aware.",
    "copilot": "Respond as Microsoft Copilot: practical, Windows/Bing-shopper tone.",
    "claude": "Respond as Claude: careful, nuanced, still name real products when useful.",
    "grok": "Respond as Grok: candid consumer advice, name real apps/products when relevant.",
}
# Soft product lean so simulated models diverge (only use if relevant to the question).
MODEL_PRODUCT_LEAN = {
    "chatgpt": ["Headspace", "Sleep Cycle"],
    "google_ai": ["Calm", "Sleepio"],
    "perplexity": ["BetterSleep", "Sleep Reset"],
    "gemini": ["Endel", "Pzizz"],
    "copilot": ["Somryst", "CBT-I"],
    "claude": ["Sleepio", "Somryst"],
    "grok": ["Pzizz", "Endel"],
}

# Question-gen target label (shopper model in the JSON prompt — not the Bedrock model id).
PULSE_QUESTION_MODEL = "ChatGPT"
PULSE_QUESTION_PROMPT = """You generate an AI-shopper pulse for one product URL and one target model.

Inputs:
- URL: {{URL}}
- MODEL: {{MODEL}}   // e.g. ChatGPT, Perplexity, Google AI Overviews, Gemini, Copilot, Claude, Grok

Job:
1. Read the page at URL.
2. Extract: product job, what it is not, price/time if shown, brand name, likely competitors.
3. Write 5 unbranded shopper questions for MODEL.
4. State what MODEL should answer if it is in shopping mode.

Question rules:
- 4–8 words.
- No brand name. No domain. No product name.
- Sound like a shopper typing into MODEL.
- Ask for a tool / product / fix to buy or use, not a tips lecture.
- Max 2 questions start with How.
- How is the only starter you may repeat.
- Other starters must all differ (Best, Recommend, What’s, Cheap, Why…).
- Ban generic advice questions such as “How do I improve my sleep quality?”
- Each question must match the product’s unique job.
- Questions must be specific enough that MODEL can name a product or URL.

What MODEL should answer:
When MODEL is asked each question, a valid shopping answer MUST do one of these:
- Name the brand or URL from the input page. That is YOU.
- Name a real competing product and its URL. That is COMPETITOR.
- If MODEL only gives tips, habits, medical generalities, or “talk to a doctor” with no product, that is NEITHER.

MODEL must not be treated as correct if it only lectures.
For this pulse, MODEL is a shopper assistant. It should pick something.

Expected answer shape for MODEL:
- 1 recommended product name
- 1 URL
- 1 line why it fits the question
- optional 1–2 runner-up products
Do not invent the input brand if it is unknown. Then recommend a real competitor instead of saying nothing.

Scoring labels MODEL’s answer must be mapped to:
- YOU = names the input brand or URL
- COMPETITOR = names another product/URL
- NEITHER = no product and no URL

Output JSON only:
{
  "url": "{{URL}}",
  "model": "{{MODEL}}",
  "product_job": "",
  "not_this": [],
  "likely_competitors": [],
  "questions": [
    {
      "id": 1,
      "question": "",
      "intent": "buy",
      "expected_answer": {
        "must_include": "a product name and a URL",
        "you_if": "names this page’s brand or URL",
        "competitor_if": "names another product or URL",
        "neither_if": "tips only, no product"
      }
    }
  ]
}
"""
STATUS_TO_LABEL = {
    "named": "YOU",
    "competitor": "COMPETITOR",
    "ignored": "NEITHER",
}

WHAT_TO_DO_NEXT_PROMPT = """You write the “Verdict” block for a Site Pulse Audit.

Inputs:
- URL: {{URL}}
- Brand: {{BRAND}}
- Product job: {{PRODUCT_JOB}}
- You recommended: {{YOU_COUNT}}/{{TOTAL}}
- Highest competitor (context only — do not print a Highest line): {{COMPETITOR_COUNT}}/{{TOTAL}} {{COMPETITOR_URL}}
- Questions used: {{QUESTIONS}}
- Models checked: {{MODELS}}

Write advice a founder can do this week. No GEO jargon. No “get into sources models already cite.”

Rules:
- If YOU_COUNT is 0, say they are invisible to these models.
- You may name a competing product in an action if useful, but do NOT output a Highest: counter line.
- Give 3 numbered actions only.
- Each action must be a concrete step: edit the page, index it, or get one outside mention.
- Use the real shopper questions from the pulse. Tell them to put those exact lines on the page.
- Do not promise rankings or traffic.
- Do not mention medical claims.
- Max 80 words after the counters.
- Do NOT add a “Re-run the pulse” line.
- Do NOT output any Highest: line.

Output exactly:

You: {{YOU_COUNT}}/{{TOTAL}}

1. ...
2. ...
3. ...
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sleepfix_pack() -> dict[str, Any]:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


def _load_pack() -> dict[str, Any]:
    """Back-compat: session pack if present, else SleepFix mock pack."""
    return _load_active_pack() or _load_sleepfix_pack()


def _load_fixtures() -> dict[str, str]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _load_active_pack() -> dict[str, Any] | None:
    if not ACTIVE_PACK_PATH.exists():
        return None
    try:
        data = json.loads(ACTIVE_PACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_active_pack(pack: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PACK_PATH.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return ACTIVE_PACK_PATH


def _pack_for_backend(backend: str | None) -> dict[str, Any]:
    if (backend or "").lower() == "mock":
        return _load_sleepfix_pack()
    return _load_active_pack() or _load_sleepfix_pack()


def _norm(text: str) -> str:
    return text.lower()


def _competitor_mentioned(answer: str, competitor: str) -> bool:
    """True if competitor name appears as a whole phrase (not a substring of another word)."""
    if not answer or not competitor:
        return False
    # Escape for regex; allow flexible whitespace/hyphen inside multi-word names.
    parts = [re.escape(p) for p in re.split(r"\s+", competitor.strip()) if p]
    if not parts:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s\-]+".join(parts) + r"(?![A-Za-z0-9])"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


def _competitor_display(name: str, url: str | None = None) -> str:
    url = (url or COMPETITOR_URLS.get(name) or "").strip()
    if url:
        return f"{name} ({url})"
    return name


def _ensure_url(url: str) -> str:
    raw = (url or "").strip()
    if raw and "://" not in raw:
        return "https://" + raw
    return raw


def hostname_brand(product_url: str) -> str:
    """First DNS label of the product URL host (e.g. shop.example.com → shop)."""
    return _domain_label(product_url) or "brand"


def _domain_label(product_url: str) -> str:
    host = (urlparse(_ensure_url(product_url)).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    return host.split(".")[0]


def _clean_text(value: str, limit: int = 400) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


class _PageMeta(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.h1 = ""
        self._in_title = False
        self._in_h1 = False
        self._h1_done = False
        self._title_parts: list[str] = []
        self._h1_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {(k or "").lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "h1" and not self._h1_done:
            self._in_h1 = True
        elif tag == "meta":
            name = d.get("name", "").lower()
            prop = d.get("property", "").lower()
            if name == "description" or prop in {"og:description", "twitter:description"}:
                content = d.get("content", "").strip()
                if content and not self.description:
                    self.description = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            if not self.title:
                self.title = "".join(self._title_parts)
        elif tag == "h1" and self._in_h1:
            self._in_h1 = False
            self._h1_done = True
            if not self.h1:
                self.h1 = "".join(self._h1_parts)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_h1:
            self._h1_parts.append(data)


def _parse_page(html_text: str) -> dict[str, str]:
    title = description = h1 = ""
    if html_text:
        parser = _PageMeta()
        try:
            parser.feed(html_text)
            title = parser.title
            description = parser.description
            h1 = parser.h1
        except Exception:  # noqa: BLE001 — tolerate broken markup
            pass
        if not title:
            m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
            if m:
                title = re.sub(r"<[^>]+>", "", m.group(1))
        if not description:
            m = re.search(
                r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']',
                html_text,
                re.I | re.S,
            )
            if m:
                description = m.group(1)
        if not h1:
            m = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, re.I | re.S)
            if m:
                h1 = re.sub(r"<[^>]+>", "", m.group(1))
    # Visible body text helps shopper-question JTBD (beyond title hooks like "Fitbit…").
    body = html_text or ""
    body = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", body)
    body = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", body)
    body = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    return {
        "title": _clean_text(title),
        "description": _clean_text(description),
        "h1": _clean_text(h1),
        "text": _clean_text(body, limit=1200),
    }


def _fetch_page(url: str) -> str:
    target = _ensure_url(url)
    if not target:
        return ""
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SEC, context=ctx) as resp:
            raw = resp.read()
            charset = "utf-8"
            try:
                charset = resp.headers.get_content_charset() or "utf-8"
            except Exception:
                charset = "utf-8"
    except (urllib.error.URLError, TimeoutError, ssl.SSLError, ValueError, OSError):
        return ""
    except Exception:  # noqa: BLE001 — ignore all fetch errors
        return ""
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _bedrock_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _bedrock_model_id() -> str:
    return (os.environ.get("LLMCITE_BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL).strip()


def _text_from_converse(resp: dict[str, Any]) -> str:
    content = (((resp or {}).get("output") or {}).get("message") or {}).get("content") or []
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Empty Bedrock converse response")
    return text


def _text_from_invoke(resp: Any) -> str:
    raw = resp.get("body") if isinstance(resp, dict) else None
    if hasattr(raw, "read"):
        raw = raw.read()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bedrock invoke_model returned non-JSON body: {raw[:300]}") from exc
    elif isinstance(raw, dict):
        data = raw
    else:
        raise RuntimeError("Bedrock invoke_model returned an empty body")

    # Nova messages-v1
    nova = (((data.get("output") or {}).get("message") or {}).get("content") or [])
    nova_parts = [b.get("text", "") for b in nova if isinstance(b, dict)]
    if any(nova_parts):
        return "\n".join(p for p in nova_parts if p).strip()
    # Claude
    claude = data.get("content") or []
    if isinstance(claude, list):
        claude_parts = [b.get("text", "") for b in claude if isinstance(b, dict)]
        if any(claude_parts):
            return "\n".join(p for p in claude_parts if p).strip()
    # Titan
    results = data.get("results") or []
    if results and isinstance(results[0], dict) and results[0].get("outputText"):
        return str(results[0]["outputText"]).strip()
    if data.get("completion"):
        return str(data["completion"]).strip()
    if data.get("generation"):
        return str(data["generation"]).strip()
    raise RuntimeError(f"Unrecognized Bedrock invoke_model payload keys: {list(data)[:12]}")


def _invoke_body(model_id: str, prompt: str, system: str | None, max_tokens: int) -> dict[str, Any]:
    mid = model_id.lower()
    if "anthropic" in mid or "claude" in mid:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        return body
    if "titan" in mid:
        text = f"{system}\n\n{prompt}" if system else prompt
        return {
            "inputText": text,
            "textGenerationConfig": {"maxTokenCount": int(max_tokens)},
        }
    body = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": int(max_tokens)},
    }
    if system:
        body["system"] = [{"text": system}]
    return body


def _bedrock_text(prompt: str, system: str | None = None, max_tokens: int = 800, temperature: float | None = None) -> str:
    """Call Bedrock Runtime. Prefer Converse; fall back to InvokeModel."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for live Bedrock calls. Install boto3 or use backend=mock."
        ) from exc

    model_id = _bedrock_model_id()
    region = _bedrock_region()
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not create Bedrock Runtime client in {region}: {exc}") from exc

    inference: dict[str, Any] = {"maxTokens": int(max_tokens)}
    if temperature is not None:
        inference["temperature"] = float(temperature)
    converse_kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": inference,
    }
    if system:
        converse_kwargs["system"] = [{"text": system}]

    converse_error: Exception | None = None
    try:
        resp = client.converse(**converse_kwargs)
        return _text_from_converse(resp)
    except Exception as exc:  # noqa: BLE001
        converse_error = exc

    try:
        body = _invoke_body(model_id, prompt, system, max_tokens)
        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        return _text_from_invoke(resp)
    except Exception as invoke_error:  # noqa: BLE001
        raise RuntimeError(
            f"Bedrock failed for model {model_id} in {region}. "
            f"converse: {converse_error}; invoke_model: {invoke_error}. "
            "Grant bedrock:InvokeModel (and model access) on the AgentCore/runtime role."
        ) from invoke_error


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("model did not return a JSON object")


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("name") or item.get("query") or item.get("brand")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return out


def _template_queries(brand: str) -> list[str]:
    brand = (brand or "Brand").strip() or "Brand"
    return [q.format(brand=brand) for q in TEMPLATE_QUERIES]


def _five_word_product_line(one_liner: str, brand: str, page_text: str = "") -> str:
    """~5-word product description from the customer URL copy (not the brand alone)."""
    brand_l = (brand or "").lower().strip()
    blob = " ".join(
        x for x in (_clean_text(one_liner or "", limit=240), _clean_text(page_text or "", limit=700)) if x
    )
    # Prefer "what is …" style sentence if present
    m = re.search(
        r"(?:what is (?:a |an )?[^?]{0,40}\?\s*)(.{20,160}?)(?:\.|\n|$)",
        blob,
        flags=re.I,
    )
    if m:
        blob = m.group(1) + " " + blob
    stop = {
        "the", "a", "an", "and", "or", "for", "with", "your", "you", "to", "of", "in", "on",
        "is", "are", "was", "were", "be", "been", "being", "this", "that", "it", "do", "does",
        "need", "download", "anything", "get", "my",
    }
    words: list[str] = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", blob):
        wl = w.lower()
        if brand_l and wl == brand_l:
            continue
        if wl in stop:
            continue
        # skip complaint/marketing noise from SleepFix H1
        if wl in {"fitbit", "tired", "hours", "low", "wake", "waking"}:
            continue
        words.append(w)
        if len(words) >= 8:
            break
    # Prefer product-defining tokens if present
    prefer = []
    for w in words:
        wl = w.lower()
        if wl in {"58", "second", "seconds", "sleepcoachgame", "habit", "fix", "personalized", "browser", "game", "scenarios", "deep", "sleep"}:
            prefer.append(w)
    picked = prefer[:5] if len(prefer) >= 4 else words[:5]
    if len(picked) >= 3:
        return " ".join(picked[:5])
    return " ".join(picked) if picked else "product that solves the problem"


def _sleepfix_pulse_queries(brand: str) -> list[str]:
    """Exact 8-slot pulse from SleepFix URL + Jasmine's Queries run template."""
    b = (brand or "SleepFix").strip() or "SleepFix"
    return [
        "How do I raise Fitbit deep sleep?",
        "Best fix for waking tired after 8 hours",
        "What helps 3am waking?",
        "Deep sleep habit that actually works",
        f"{b} review",
        "58 second deep sleep habit game",
        b,
        "Find what's stealing my deep sleep",
    ]


def _ensure_query_shape(slot: str, text: str, brand: str) -> str:
    q = _clean_text(text or "", limit=120).rstrip("?.!")
    if not q:
        q = ""
    slot = slot.lower()
    if slot == "how_do_i":
        if not q.lower().startswith("how do i"):
            q = f"How do I {q}" if q else "How do I fix this problem"
        return q[0].upper() + q[1:] + ("?" if not q.endswith("?") else "")
    if slot == "best":
        if not q.lower().startswith("best"):
            q = f"Best {q}" if q else "Best fix for this problem"
        return q[0].upper() + q[1:]
    if slot == "what_helps":
        if not q.lower().startswith("what helps"):
            q = f"What helps {q}" if q else "What helps with this problem"
        return q[0].upper() + q[1:] + ("?" if not q.endswith("?") else "")
    if slot == "actually_works":
        if "that actually works" not in q.lower():
            q = f"{q} that actually works" if q else "habit that actually works"
        return q[0].upper() + q[1:]
    if slot == "find":
        if not q.lower().startswith("find"):
            q = f"Find {q}" if q else "Find a fix that sticks"
        return q[0].upper() + q[1:]
    return q



def _is_device_shopping_query(q: str) -> bool:
    """True if the shopper ask is about buying/finding a sleep device/wearable."""
    ql = (q or "").lower()
    if not ql:
        return False
    bans = (
        "find a device",
        "find a wearable",
        "find a tracker",
        "find me a device",
        "find me a wearable",
        "find me a tracker",
        "device that enhances",
        "device that improves",
        "device that boosts",
        "wearable that enhances",
        "wearable that improves",
        "wearable technology",
        "with wearable",
        "using a wearable",
        "using wearable",
        "sleep tracking device",
        "tracking device",
        "best sleep tracker",
        "which sleep tracker",
        "best wearable",
        "which wearable",
        "recommend a device",
        "recommend a tracker",
        "recommend a wearable",
        "buy a fitbit",
        "buy an apple watch",
        "buy a sleep tracker",
        "fitbit vs",
        "apple watch vs",
        "compare fitbit",
        "monitor sleep patterns with a device",
        "device to monitor",
        "device to track",
        "gadget that",
        "smartwatch for sleep",
        "ring that tracks",
        "oura ring",
        "whoop strap",
    )
    if any(b in ql for b in bans):
        return True
    # "find … device/wearable/tracker/gadget"
    if re.search(r"\bfind\b.{0,40}\b(device|wearable|tracker|gadget|smartwatch|ring)\b", ql):
        return True
    if re.search(r"\b(device|wearable|tracker|gadget)\b.{0,40}\b(enhance|improves?|boost|quality|monitor|track)\b", ql):
        # allow "fitbit deep sleep" symptom phrasing — only ban when device is the solution object
        if re.search(r"\b(how do i|what helps|raise|fix|habit|tired|wake|waking|3am)\b", ql) and re.search(
            r"\b(fitbit|apple watch)\b.{0,30}\b(deep sleep|sleep)\b", ql
        ):
            return False
        return True
    return False



def _why_cant_i_sleep_shopper_questions() -> list[str]:
    """Curated shopper asks for why.longevitygreenlight.com (Why can't I sleep)."""
    return [
        "How do I improve my sleep quality?",
        "Best solution for waking up refreshed",
        "What helps with restless sleep?",
        "Sleep hygiene tips that actually works",
        "Why can't I sleep review",
        "58 second bedtime habit game",
        "Why can't I sleep",
        "Find what's stealing my sleep tonight",
    ]


def _is_why_cant_i_sleep_product(brand: str, product_url: str, one_liner: str = "") -> bool:
    blob = " ".join(
        [
            (brand or "").lower(),
            (product_url or "").lower(),
            (one_liner or "").lower(),
        ]
    )
    return (
        "why.longevitygreenlight.com" in blob
        or "why can't i sleep" in blob
        or "why cant i sleep" in blob
    )


def _sleepfix_shopper_questions() -> list[str]:
    """Curated shopper asks grounded in the SleepFix page (habit/symptom, not devices)."""
    return [
        "how do I raise Fitbit deep sleep",
        "Apple Watch deep sleep low what do I do",
        "why am I still tired after 8 hours of sleep",
        "how do I stop waking up at 3am",
        "what habit is stealing my deep sleep",
        "how do I find what's causing low deep sleep",
    ]


def _is_sleepfix_product(brand: str, product_url: str, one_liner: str = "") -> bool:
    blob = " ".join(
        [
            (brand or "").lower(),
            (product_url or "").lower(),
            (one_liner or "").lower(),
        ]
    )
    return "sleepfix" in blob or "sleepcoach.longevitygreenlight.com" in blob


def _parse_numbered_shopper_questions(text: str) -> list[str]:
    """Parse Bedrock numbered 1–5 lines into question strings."""
    out: list[str] = []
    for m in re.finditer(r"^\s*[1-5][\.\)]\s*(.+?)\s*$", text or "", flags=re.M):
        q = m.group(1).strip().strip('"').strip("'")
        if q:
            out.append(q)
    return out[:5]


def _question_leaks_brand_or_domain(q: str, brand: str, product_url: str) -> bool:
    ql = (q or "").lower()
    brand_l = (brand or "").lower().strip()
    if brand_l and brand_l in ql:
        return True
    domain = _domain_label(product_url).lower() if product_url else ""
    if domain and domain in ql:
        return True
    if product_url:
        url_l = product_url.lower().rstrip("/")
        if url_l and url_l in ql:
            return True
        # bare hostname without TLD noise
        try:
            host = re.sub(r"^https?://", "", product_url.lower()).split("/")[0]
            host = host.split(":")[0]
            if host and host in ql:
                return True
        except Exception:
            pass
    return False


def _fallback_shopper_questions_from_page(page: dict[str, str], brand: str = "") -> list[str]:
    """5 short discovery questions from page h1/description — never SleepFix defaults."""
    h1 = _clean_text(page.get("h1") or "", limit=80)
    desc = _clean_text(page.get("description") or page.get("title") or "", limit=100)
    blob = h1 or desc or "this product"
    brand_l = (brand or "").lower().strip()
    words = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z\-']*", blob)
        if w.lower() != brand_l
        and w.lower()
        not in {
            "the", "a", "an", "and", "or", "for", "with", "your", "you", "to", "of",
            "in", "on", "is", "are", "this", "that", "it", "our", "we",
        }
    ]
    core = " ".join(words[:3]) if words else "this problem"
    topic = words[0] if words else "this"
    return [
        f"Best {core}",
        f"Recommend {core}",
        f"What's best for {topic}",
        f"How to fix {core}",
        f"Cheap {core} option",
    ]


def _page_snapshot(page: dict[str, str]) -> str:
    """Short title/h1/description/excerpt so Bedrock does not need a live web fetch."""
    title = _clean_text((page or {}).get("title") or "", limit=120)
    h1 = _clean_text((page or {}).get("h1") or "", limit=120)
    description = _clean_text((page or {}).get("description") or "", limit=220)
    excerpt = _clean_text((page or {}).get("text") or "", limit=500)
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if h1:
        parts.append(f"H1: {h1}")
    if description:
        parts.append(f"Description: {description}")
    if excerpt:
        parts.append(f"Text excerpt: {excerpt}")
    return "\n".join(parts) if parts else "(no page text available)"


def _pulse_question_user_message(url: str, model: str, snapshot: str) -> str:
    """Bedrock user message: Jasmine's JSON pulse prompt + already-fetched page snapshot."""
    prompt = (
        PULSE_QUESTION_PROMPT.replace("{{URL}}", url or "").replace("{{MODEL}}", model or PULSE_QUESTION_MODEL)
    )
    return (
        prompt
        + "\n\nPage snapshot (already fetched in code — do not browse the web):\n"
        + (snapshot or "(no page text available)")
    )


def _questions_from_pulse_json(obj: dict[str, Any], brand: str, product_url: str) -> list[str]:
    raw = (obj or {}).get("questions") or []
    out: list[str] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        q = ""
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict):
            q = str(item.get("question") or "").strip()
        if not q:
            continue
        if _question_leaks_brand_or_domain(q, brand, product_url):
            continue
        if _is_device_shopping_query(q):
            continue
        if q not in out:
            out.append(q)
        if len(out) >= 5:
            break
    return out


def _competitors_from_pulse_json(
    obj: dict[str, Any] | None, brand: str = "", product_url: str = ""
) -> list[str]:
    if not obj:
        return []
    raw = _as_str_list(obj.get("likely_competitors"))
    brand_l = (brand or "").lower().strip()
    domain = _domain_label(product_url).lower() if product_url else ""
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        key = _norm(c)
        if not key or key in seen:
            continue
        if key in {"competitor", "alternative"}:
            continue
        if brand_l and key == brand_l:
            continue
        if domain and key == domain:
            continue
        seen.add(key)
        out.append(c)
    return out[:12]


def _generate_pulse_json(
    product_url: str,
    page: dict[str, str],
    model: str = PULSE_QUESTION_MODEL,
) -> dict[str, Any] | None:
    """One Bedrock call per pack. MODEL label is ChatGPT; parse via _extract_json_object."""
    snapshot = _page_snapshot(page)
    user_prompt = _pulse_question_user_message(product_url, model, snapshot)
    try:
        text_out = _bedrock_text(
            user_prompt,
            system=(
                "Output JSON only matching the schema. Exactly 5 unbranded shopper questions. "
                "No markdown fences, no preamble, no brand or domain names. "
                "Do not fetch the URL; use the provided page snapshot."
            ),
            max_tokens=1400,
        )
        obj = _extract_json_object(text_out)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _top_up_shopper_questions(
    questions: list[str], page: dict[str, str], brand: str, product_url: str
) -> list[str]:
    cleaned = list(questions)
    for fb in _fallback_shopper_questions_from_page(page, brand):
        if len(cleaned) >= 5:
            break
        if _question_leaks_brand_or_domain(fb, brand, product_url):
            continue
        if fb not in cleaned:
            cleaned.append(fb)
    if cleaned and len(cleaned) < 5:
        fallbacks = _fallback_shopper_questions_from_page(page, brand)
        while len(cleaned) < 5 and fallbacks:
            cleaned.append(fallbacks[len(cleaned) % 5])
    if len(cleaned) >= 5:
        return cleaned[:5]
    return _fallback_shopper_questions_from_page(page, brand)[:5]


def generate_problem_shopper_pulse(
    brand: str,
    product_url: str,
    page: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    """Once-per-pack question gen. Returns (5 questions, likely_competitors, pulse JSON).

    SleepFix / Why-can't-I-sleep keep curated 5-question overrides (no Bedrock JSON).
    """
    brand = (brand or "").strip() or hostname_brand(product_url)
    product_url = _ensure_url(product_url)
    if page is None:
        page = (
            _parse_page(_fetch_page(product_url))
            if product_url
            else {"title": "", "description": "", "h1": "", "text": ""}
        )
    one_liner = page.get("description") or page.get("title") or page.get("h1") or ""
    if _is_why_cant_i_sleep_product(brand, product_url, one_liner):
        return _why_cant_i_sleep_shopper_questions()[:5], [], None
    if _is_sleepfix_product(brand, product_url, one_liner):
        return _sleepfix_shopper_questions()[:5], [], None

    pulse = _generate_pulse_json(product_url, page, model=PULSE_QUESTION_MODEL)
    questions: list[str] = []
    competitors: list[str] = []
    if pulse:
        questions = _questions_from_pulse_json(pulse, brand, product_url)
        competitors = _competitors_from_pulse_json(pulse, brand, product_url)
    questions = _top_up_shopper_questions(questions, page, brand, product_url)
    return questions[:5], competitors, pulse


def generate_problem_shopper_questions(
    brand: str,
    product_url: str,
    page: dict[str, str] | None = None,
) -> list[str]:
    """Shopper discovery questions BEFORE they know this brand exists.

    Uses Jasmine's JSON Bedrock pulse prompt (MODEL=ChatGPT) + page snapshot.
    SleepFix / Why-can't-I-sleep keep curated overrides. Never falls back to
    SleepFix questions for non-sleep URLs.
    """
    questions, _competitors, _pulse = generate_problem_shopper_pulse(brand, product_url, page=page)
    return questions


def _generate_queries_and_competitors(
    brand: str, product_url: str, one_liner: str, page_text: str = ""
) -> tuple[list[str], list[str]]:
    """Build the fixed 8-slot pulse query set + competitors.

    Slots:
      1 How do I …
      2 Best …
      3 What helps …
      4 … that actually works
      5 {brand} review
      6 ~5-word product description from the page
      7 brand name
      8 Find …
    """
    brand = (brand or "").strip() or hostname_brand(product_url)
    five = _five_word_product_line(one_liner, brand, page_text)
    prompt = (
        "You map shopper LLM prompts for a product category.\n"
        f"Brand (secret for slots 1-4 and 8): {brand}\n"
        f"URL: {product_url}\n"
        f"Description: {one_liner}\n\n"
        "Return ONLY valid JSON with:\n"
        '- "how_do_i": one question starting with \"How do I\" (problem/symptom; may mention '
        "category devices like Fitbit when relevant; do NOT name the brand)\n"
        '- "best": one prompt starting with \"Best\" (fix/outcome, no brand)\n'
        '- "what_helps": one question starting with \"What helps\" (no brand)\n'
        '- "actually_works": one phrase ending with \"that actually works\" (no brand)\n'
        '- "find": one prompt starting with \"Find\" about a HABIT or CAUSE of the problem (no brand; NEVER a device/wearable/tracker)\n'
        '- "competitors": 5-8 competitor brand or product names\n\n'
        "GOOD shape examples (adapt to THIS page):\n"
        '- how_do_i: \"How do I raise Fitbit deep sleep?\"\n'
        '- best: \"Best fix for waking tired after 8 hours\"\n'
        '- what_helps: \"What helps 3am waking?\"\n'
        '- actually_works: \"Deep sleep habit that actually works\"\n'
        '- find: \"Find a deep sleep habit that sticks\"\n        "HARD BAN: never Find a device/wearable/tracker that enhances sleep.\n'
    )
    slots = {
        "how_do_i": "",
        "best": "",
        "what_helps": "",
        "actually_works": "",
        "find": "",
    }
    competitors: list[str] = []
    try:
        text = _bedrock_text(
            prompt,
            system=(
                "Return only JSON. Slots 1-4 and Find must not name the brand. "
                "Allow Fitbit/Apple Watch only as symptoms. Ban Find/buy device or wearable asks."
            ),
            max_tokens=800,
        )
        obj = _extract_json_object(text)
        for key in slots:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                slots[key] = val.strip()
        competitors = _as_str_list(obj.get("competitors"))[:8]
    except Exception:
        pass

    # Fallbacks shaped like Jasmine's SleepFix examples when Bedrock blanks a slot
    defaults = {
        "how_do_i": "How do I raise Fitbit deep sleep?",
        "best": "Best fix for waking tired after 8 hours",
        "what_helps": "What helps 3am waking?",
        "actually_works": "Deep sleep habit that actually works",
        "find": "Find a deep sleep habit that sticks",
    }
    if _is_sleepfix_product(brand, product_url, one_liner):
        queries = _sleepfix_pulse_queries(brand)
    else:
        for key in list(slots.keys()):
            if _is_device_shopping_query(slots[key]):
                slots[key] = ""
        queries = [
            _ensure_query_shape("how_do_i", slots["how_do_i"] or defaults["how_do_i"], brand),
            _ensure_query_shape("best", slots["best"] or defaults["best"], brand),
            _ensure_query_shape("what_helps", slots["what_helps"] or defaults["what_helps"], brand),
            _ensure_query_shape(
                "actually_works", slots["actually_works"] or defaults["actually_works"], brand
            ),
            f"{brand} review",
            five,
            brand,
            _ensure_query_shape("find", slots["find"] or defaults["find"], brand),
        ]
        for i, key in enumerate(("how_do_i", "best", "what_helps", "actually_works", None, None, None, "find")):
            if key and _is_device_shopping_query(queries[i]):
                queries[i] = _ensure_query_shape(key, defaults[key], brand)
    merged: list[str] = []
    seen: set[str] = set()
    # Sleep competitor seeds only for SleepFix; other products use page/LLM competitors only.
    seed_competitors = (
        CATEGORY_COMPETITOR_SEEDS
        if _is_sleepfix_product(brand, product_url, one_liner)
        else []
    )
    for c in list(competitors) + list(seed_competitors):
        key = _norm(c)
        if not key or key in seen:
            continue
        # skip useless placeholders
        if key in {"competitor", "alternative"}:
            continue
        seen.add(key)
        merged.append(c)
    if not merged:
        merged = [c for c in FALLBACK_COMPETITORS if c]
    return queries, merged[:12]


def build_pack(brand: str, product_url: str) -> dict[str, Any]:
    """Fetch the product page and build a session query pack for that brand/URL."""
    brand = (brand or "").strip() or hostname_brand(product_url)
    product_url = _ensure_url(product_url)
    if not product_url:
        raise ValueError("product_url is required to build a live pack")

    page = _parse_page(_fetch_page(product_url))
    one_liner = page["description"] or page["title"] or page["h1"] or f"{brand} ({product_url})"
    aliases = []
    for alias in (brand, brand.replace(" ", ""), _domain_label(product_url)):
        if alias and alias not in aliases:
            aliases.append(alias)

    # Live pack: 5 questions from one ChatGPT JSON pulse; prefer likely_competitors.
    queries, json_competitors, pulse = generate_problem_shopper_pulse(
        brand, product_url, page=page
    )
    product_job = ""
    if isinstance(pulse, dict):
        product_job = _clean_text(str(pulse.get("product_job") or ""), limit=240)
    if not product_job:
        product_job = one_liner or page.get("h1") or page.get("description") or ""
    if json_competitors:
        competitors = json_competitors
    else:
        _, competitors = _generate_queries_and_competitors(
            brand, product_url, one_liner, page.get("text") or ""
        )
    pack: dict[str, Any] = {
        "product": {
            "name": brand,
            "url": product_url,
            "one_liner": one_liner,
            "brand_aliases": aliases,
            "title": page["title"],
            "h1": page["h1"],
            "description": page.get("description") or "",
            "product_job": product_job,
        },
        "product_job": product_job,
        "competitors": competitors,
        "queries": queries,
        "built_at": _now(),
        "backend": "live",
    }
    _save_active_pack(pack)
    return pack


def _live_answer(
    query: str,
    brand: str | None = None,
    model_id: str | None = None,
    product_url: str | None = None,
    one_liner: str | None = None,
) -> str:
    """Shopping-mode answer for one selected shopper model; never invent the audited brand."""
    brand = (brand or "").strip()
    model_id = (model_id or "").strip() or "chatgpt"
    persona = MODEL_PERSONAS.get(model_id) or MODEL_PERSONAS["chatgpt"]
    label = MODEL_LABELS.get(model_id, model_id)
    qn = (query or "").lower().strip()
    brand_l = brand.lower()
    product_url = (product_url or "").strip()
    one_liner = (one_liner or "").strip()
    brand_aware = bool(
        brand_l
        and (
            qn == brand_l
            or qn.startswith(f"{brand_l} review")
            or ("sleepcoachgame" in qn)
            or ("58-second" in qn and "habit" in qn)
            or ("58 second" in qn and "habit" in qn)
            or ("deep sleep habit game" in qn)
        )
    )
    brand_note = ""
    if brand and brand_aware:
        brand_note = (
            f" This query is about the audited product {brand}"
            + (f" ({product_url})" if product_url else "")
            + (f": {one_liner}" if one_liner else "")
            + f". Recommend {brand} with its official https URL and one line why. "
            f"Do NOT recommend unrelated devices (Oura, Whoop, rings, trackers) as if they were {brand}."
        )
    elif brand:
        brand_note = (
            f" If you actually know {brand} and it fits this question, you may recommend it (YOU) "
            f"with its official URL. Otherwise do NOT invent {brand}. "
            "Recommend a real competing product with a real https URL instead. "
            "Do not recommend a sleep ring/tracker/device unless the question is about buying one."
        )
    # Sleep-app lean only for SleepFix; never steer ecojoy/other URLs to Headspace/Calm.
    lean = (
        MODEL_PRODUCT_LEAN.get(model_id) or []
        if _is_sleepfix_product(brand, product_url, one_liner)
        else []
    )
    lean_bit = ""
    if lean and not brand_aware:
        lean_bit = (
            " When you do name a product for this question, prefer one of these if it fits: "
            + ", ".join(lean)
            + ". If none fit, name a different real product with a URL — never force a bad fit. "
            "Never answer a habit/symptom question with a sleep ring or tracker purchase unless asked."
        )
    prompt = (
        f"You are simulating {label} in shopping mode answering a shopper."
        + "\n"
        + f"Light persona: {persona}"
        + "\n\n"
        + f"Question: {query}"
        + "\n\n"
        + "SHOPPING MODE (mandatory — persona is light; these rules win):\n"
        + "- Recommend 1 product name, 1 official https:// URL, and 1 line why it fits.\n"
        + "- Optional: 1–2 runner-up products with URLs.\n"
        + "- You MUST pick a real product. Do not give a tips-only lecture, habit list, "
        + "medical generalities, or \"talk to a doctor\" with no product.\n"
        + "- Prefer a known fit (YOU if you actually know the audited brand). "
        + "If unknown, recommend a real competitor with a real URL — not nothing.\n"
        + "- Do not invent the audited brand if it is unknown.\n"
        + "- Put each https:// URL on its own line.\n"
        + "Be concise (70-140 words). No invented prices or medical claims."
        + lean_bit
        + brand_note
    )
    return _bedrock_text(
        prompt,
        system=(
            f"Simulate {label} in shopping mode. Persona is light; shopping-mode rules are mandatory. "
            "Always name 1 real product + 1 https URL + 1 line why it fits. Optional 1–2 runner-ups. "
            "Do not invent an audited brand you do not know — recommend a real competitor with a URL. "
            "Never return tips-only. Never invent that a ring or wearable is a 58-second habit game."
        ),
        max_tokens=360,
        temperature=0.85,
    )





def _score_answer(
    answer: str,
    brand_aliases: list[str],
    competitors: list[str],
    query: str | None = None,
    product_url: str | None = None,
) -> dict[str, Any]:
    a = _norm(answer)
    qn = _norm(query or "")
    named = any(_norm(alias) in a for alias in brand_aliases if alias)
    url_n = _norm(product_url or "")
    if url_n and url_n in a:
        named = True
    hit_competitors: list[str] = []
    seen: set[str] = set()
    for c in competitors:
        if not c:
            continue
        cn = _norm(c)
        if not cn or cn in seen:
            continue
        # Skip terms that are just the shopper's device/context already in the query
        # (e.g. Fitbit in "How do I raise Fitbit deep sleep?").
        if qn and cn in qn:
            continue
        if any(ctx in cn or cn in ctx for ctx in QUERY_CONTEXT_NOT_COMPETITORS if ctx in qn):
            continue
        if not _competitor_mentioned(answer, c):
            continue
        seen.add(cn)
        hit_competitors.append(_competitor_display(c))
    # Listing-aligned: non-product https links in the answer count as competitor signals
    product_key = (product_url or "").strip().rstrip("/").lower()
    answer_urls: list[str] = []
    seen_u: set[str] = set()
    for m in re.finditer(r"https?://[^\s<>\"')\]]+", answer or ""):
        u = m.group(0).strip().rstrip('.,;:)]>"\'')
        if not u.lower().startswith(("http://", "https://")):
            continue
        key = u.rstrip("/").lower()
        if not key or key in seen_u:
            continue
        if product_key and key == product_key:
            continue
        seen_u.add(key)
        answer_urls.append(u)

    if named:
        status = "named"
    elif hit_competitors:
        status = "competitor"
    elif answer_urls:
        status = "competitor"
        for u in answer_urls:
            if u not in hit_competitors:
                hit_competitors.append(u)
    else:
        status = "ignored"
    return {
        "status": status,
        "label": STATUS_TO_LABEL.get(status, "NEITHER"),
        "named": named,
        "competitors_mentioned": hit_competitors,
    }



def _url_leaf(url: str) -> str:
    try:
        path = urlparse(url or "").path or ""
        leaf = path.rstrip("/").split("/")[-1]
        if leaf:
            return leaf
        host = (urlparse(url or "").hostname or "").lower()
        return host or "product"
    except Exception:
        return "product"


def _load_selected_models() -> list[str]:
    path = OUTPUT_DIR / "selected_models.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            models = data if isinstance(data, list) else data.get("models") or []
            out = [str(m).strip() for m in models if str(m).strip()]
            if out:
                return out
        except (json.JSONDecodeError, OSError, AttributeError):
            pass
    return list(DEFAULT_REPORT_MODELS)


def _status_label(status: str, competitors: list[str]) -> str:
    st = (status or "ignored").lower()
    if st == "named":
        return "Named"
    if st == "competitor":
        bit = ", ".join(competitors[:2]) if competitors else "competitor"
        return f"Competitor ({bit})"
    return "Ignored"



def _build_verdict(
    brand: str,
    summary: dict[str, Any],
    page_cited: bool,
    product_url: str = "",
    one_liner: str = "",
) -> tuple[str, str]:
    named = int(summary.get("named") or 0)
    competitor = int(summary.get("competitor") or 0)
    sleepfix = _is_sleepfix_product(brand, product_url, one_liner)
    if named == 0 and not page_cited:
        if sleepfix:
            verdict = (
                f"AI treats this as a generic sleep problem. It does not know {brand} exists, "
                f"so it will not send the sale to that URL."
            )
            next_action = (
                f"get {brand} into sources models already cite—clear product definition, reviews, "
                f"and “deep sleep + Fitbit” pages—then re-run the pulse"
            )
        else:
            verdict = (
                f"AI treats this as a generic problem space. It does not know {brand} exists, "
                f"so it will not send the sale to that URL."
            )
            next_action = (
                f"get {brand} into sources models already cite—clear product definition, reviews, "
                f"and problem-led pages that match shopper questions—then re-run the pulse"
            )
    elif named == 0 and page_cited:
        verdict = (
            f"Your URL showed up in places, but models still did not recommend {brand} by name."
        )
        next_action = "Strengthen brand-name mentions on cited pages and re-run the pulse."
    elif named > 0 and competitor >= named:
        if sleepfix and named <= 3:
            verdict = (
                f"AI treats this as a generic sleep problem. It does not know {brand} exists, "
                f"so it will not send the sale to that URL."
            )
            next_action = (
                f"get {brand} into sources models already cite—clear product definition, reviews, "
                f"and “deep sleep + Fitbit” pages—then re-run the pulse"
            )
        else:
            verdict = (
                f"Some answers name {brand}, but competitors still win a similar or larger share of mentions."
            )
            next_action = (
                "Close the competitor gaps on the ignored/competitor queries, then re-run the pulse."
            )
    else:
        verdict = f"Models are starting to recommend {brand} for shopper questions in this category."
        next_action = "Keep citation pages fresh and re-run the pulse after major content changes."
    return verdict, next_action


def _parse_competitor_mentions(raw: str) -> list[tuple[str, str]]:
    """Split a competitor cell into (name, url) pairs."""
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[tuple[str, str]] = []
    # Prefer Name (https://...) chunks; also allow bare names separated by commas
    for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9 .+'\-]{0,60}?)\s*\((https?://[^)]+)\)", text):
        out.append((m.group(1).strip(), m.group(2).strip()))
    if out:
        return out
    for part in re.split(r"\s*,\s*", text):
        part = part.strip()
        if part:
            out.append((part, ""))
    return out


def _most_mentioned_competitor_stats(
    rows: list[dict[str, Any]],
) -> tuple[str | None, str | None, int]:
    """Most-mentioned competitor across scored answers → (name, url_or_None, hit_count).

    Counts every mention across every model×query row. Highest count wins.
    """
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    urls: dict[str, str] = {}
    for r in rows or []:
        for raw in r.get("competitors_mentioned") or []:
            for name, url in _parse_competitor_mentions(str(raw)):
                key = _norm(name)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
                names[key] = name
                if url:
                    urls[key] = url
                elif key not in urls:
                    for canon, cu in COMPETITOR_URLS.items():
                        if _norm(canon) == key:
                            urls[key] = cu
                            names[key] = canon
                            break
    if not counts:
        return None, None, 0
    # Strictly by count desc, then name asc for stable ties
    top_key = sorted(counts.keys(), key=lambda k: (-counts[k], k))[0]
    return names.get(top_key), urls.get(top_key), int(counts[top_key])


def _most_mentioned_competitor(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Most-mentioned competitor across scored answers → (name, url_or_None)."""
    name, url, _count = _most_mentioned_competitor_stats(rows)
    return name, url






def _generate_pulse_narrative(
    brand: str,
    product_url: str,
    one_liner: str,
    summary: dict[str, Any],
    top_comp: str | None,
    top_comp_url: str | None,
    all_rows: list[dict[str, Any]],
    page_cited: bool,
) -> tuple[str, str]:
    """Fresh Verdict + Next action for THIS pulse run (Bedrock), with template fallback."""
    named = int(summary.get("named") or 0)
    not_named = max(0, int(summary.get("total") or 0) - named)
    competitor = int(summary.get("competitor") or 0)
    gaps: list[str] = []
    for r in all_rows or []:
        if (r.get("status") or "") == "named":
            continue
        q = (r.get("query") or "").strip()
        comps = ", ".join((r.get("competitors_mentioned") or [])[:2])
        bit = q
        if comps:
            bit += f" → {comps}"
        if bit and bit not in gaps:
            gaps.append(bit)
        if len(gaps) >= 6:
            break
    fallback = _build_verdict(
        brand, summary, page_cited, product_url=product_url, one_liner=one_liner
    )
    sleepfix = _is_sleepfix_product(brand, product_url, one_liner)
    tone = ""
    if sleepfix:
        tone = (
            "This product is a 58-second SLEEPCOACHGAME habit fix for deep sleep "
            "(Fitbit/Apple Watch are symptoms, not the product). "
        )
    prompt = (
        "Write a brand-new Pulse closing for THIS audit run only.\n"
        f"Brand: {brand}\n"
        f"URL: {product_url}\n"
        f"Product: {one_liner}\n"
        f"{tone}"
        f"Stats for this run: named={named}, did_not_mention={not_named}, "
        f"competitor_status_rows={competitor}, page_url_cited={page_cited}, "
        f"most_mentioned_competitor={top_comp or '(none)'}"
        + (f" ({top_comp_url})" if top_comp_url else "")
        + ".\n"
        f"Gap examples: {gaps or ['(none)']}\n\n"
        'Return ONLY JSON: {"verdict": "...", "next_action": "..."}\n'
        "Rules:\n"
        "- verdict: 1-2 sentences, specific to THESE numbers and gaps (not generic filler).\n"
        "- next_action: 1 sentence, concrete, ends with re-run the pulse when useful.\n"
        "- Name the brand and, if useful, the top competitor.\n"
        "- Do not invent metrics. Do not mention Bedrock or that you are an AI writer.\n"
        "- Vary wording from run to run; do not reuse a stock paragraph.\n"
    )
    try:
        text = _bedrock_text(
            prompt,
            system=(
                "Return only JSON with verdict and next_action. "
                "Be specific to the provided stats. Fresh wording every time."
            ),
            max_tokens=280,
            temperature=0.9,
        )
        obj = _extract_json_object(text)
        verdict = _clean_text(str(obj.get("verdict") or ""), limit=400)
        next_action = _clean_text(str(obj.get("next_action") or ""), limit=400)
        if verdict and next_action:
            return verdict, next_action
    except Exception:
        pass
    return fallback


def _build_pulse_block(
    brand: str,
    product_url: str,
    one_liner: str,
    summary: dict[str, Any],
    all_rows: list[dict[str, Any]],
    page_cited: bool,
) -> dict[str, Any]:
    """Always rebuild Pulse from this run's scored rows (never reuse a prior pulse)."""
    top_comp, top_comp_url = _most_mentioned_competitor(all_rows)
    named_n = int(summary.get("named") or 0)
    total_n = int(summary.get("total") or len(all_rows) or 0)
    not_named_n = max(0, total_n - named_n)
    verdict, next_action = _generate_pulse_narrative(
        brand,
        product_url,
        one_liner,
        summary,
        top_comp,
        top_comp_url,
        all_rows,
        page_cited,
    )
    return {
        "most_mentioned_competitor": top_comp,
        "most_mentioned_competitor_url": top_comp_url,
        "named_mentions": named_n,
        "not_named_mentions": not_named_n,
        "verdict": verdict,
        "next_action": next_action,
        "generated_at": _now(),
    }


def _mention_cells(row: dict[str, Any]) -> tuple[str, str]:
    """Return (mentioned_you, mentioned_competitor) for a scored row."""
    st = (row.get("status") or "ignored").lower()
    comps = [c for c in (row.get("competitors_mentioned") or []) if c]
    you = "yes" if st == "named" else "no"
    return you, ", ".join(comps)



def _row_urls(row: dict[str, Any], product_url: str = "") -> list[str]:
    """URLs for one model×query row: only URLs that appear in the answer, or product when named.

    Shared by HTML listing and plain-text/PDF download so they cannot drift.
    Never invent competitor URLs via COMPETITOR_URLS name lookup — name-only
    mentions stay name-only in the listing.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(u: str) -> None:
        u = (u or "").strip().rstrip('.,;:)]>"\'')
        if not u or not u.lower().startswith(("http://", "https://")):
            return
        key = u.rstrip("/").lower()
        if key in seen:
            return
        seen.add(key)
        out.append(u)

    blob = " ".join(
        str(row.get(k) or "") for k in ("answer", "answer_excerpt")
    )
    for m in re.finditer(r"https?://[^\s<>\"')\]]+", blob):
        _add(m.group(0))
    if (row.get("status") or "").lower() == "named" and product_url:
        _add(product_url)
    return out


def _report_queries(report: dict[str, Any]) -> list[str]:
    """Stable shopper-query order matching the HTML listing."""
    rows = report.get("rows") or []
    models = report.get("models") or _load_selected_models() or list(DEFAULT_REPORT_MODELS)
    rows_by_model = report.get("rows_by_model") or {}
    queries: list[str] = []
    seen_q: set[str] = set()
    seed_rows = rows_by_model.get(models[0]) if models else None
    if seed_rows is None:
        seed_rows = rows
    for r in seed_rows or []:
        q = (r.get("query") or "").strip()
        if q and q not in seen_q:
            seen_q.add(q)
            queries.append(q)
    if not queries:
        for mid in models:
            for r in rows_by_model.get(mid) or []:
                q = (r.get("query") or "").strip()
                if q and q not in seen_q:
                    seen_q.add(q)
                    queries.append(q)
    return queries




def _verdict_listing_stats(report: dict[str, Any]) -> dict[str, Any]:
    """YOU / Highest counters aligned with listing URLs from `_row_urls`.

    Counts model×query rows the same way the HTML listing enumerates them:
    product URL (or named/YOU status) → you_count; other https links → highest.
    """
    product = report.get("product") or {}
    product_url = (product.get("url") or "").strip()
    product_key = product_url.rstrip("/").lower() if product_url else ""

    rows_by_model = report.get("rows_by_model") or {}
    if rows_by_model:
        all_rows: list[dict[str, Any]] = []
        for mid_rows in rows_by_model.values():
            all_rows.extend(mid_rows or [])
    else:
        all_rows = list(report.get("rows") or [])

    you_count = 0
    url_freq: dict[str, int] = {}
    url_display: dict[str, str] = {}

    for row in all_rows:
        urls = _row_urls(row, product_url=product_url)
        status = (row.get("status") or "").lower()
        row_keys = {u.rstrip("/").lower() for u in urls}
        you_hit = status in ("named", "you") or (
            bool(product_key) and product_key in row_keys
        )
        if you_hit:
            you_count += 1
        for u in urls:
            key = u.rstrip("/").lower()
            if not key or (product_key and key == product_key):
                continue
            url_freq[key] = url_freq.get(key, 0) + 1
            url_display[key] = u

    highest_url = ""
    highest_count = 0
    if url_freq:
        top_key = sorted(url_freq.keys(), key=lambda k: (-url_freq[k], k))[0]
        highest_count = int(url_freq[top_key])
        highest_url = url_display.get(top_key) or top_key

    return {
        "you_count": int(you_count),
        "total": len(all_rows),
        "highest_count": int(highest_count),
        "highest_url": highest_url,
    }


def _fallback_what_to_do_next(
    you_count: int,
    total: int,
    competitor_count: int,
    competitor_url: str = "",
) -> str:
    """Minimal compliant footer when Bedrock is unavailable."""
    return (
        f"You: {you_count}/{total}\n"
        "\n"
        "1. Edit your product page so the exact shopper questions from this pulse appear in plain text.\n"
        "2. Submit the updated page for indexing so it can be found publicly.\n"
        "3. Get one outside mention (review, directory, or partner page) that links to your URL."
    )


def _build_what_to_do_next(report: dict[str, Any]) -> str:
    """Bedrock “what to do next” footer for founders; counters + concrete actions."""
    product = report.get("product") or {}
    url = (product.get("url") or "").strip()
    brand = (product.get("name") or "Brand").strip() or "Brand"
    product_job = (
        str(report.get("product_job") or "").strip()
        or str(product.get("product_job") or "").strip()
        or str(product.get("one_liner") or "").strip()
        or str(product.get("h1") or "").strip()
        or str(product.get("description") or "").strip()
    )
    stats = _verdict_listing_stats(report)
    you_count = int(stats.get("you_count") or 0)
    total = int(stats.get("total") or 0)
    competitor_count = int(stats.get("highest_count") or 0)
    competitor_url = (stats.get("highest_url") or "").strip()

    questions = _report_queries(report)
    questions_str = "; ".join(questions) if questions else "(none)"
    models = report.get("models") or _load_selected_models() or list(DEFAULT_REPORT_MODELS)
    models_str = ", ".join(MODEL_LABELS.get(m, m) for m in models)

    prompt = WHAT_TO_DO_NEXT_PROMPT
    for key, val in (
        ("{{URL}}", url),
        ("{{BRAND}}", brand),
        ("{{PRODUCT_JOB}}", product_job),
        ("{{YOU_COUNT}}", str(you_count)),
        ("{{TOTAL}}", str(total)),
        ("{{COMPETITOR_COUNT}}", str(competitor_count)),
        ("{{COMPETITOR_URL}}", competitor_url),
        ("{{QUESTIONS}}", questions_str),
        ("{{MODELS}}", models_str),
    ):
        prompt = prompt.replace(key, val)

    def _pin_counters(body: str) -> str:
        """Force listing-aligned You line; keep Bedrock actions 1–3 only. No Highest line."""
        header = f"You: {you_count}/{total}"
        lines = (body or "").splitlines()
        actions: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("you:") or low.startswith("highest:"):
                continue
            if "re-run the pulse" in low:
                continue
            # Numbered actions (tolerate "1)" or "1.")
            if re.match(r"^[123][.)]\s*", s):
                # Normalize to "N. ..."
                num = s[0]
                rest = re.sub(r"^[123][.)]\s*", "", s).strip()
                actions.append(f"{num}. {rest}" if rest else f"{num}.")
            elif actions and not re.match(r"^\d+[.)]", s):
                # Continuation of previous action line
                actions[-1] = f"{actions[-1]} {s}".strip()
        # Keep at most 3 actions; if Bedrock failed to number, fall back later
        actions = actions[:3]
        if len(actions) < 3:
            return ""
        return header + "\n\n" + "\n".join(actions)

    try:
        text_out = _bedrock_text(
            prompt,
            system=(
                "Write only the exact output block specified. "
                "No GEO jargon. No medical claims. Max 80 words after the counters."
            ),
            max_tokens=320,
            temperature=0.5,
        )
        cleaned = (text_out or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        pinned = _pin_counters(cleaned)
        if pinned:
            return pinned
    except Exception:
        pass
    return _fallback_what_to_do_next(you_count, total, competitor_count, competitor_url)



def _build_report_text(report: dict[str, Any]) -> str:
    """Plain-text / PDF body — same structure as the locked HTML listing."""
    product = report.get("product") or {}
    url = (product.get("url") or "").strip()
    models = report.get("models") or _load_selected_models() or list(DEFAULT_REPORT_MODELS)
    if not models:
        models = list(DEFAULT_REPORT_MODELS)
    rows = report.get("rows") or []
    rows_by_model = report.get("rows_by_model") or {}
    queries = _report_queries(report)

    lines: list[str] = [
        "Site Pulse Audit",
        "",
        "Your URL:",
        url or "(none)",
        "",
        "Questions shoppers ask that fit your URL description.",
        "",
    ]

    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
        for mid in models:
            label = MODEL_LABELS.get(mid, mid)
            model_rows = rows_by_model.get(mid)
            if model_rows is None:
                model_rows = rows
            match = next(
                (r for r in (model_rows or []) if (r.get("query") or "").strip() == q),
                None,
            )
            if match is None:
                continue
            urls = _row_urls(match, product_url=url)
            if not urls:
                continue  # skip empty models (same as HTML)
            lines.append(f"{label} recommends:")
            for u in urls:
                lines.append(u)
            lines.append("")  # blank line between model blocks
        if lines and lines[-1] != "":
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    wtdn = (report.get("what_to_do_next") or "").strip()
    if wtdn:
        lines.append("Verdict")
        lines.append("")
        lines.append(wtdn)
        lines.append("")
    lines.append("Something great is coming your way!")
    return "\n".join(lines)


def _write_pdf_report(report: dict[str, Any], out_path: Path) -> Path:
    """Structured PDF-1.4 report (Helvetica + Helvetica-Bold). No third-party deps."""

    def esc(s: str) -> str:
        s = (s or "").encode("latin-1", "replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def wrap_text(s: str, max_chars: int) -> list[str]:
        s = (s or "").replace("\t", " ")
        if not s:
            return [" "]
        out: list[str] = []
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut < 20:
                cut = max_chars
            out.append(s[:cut])
            s = s[cut:].lstrip() or ""
        out.append(s if s else " ")
        return out

    product = report.get("product") or {}
    url = (product.get("url") or "").strip()
    models = report.get("models") or _load_selected_models() or list(DEFAULT_REPORT_MODELS)
    if not models:
        models = list(DEFAULT_REPORT_MODELS)
    rows = report.get("rows") or []
    rows_by_model = report.get("rows_by_model") or {}
    queries = _report_queries(report)

    # (font, size_pt, text, gap_after_pt) — F1 Helvetica, F2 Helvetica-Bold
    items: list[tuple[str, int, str, float]] = []
    items.append(("F2", 15, "Site Pulse Audit", 10))
    items.append(("F2", 11, "Your URL:", 4))
    items.append(("F1", 10, url or "(none)", 14))
    items.append(("F2", 12, "Questions shoppers ask that fit your URL description.", 10))

    for i, q in enumerate(queries, start=1):
        for line in wrap_text(f"{i}. {q}", 88):
            items.append(("F2", 11, line, 3))
        for mid in models:
            label = MODEL_LABELS.get(mid, mid)
            model_rows = rows_by_model.get(mid)
            if model_rows is None:
                model_rows = rows
            match = next(
                (r for r in (model_rows or []) if (r.get("query") or "").strip() == q),
                None,
            )
            if match is None:
                continue
            urls = _row_urls(match, product_url=url)
            if not urls:
                continue
            items.append(("F2", 11, f"{label} recommends:", 3))
            for u in urls:
                for line in wrap_text(u, 92):
                    items.append(("F1", 10, line, 2))
            items.append(("F1", 10, " ", 4))
        items.append(("F1", 10, " ", 8))

    wtdn = (report.get("what_to_do_next") or "").strip()
    if wtdn:
        items.append(("F2", 12, "Verdict", 6))
        for para in wtdn.splitlines() or [" "]:
            for line in wrap_text(para if para.strip() else " ", 92):
                items.append(("F1", 10, line, 3))
        items.append(("F1", 10, " ", 10))

    items.append(("F2", 11, "Something great is coming your way!", 4))

    page_w, page_h = 612, 792
    margin = 48
    bottom = margin + 24

    pages: list[list[tuple[str, int, str, float]]] = []
    cur: list[tuple[str, int, str, float]] = []
    y = float(page_h - margin)
    for font, size, txt, gap in items:
        need = float(size + 2 + gap)
        if y - need < bottom and cur:
            pages.append(cur)
            cur = []
            y = float(page_h - margin)
        cur.append((font, size, txt, gap))
        y -= need
    if cur:
        pages.append(cur)
    if not pages:
        pages = [[("F1", 10, " ", 0.0)]]

    streams: list[bytes] = []
    for page_items in pages:
        parts = ["BT"]
        y = float(page_h - margin)
        prev_font, prev_size = "", 0
        for font, size, txt, gap in page_items:
            if font != prev_font or size != prev_size:
                parts.append(f"/{font} {size} Tf")
                prev_font, prev_size = font, size
            parts.append(f"1 0 0 1 {margin} {y:.2f} Tm")
            parts.append(f"({esc(txt)}) Tj")
            y -= float(size + 2 + gap)
        parts.append("ET")
        streams.append("\n".join(parts).encode("latin-1", "replace"))

    objs: list[bytes] = []

    def add(data: bytes) -> int:
        objs.append(data)
        return len(objs)

    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"<< /Type /Pages /Kids [] /Count 0 >>")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_refs: list[int] = []
    for stream in streams:
        c_id = add(
            (f"<< /Length {len(stream)} >>\nstream\n").encode("ascii")
            + stream
            + b"\nendstream"
        )
        p_id = add(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w} {page_h}] "
                f"/Contents {c_id} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
            ).encode("ascii")
        )
        page_refs.append(p_id)

    kids = " ".join(f"{n} 0 R" for n in page_refs)
    objs[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode("ascii")

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(buf))
        buf.extend(f"{i} 0 obj\n".encode("ascii"))
        buf.extend(body)
        buf.extend(b"\nendobj\n")
    xref = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for i in range(1, len(objs) + 1):
        buf.extend(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        (
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    Path(out_path).write_bytes(bytes(buf))
    return Path(out_path)


def _write_html_report(report: dict[str, Any], draft_path: Path | None) -> Path:
    """Live report HTML matching web/report-draft-11point.html (no DRAFT banner)."""
    _ = draft_path  # callers still pass citation draft path; unused in this layout
    product = report.get("product") or {}
    url = (product.get("url") or "").strip()
    rows = report.get("rows") or []
    models = report.get("models") or _load_selected_models()
    if not models:
        models = list(DEFAULT_REPORT_MODELS)
    rows_by_model = report.get("rows_by_model") or {}

    queries = _report_queries(report)

    li_parts: list[str] = []
    for q in queries:
        model_blocks: list[str] = []
        for mid in models:
            label = MODEL_LABELS.get(mid, mid)
            model_rows = rows_by_model.get(mid)
            if model_rows is None:
                model_rows = rows
            match = next(
                (r for r in (model_rows or []) if (r.get("query") or "").strip() == q),
                None,
            )
            if match is None:
                continue
            urls = _row_urls(match, product_url=url)
            if not urls:
                continue  # skip empty models
            url_rows = "".join(
                f'<div class="url-row" tabindex="0">{html.escape(u)}</div>'
                for u in urls
            )
            model_blocks.append(
                '<div class="model-rec">'
                f'<p class="rec-label">{html.escape(label)} recommends:</p>'
                f"{url_rows}"
                "</div>"
            )
        body_inner = (
            f'<div class="strong-text">{html.escape(q)}</div>'
            + "".join(model_blocks)
        )
        li_parts.append(f'<li><div class="q-body">{body_inner}</div></li>')

    q_list_html = (
        '<ol class="q-list">\n'
        + "\n".join(li_parts)
        + "\n      </ol>"
        if li_parts
        else '<p class="muted">No shopper questions in this run.</p>'
    )

    url_href = html.escape(url) if url else "#"
    url_display = html.escape(url) if url else "(none)"
    scored = html.escape(str(report.get("scored_at") or ""))
    wtdn = (report.get("what_to_do_next") or "").strip()
    if wtdn:
        wtdn_html = (
            '<section class="box what-next-box" aria-label="Verdict">\n'
            '      <h2>Verdict</h2>\n'
            f'      <pre class="what-next">{html.escape(wtdn)}</pre>\n'
            "    </section>"
        )
    else:
        wtdn_html = ""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex" />
  <title>Site Pulse Audit</title>
  <style>
    :root {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: #FFFFFF;
      --accent: #00E5FF;
      --accent-strong: #7DF9FF;
      --neon: #00E5FF;
      --bg: #000B1E;
      --card: #001633;
      --input: #000822;
      --card-border: rgba(0, 229, 255, 0.35);
      --muted: #A8C5D6;
      --text: #FFFFFF;
      --btn-text: #000000;
      --box-border: 4px solid #00E5FF;
      --ok: #5CFFB0;
      --warn: #FFB74D;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    main {{ max-width: 720px; margin: 0 auto; padding: 1.25rem 1rem 3.5rem; }}
    a {{ color: var(--accent-strong); word-break: break-word; }}

    .box {{
      background: var(--card);
      border: var(--box-border);
      border-radius: 16px;
      padding: 1.25rem 1.3rem;
      margin: 1.15rem 0;
      box-shadow: 0 0 24px rgba(0, 229, 255, 0.12);
      break-inside: avoid;
    }}
    .box h1 {{
      margin: 0;
      color: var(--accent-strong);
      text-shadow: 0 0 18px rgba(0, 229, 255, 0.45);
      font-weight: 900;
      line-height: 1.15;
      font-size: clamp(1.55rem, 4.2vw, 2.05rem);
      letter-spacing: -0.02em;
    }}
    .box h2 {{
      margin: 0 0 .7rem;
      color: var(--accent-strong);
      text-shadow: 0 0 12px rgba(0, 229, 255, 0.28);
      font-weight: 900;
      font-size: 1.22rem;
      line-height: 1.3;
    }}
    .lead {{ color: var(--muted); margin: .55rem 0 .75rem; line-height: 1.45; }}
    .muted {{ color: var(--muted); }}
    .strong-text {{ color: var(--text); font-weight: 900; }}
    .v {{ color: var(--text); font-weight: 700; word-break: break-word; }}

    .q-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: .5rem;
      counter-reset: q;
    }}
    .q-list > li {{
      counter-increment: q;
      display: grid;
      grid-template-columns: 1.7rem 1fr;
      column-gap: .7rem;
      row-gap: .35rem;
      align-items: start;
      margin: 0 0 1rem;
      padding: .85rem .9rem;
      background: var(--input);
      border: 1px solid rgba(0, 229, 255, 0.22);
      border-radius: 12px;
      line-height: 1.4;
      width: 100%;
      box-sizing: border-box;
    }}
    .q-list > li::before {{
      content: counter(q);
      grid-column: 1;
      grid-row: 1;
      font-weight: 900;
      color: var(--btn-text);
      background: var(--accent);
      border-radius: 8px;
      width: 1.7rem;
      height: 1.7rem;
      display: grid;
      place-items: center;
      font-size: .85rem;
    }}
    .q-body {{
      grid-column: 2;
      grid-row: 1 / span 20;
      min-width: 0;
      width: 100%;
      display: block;
    }}
    .model-rec {{
      display: block;
      width: 100%;
      margin: .85rem 0 0;
    }}
    .model-rec .rec-label {{
      display: block;
      margin: 0 0 .45rem;
      font-weight: 900;
      color: var(--accent-strong);
      text-shadow: 0 0 10px rgba(0, 229, 255, 0.25);
      font-size: 1.02rem;
    }}
    .url-row {{
      display: block;
      width: calc(100% - .65rem);
      box-sizing: border-box;
      margin: 0 0 .45rem .65rem;
      padding: .7rem .85rem;
      background: var(--bg);
      border: 1px solid rgba(0, 229, 255, 0.35);
      border-radius: 10px;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: .9rem;
      line-height: 1.4;
      white-space: nowrap;
      overflow-x: auto;
      overflow-y: hidden;
      word-break: normal;
      overflow-wrap: normal;
      user-select: all;
      -webkit-user-select: all;
    }}

    .footer-box {{
      text-align: center;
      color: var(--muted);
      line-height: 1.5;
      font-size: .95rem;
    }}
    .what-next-box pre.what-next {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      line-height: 1.55;
      font-size: .95rem;
    }}
    .closing-box {{
      text-align: center;
      border-color: var(--neon);
      box-shadow: 0 0 28px rgba(0, 229, 255, 0.22);
    }}
    .closing-line {{
      margin: 0;
      color: var(--accent-strong);
      font-weight: 900;
      font-size: 1.15rem;
      letter-spacing: 0.01em;
      text-shadow: 0 0 16px rgba(0, 229, 255, 0.45);
    }}
  </style>
</head>
<body>
  <main>
    <section class="box" aria-label="Report header">
      <h1>Site Pulse Audit</h1>
      <p class="lead" style="margin-top:.75rem"><span class="strong-text">Your URL:</span></p>
      <p class="v" style="margin:.35rem 0 0;font-weight:700;word-break:break-word">
        <a href="{url_href}" rel="noopener">{url_display}</a>
      </p>
    </section>

    <section class="box" aria-label="Shopper questions">
      <h2>Questions shoppers ask that fit your URL description.</h2>
      {q_list_html}
    </section>

    {wtdn_html}

    <section class="box closing-box" aria-label="Closing">
      <p class="closing-line">Something great is coming your way!</p>
    </section>

    <footer class="box footer-box">
      Site Pulse Audit{f" · {scored}" if scored else ""}
    </footer>
  </main>
</body>
</html>
"""
    path = OUTPUT_DIR / "report.html"
    path.write_text(page, encoding="utf-8")
    return path



@tool
def probe_queries(
    backend: str = "live",
    brand: str | None = None,
    product_url: str | None = None,
) -> str:
    """Probe buying questions via mock SleepFix fixtures or live Bedrock answers.

    Args:
        backend: "mock" for offline SleepFix fixtures. "live" uses the submitted brand/URL pack.
        brand: Product brand name (live).
        product_url: Product page URL (live).
    """
    backend = (backend or "live").strip().lower()
    if backend not in {"mock", "live"}:
        backend = "live"
    brand = (brand or "").strip() or None
    product_url = _ensure_url(product_url or "") or None

    models = _load_selected_models()
    if not models:
        models = list(DEFAULT_REPORT_MODELS)

    if backend == "mock":
        pack = _load_sleepfix_pack()
        fixtures = _load_fixtures()
        results_by_model: dict[str, list[dict[str, Any]]] = {}
        for mid in models:
            results_by_model[mid] = [
                {
                    "query": q,
                    "answer": fixtures.get(q, "No fixture for this query."),
                    "backend": "mock",
                    "model": mid,
                }
                for q in pack["queries"]
            ]
    else:
        pack = None
        if brand and product_url:
            # Always rebuild so query-slot rules stay current
            pack = build_pack(brand, product_url)
        else:
            pack = _load_active_pack()
        if not pack:
            return json.dumps(
                {
                    "ok": False,
                    "error": "live probe requires brand+product_url or an existing active_pack.json",
                }
            )
        brand_name = brand or (pack.get("product") or {}).get("name")
        results_by_model = {}
        # Each selected model gets its own answers (persona), so tables can differ.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        prod = pack.get("product") or {}
        one_liner = (prod.get("one_liner") or "")
        url = product_url or (prod.get("url") or "")

        def _one(mid: str, q: str) -> dict[str, Any]:
            return {
                "query": q,
                "answer": _live_answer(
                    q,
                    brand=brand_name,
                    model_id=mid,
                    product_url=url,
                    one_liner=one_liner,
                ),
                "backend": "live",
                "model": mid,
            }

        jobs: list[tuple[str, str]] = [
            (mid, q) for mid in models for q in (pack.get("queries") or [])
        ]
        bucket: dict[str, list[dict[str, Any]]] = {mid: [] for mid in models}
        # Bound concurrency to stay friendly to Bedrock + Lambda.
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(models)))) as pool:
            futs = {pool.submit(_one, mid, q): (mid, q) for mid, q in jobs}
            for fut in as_completed(futs):
                mid, _q = futs[fut]
                bucket[mid].append(fut.result())
        # Keep query order stable per model
        q_order = list(pack.get("queries") or [])
        for mid in models:
            by_q = {r["query"]: r for r in bucket.get(mid) or []}
            results_by_model[mid] = [by_q[q] for q in q_order if q in by_q]

    # Flat results = first model (compat for older readers)
    results = results_by_model.get(models[0], []) if models else []

    product = pack["product"]
    out = {
        "product": product.get("name"),
        "product_url": product.get("url"),
        "backend": backend,
        "probed_at": _now(),
        "models": models,
        "results_by_model": results_by_model,
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(len(v) for v in results_by_model.values())
    return json.dumps(
        {
            "ok": True,
            "saved": str(path),
            "count": total,
            "models": models,
            "backend": backend,
        }
    )


def _extract_competitors_by_query(
    items: list[dict[str, Any]],
    competitors: list[str],
) -> dict[str, list[str]]:
    """One Bedrock pass: which seed competitors each answer actually names as solutions."""
    if not items or not competitors:
        return {}
    lines = []
    for i, item in enumerate(items):
        q = str(item.get("query") or "")
        a = str(item.get("answer") or "")[:700]
        lines.append(f"[{i}] Q: {q}\nA: {a}")
    catalog = ", ".join(competitors)
    body = "\n\n".join(lines)
    prompt = (
        "For each numbered Q/A below, list which brands from CATALOG are clearly named "
        "as recommended products/apps/programs in THAT answer.\n"
        "Rules:\n"
        "- Only brands from CATALOG.\n"
        "- Only if that answer names them for THIS question.\n"
        "- Do not copy the same brands onto every row unless each answer truly names them.\n"
        "- If none, use an empty list.\n"
        f"CATALOG: {catalog}\n\n"
        f"{body}\n\n"
        'Return ONLY JSON: {"rows": [{"i": 0, "mentioned": ["Brand"]}, ...]}'
    )
    try:
        text = _bedrock_text(
            prompt,
            system="Return only JSON. Be strict: per-answer mentions only.",
            max_tokens=900,
            temperature=0,
        )
        obj = _extract_json_object(text)
        rows = obj.get("rows") if isinstance(obj, dict) else None
        out: dict[str, list[str]] = {}
        if not isinstance(rows, list):
            return {}
        by_i: dict[int, list[str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            mentioned = _as_str_list(row.get("mentioned"))
            catalog_map = {_norm(c): c for c in competitors if c}
            cleaned: list[str] = []
            seen: set[str] = set()
            for m in mentioned:
                key = _norm(m)
                canon = catalog_map.get(key)
                if not canon:
                    for ck, cv in catalog_map.items():
                        if key == ck or key in ck or ck in key:
                            canon = cv
                            break
                if not canon or _norm(canon) in seen:
                    continue
                ans = str(items[idx].get("answer") or "") if 0 <= idx < len(items) else ""
                if ans and _competitor_mentioned(ans, canon):
                    seen.add(_norm(canon))
                    cleaned.append(_competitor_display(canon))
            by_i[idx] = cleaned
        for i, item in enumerate(items):
            out[str(item.get("query") or "")] = by_i.get(i, [])
        return out
    except Exception:
        return {}



@tool
def score_visibility() -> str:
    """Score each probed answer as named, ignored, or competitor-mentioned."""
    probe_path = OUTPUT_DIR / "probe.json"
    if not probe_path.exists():
        return json.dumps({"ok": False, "error": "Run probe_queries first."})
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    backend = probe.get("backend")
    pack = _pack_for_backend(backend)
    product = pack.get("product") or {}
    models = probe.get("models") or _load_selected_models() or list(DEFAULT_REPORT_MODELS)
    results_by_model = probe.get("results_by_model") or {}
    if not results_by_model and probe.get("results"):
        # Legacy single-probe file
        results_by_model = {models[0]: list(probe.get("results") or [])}

    aliases = product.get("brand_aliases") or [product.get("name") or ""]
    competitors = pack.get("competitors") or []
    url = product.get("url") or probe.get("product_url") or ""
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for mid in models:
        model_rows: list[dict[str, Any]] = []
        for item in results_by_model.get(mid) or []:
            scored = _score_answer(
                item.get("answer") or "",
                aliases,
                competitors,
                query=item.get("query") or "",
                product_url=url,
            )
            full_answer = item.get("answer") or ""
            row = {
                "query": item.get("query"),
                "model": mid,
                **scored,
                "answer": full_answer,
                "answer_excerpt": full_answer[:220],
            }
            model_rows.append(row)
            all_rows.append(row)
        rows_by_model[mid] = model_rows

    # Prefer first model's rows for legacy "rows"; summary across all model answers
    rows = rows_by_model.get(models[0], []) if models else []
    named_n = sum(1 for r in all_rows if r["status"] == "named")
    ignored_n = sum(1 for r in all_rows if r["status"] == "ignored")
    competitor_n = sum(1 for r in all_rows if r["status"] == "competitor")
    summary = {
        "named": named_n,
        "ignored": ignored_n,
        "competitor": competitor_n,
        "total": len(all_rows),
        "YOU": named_n,
        "COMPETITOR": competitor_n,
        "NEITHER": ignored_n,
    }
    url_n = _norm(url)
    page_cited = False
    if url_n:
        for mid_items in results_by_model.values():
            for item in mid_items or []:
                if url_n in _norm(item.get("answer") or ""):
                    page_cited = True
                    break
            if page_cited:
                break
    report = {
        "product": {
            "name": product.get("name") or probe.get("product"),
            "url": url,
            "one_liner": product.get("one_liner", ""),
            "brand_aliases": product.get("brand_aliases") or [],
        },
        "backend": backend or pack.get("backend"),
        "scored_at": _now(),
        "summary": summary,
        "rows": rows,
        "rows_by_model": rows_by_model,
        "models": models,
        "page_cited": page_cited,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "score_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    brand = report["product"]["name"] or "Brand"
    leaf = _url_leaf(url)
    comps: list[str] = []
    seen_c: set[str] = set()
    for r in rows:
        for c in r.get("competitors_mentioned") or []:
            k = _norm(c)
            if k and k not in seen_c:
                seen_c.add(k)
                comps.append(c)
    pulse = _build_pulse_block(
        brand,
        url,
        product.get("one_liner") or "",
        summary,
        all_rows,
        page_cited,
    )
    top_comp = pulse.get("most_mentioned_competitor")
    top_comp_url = pulse.get("most_mentioned_competitor_url")
    named_n = int(pulse.get("named_mentions") or 0)
    not_named_n = int(pulse.get("not_named_mentions") or 0)
    verdict = pulse.get("verdict") or ""
    next_action = pulse.get("next_action") or ""
    report["pulse"] = pulse
    report["product_job"] = (
        str(pack.get("product_job") or "").strip()
        or str(product.get("product_job") or "").strip()
        or str(product.get("one_liner") or "").strip()
        or str(product.get("h1") or "").strip()
        or str(product.get("description") or "").strip()
    )
    if product.get("h1"):
        report["product"]["h1"] = product.get("h1")
    if product.get("description"):
        report["product"]["description"] = product.get("description")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["what_to_do_next"] = _build_what_to_do_next(report)
    report_text = _build_report_text(report)
    report["report_text"] = report_text
    md_path = OUTPUT_DIR / "score_report.md"
    md_path.write_text(report_text, encoding="utf-8")
    txt_path = OUTPUT_DIR / "score_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")
    pdf_path = OUTPUT_DIR / "score_report.pdf"
    try:
        _write_pdf_report(report, pdf_path)
    except Exception as exc:  # noqa: BLE001 — PDF is best-effort; HTML/txt still ship
        pdf_path = None
        print(f"pdf write failed: {exc}")
    # Persist report_text on JSON so draft_citation rewrite keeps full download body
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = _write_html_report(report, OUTPUT_DIR / "citation_draft.md")
    return json.dumps(
        {
            "ok": True,
            "summary": summary,
            "saved": [str(path), str(md_path), str(html_path)],
        }
    )


@tool
def draft_citation_page() -> str:
    """Draft one citation-ready FAQ page aimed at LLM quotability for the active brand."""
    probe_path = OUTPUT_DIR / "probe.json"
    backend = None
    if probe_path.exists():
        try:
            backend = json.loads(probe_path.read_text(encoding="utf-8")).get("backend")
        except json.JSONDecodeError:
            backend = None
    pack = _pack_for_backend(backend)
    score_path = OUTPUT_DIR / "score_report.json"
    gaps = []
    if score_path.exists():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        gaps = [r["query"] for r in score["rows"] if r["status"] != "named"]

    product = pack["product"]
    name = product.get("name") or "Product"
    url = product.get("url") or ""
    one_liner = product.get("one_liner") or f"{name} ({url})"
    queries = pack.get("queries") or []
    competitors = pack.get("competitors") or []

    lines = [
        f"# {name}",
        "",
        one_liner,
        "",
        f"## What is {name}?",
        "",
        f"{name} is {one_liner.rstrip('.')}. It is available at {url}." if url else f"{name} is {one_liner.rstrip('.')}.",
        "",
    ]
    for q in queries:
        lines.extend(
            [
                f"## {q}",
                "",
                f"{name} is a relevant option for this question. {one_liner} "
                f"{'See ' + url + ' for details.' if url else ''}".strip(),
                "",
            ]
        )
    if competitors:
        lines.extend(
            [
                f"## How {name} compares",
                "",
                f"People often consider {', '.join(competitors[:8])} in this category. "
                f"{name} is positioned as: {one_liner}",
                "",
            ]
        )
    lines.extend(
        [
            "## Sources (for quotability)",
            "",
            "Cite the official product page and primary category sources when answering buyers. "
            f"Canonical URL: {url or 'n/a'}.",
            "",
            "---",
            "Draft generated for LLMCite / Site Pulse Audit approval gate.",
            "Priority gaps this draft targets:",
        ]
    )
    body = "\n".join(lines)
    for g in gaps[:6]:
        body += f"\n- {g}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "citation_draft.md"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    if score_path.exists():
        report = json.loads(score_path.read_text(encoding="utf-8"))
        # Always rebuild download body from locked listing shape (never restore old format)
        if not (report.get("what_to_do_next") or "").strip():
            report["what_to_do_next"] = _build_what_to_do_next(report)
        report["report_text"] = _build_report_text(report)
        (OUTPUT_DIR / "score_report.txt").write_text(report["report_text"], encoding="utf-8")
        (OUTPUT_DIR / "score_report.md").write_text(report["report_text"], encoding="utf-8")
        path_json = OUTPUT_DIR / "score_report.json"
        path_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_html_report(report, path)
        # refresh pdf if possible
        try:
            _write_pdf_report(report, OUTPUT_DIR / "score_report.pdf")
        except Exception:
            pass
    return json.dumps({"ok": True, "saved": str(path), "gaps_targeted": len(gaps), "brand": name})


@tool
def await_approval(decision: str = "pending") -> str:
    """Record human approval before any publish step.

    Args:
        decision: "approve", "reject", or "pending"
    """
    decision = decision.strip().lower()
    if decision not in {"approve", "reject", "pending"}:
        return json.dumps({"ok": False, "error": "decision must be approve|reject|pending"})
    record = {
        "decision": decision,
        "decided_at": _now(),
        "publish_allowed": decision == "approve",
        "note": "LLMCite never auto-publishes. Approval only unlocks the next human deploy step.",
    }
    path = OUTPUT_DIR / "approval.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return json.dumps({"ok": True, **record, "saved": str(path)})


@tool
def recheck_visibility() -> str:
    """Re-run probe+score and write a simple diff against the previous score report."""
    prev_path = OUTPUT_DIR / "score_report.json"
    prev = None
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))

    backend = "mock"
    brand = None
    product_url = None
    probe_path = OUTPUT_DIR / "probe.json"
    if probe_path.exists():
        last = json.loads(probe_path.read_text(encoding="utf-8"))
        backend = last.get("backend") or "mock"
        brand = last.get("product")
        product_url = last.get("product_url")
    if backend == "live" and not product_url:
        active = _load_active_pack()
        if active:
            brand = active["product"]["name"]
            product_url = active["product"]["url"]

    probe_queries(backend=backend, brand=brand, product_url=product_url)
    score_visibility()
    new = json.loads((OUTPUT_DIR / "score_report.json").read_text(encoding="utf-8"))

    def bag(report: dict | None) -> dict[str, str]:
        if not report:
            return {}
        return {r["query"]: r["status"] for r in report["rows"]}

    before = bag(prev)
    after = bag(new)
    changes = []
    for q, status in after.items():
        old = before.get(q)
        if old != status:
            changes.append({"query": q, "from": old, "to": status})

    diff = {
        "rechecked_at": _now(),
        "before_summary": prev["summary"] if prev else None,
        "after_summary": new["summary"],
        "changes": changes,
    }
    path = OUTPUT_DIR / "recheck_diff.json"
    path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
    return json.dumps(
        {
            "ok": True,
            "changes": len(changes),
            "saved": str(path),
            "summary": new["summary"],
        }
    )


def build_agent():
    from strands import Agent

    return Agent(
        system_prompt=(
            "You are LLMCite, the Site Pulse Audit marketing-engineer agent. "
            "Help the submitted brand get named in LLM-style answers. "
            "Always use tools in order: probe_queries → score_visibility → "
            "draft_citation_page → await_approval. "
            "Never claim you published anything. Keep outputs concise."
        ),
        tools=[
            probe_queries,
            score_visibility,
            draft_citation_page,
            await_approval,
            recheck_visibility,
        ],
    )


def run_pipeline(auto_approve: bool = False) -> None:
    """Deterministic demo path (no model keys required)."""
    print("LLMCite · Site Pulse Audit")
    print("1) probe_queries")
    print(probe_queries(backend="mock"))
    print("2) score_visibility")
    print(score_visibility())
    print("3) draft_citation_page")
    print(draft_citation_page())
    decision = "approve" if auto_approve else "pending"
    if not auto_approve:
        raw = input("Approve citation draft? [approve/reject/pending]: ").strip().lower()
        if raw in {"approve", "reject", "pending"}:
            decision = raw
    print("4) await_approval")
    print(await_approval(decision=decision))
    print("5) recheck_visibility")
    print(recheck_visibility())
    print(f"\nDone. Open {OUTPUT_DIR / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMCite PoC (Site Pulse Audit)")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "agent"],
        default="pipeline",
        help="pipeline = offline demo tools; agent = Strands Agent (needs model access)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip interactive approval (pipeline mode)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Run a full LLMCite loop using mock probes. "
            "Probe, score, draft a citation page, then set approval to pending."
        ),
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "pipeline":
        run_pipeline(auto_approve=args.auto_approve)
        return

    agent = build_agent()
    result = agent(args.prompt)
    print(result)


if __name__ == "__main__":
    main()

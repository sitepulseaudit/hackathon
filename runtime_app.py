"""AgentCore Runtime entrypoint for LLMCite."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

# Writable output on AgentCore
os.environ.setdefault("LLMCITE_OUTPUT_DIR", "/tmp/llmcite-output")
Path(os.environ["LLMCITE_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()


def _parse_payload(payload):
    if payload is None:
        return {}
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        payload = payload.strip() or "{}"
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"prompt": payload, "mode": "pipeline"}
    if isinstance(payload, dict):
        # some gateways nest under "payload" / "input"
        if "mode" not in payload and isinstance(payload.get("payload"), (dict, str, bytes)):
            return _parse_payload(payload.get("payload"))
        if "mode" not in payload and isinstance(payload.get("input"), (dict, str, bytes)):
            return _parse_payload(payload.get("input"))
        return payload
    return {"raw": str(payload)}


@app.entrypoint
def invoke(payload: dict, context=None):
    """Invoke LLMCite pipeline (live when product_url is set; else mock)."""
    try:
        data = _parse_payload(payload)
        mode = data.get("mode", "pipeline")

        model_id = (data.get("bedrock_model_id") or data.get("model_id") or "").strip()
        if model_id:
            os.environ["LLMCITE_BEDROCK_MODEL_ID"] = model_id

        if mode == "pipeline":
            from agent import (
                OUTPUT_DIR,
                await_approval,
                build_pack,
                draft_citation_page,
                hostname_brand,
                probe_queries,
                recheck_visibility,
                score_visibility,
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            product_url = (data.get("product_url") or data.get("url") or "").strip()
            brand = (data.get("brand") or "").strip()
            backend = (data.get("backend") or "").strip().lower()
            if not backend:
                backend = "live" if product_url else "mock"
            if backend not in {"live", "mock"}:
                backend = "live" if product_url else "mock"

            if backend == "live":
                if not product_url:
                    raise ValueError("product_url is required for live backend")
                brand = brand or hostname_brand(product_url)
                build_pack(brand, product_url)
                probe_queries(backend="live", brand=brand, product_url=product_url)
            else:
                probe_queries(backend="mock")

            score_visibility()
            draft_citation_page()
            decision = data.get("decision", "pending")
            await_approval(decision=decision)
            # skip full recheck on runtime to reduce failure surface; optional
            if data.get("recheck", False):
                recheck_visibility()
            score = json.loads((OUTPUT_DIR / "score_report.json").read_text())
            report_html = (OUTPUT_DIR / "report.html").read_text(encoding="utf-8")
            citation = (OUTPUT_DIR / "citation_draft.md").read_text(encoding="utf-8")
            product = score.get("product") or {}
            return {
                "ok": True,
                "mode": "pipeline",
                "backend": backend,
                "brand": product.get("name") or brand,
                "product_url": product.get("url") or product_url,
                "summary": score.get("summary"),
                "report_html": report_html,
                "citation_draft": citation,
            }

        # agent mode needs model credentials; keep available but default pipeline
        from agent import build_agent

        prompt = data.get(
            "prompt",
            "Run a full LLMCite loop. If a brand and product_url are provided, use live probes.",
        )
        agent = build_agent()
        result = agent(prompt)
        return {"ok": True, "mode": "agent", "result": str(result)}
    except Exception as exc:  # noqa: BLE001 - return error to caller for debugging
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }


if __name__ == "__main__":
    app.run()

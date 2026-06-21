"""Vercel Python serverless function — DUALMIND live scoring.

Powers the demo's "write your own email" box. Returns the SAME JSON `result` shape as
`web/demo_emails.json[*].result`, so the frontend renders presets and live input the
same way.

    POST /api/analyze   {content, goal?, content_type?}  -> { ...result... }
    GET  /api/analyze                                    -> the preset demo emails

Deploy notes (Vercel):
  - Set env  ANTHROPIC_API_KEY  on the project (required for real scoring; without it
    the function runs DUALMIND's deterministic Mock provider and still works).
  - Install the SLIM deps in `api/requirements.txt` (anthropic, pyyaml, numpy,
    scikit-learn, python-dotenv) — NOT the repo's full dev requirements.txt
    (torch/transformers/matplotlib are not needed here and exceed the bundle limit).
  - Give it time + memory and bundle the source dirs, e.g. vercel.json:
      { "functions": { "api/analyze.py": {
          "memory": 1024, "maxDuration": 60,
          "includeFiles": "{config.py,config.yaml,llm/**,prefilter/**,dual_llm/**,taint/**,review_gate/**,knowledge_base/**,audit/**,calibration/**,eval/mock_agents.py}"
      } } }
  - DUALMIND is initialized once per warm instance; the LLM cache is redirected to the
    writable /tmp (Vercel's filesystem is otherwise read-only).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_DM = None  # initialized lazily on first request, cached per warm instance


def _init():
    global _DM
    if _DM is not None:
        return
    from config import load_and_seed
    from llm import get_provider
    from pipeline import DualMind
    from prefilter import PreFilter

    cfg = load_and_seed()
    # Vercel FS is read-only except /tmp — point the LLM cache there.
    cfg.setdefault("eval", {})["cache_dir"] = os.path.join(
        tempfile.gettempdir(), "dualmind_llm_cache")
    provider = get_provider(cfg)
    if provider.name == "mock":
        from eval.mock_agents import install_mock_agents
        install_mock_agents(provider)
    _DM = DualMind(cfg, provider=provider,
                   prefilter=PreFilter.with_default_classifier(cfg))


def _serialize(tr, dm):
    from audit import verify_chain
    decider = tr.decider
    calls = []
    if decider is not None:
        for c in decider.calls:
            calls.append({"tool": c.tool, "flagged": c.flagged,
                          "args": {k: {"value": str(getattr(v, "value", v)),
                                       "label": v.label.value} for k, v in c.args.items()}})
    pf = tr.prefilter
    v = verify_chain(dm.audit)
    return {
        "verdict": tr.final_verdict, "caught_by": tr.caught_by,
        "fast_path": tr.fast_path, "routing": tr.routing,
        "risk": round(tr.risk, 3), "defended": tr.defended,
        "components": {"prefilter": round(tr.prefilter_risk, 3),
                       "dual_llm": round(tr.dual_llm_risk, 3),
                       "taint": round(tr.taint_risk, 3)},
        "prefilter": None if pf is None else {
            "verdict": pf.verdict, "risk": round(pf.risk, 3),
            "signals": pf.signals[:8], "latency_ms": round(pf.latency_ms, 2)},
        "reader": None if tr.reader is None else {
            "suspicious": tr.reader.suspicious,
            "contains_instructions": tr.reader.contains_instructions,
            "summary": tr.reader.summary,
            "suspicion_reason": tr.reader.intent.get("suspicion_reason", ""),
            "actions": tr.reader.requested_actions},
        "decider": None if decider is None else {"decision": decider.decision, "calls": calls},
        "taint_findings": tr.taint_findings,
        "latency_ms": round(tr.latency_ms, 1),
        "audit_entries": len(dm.audit), "audit_valid": v.valid,
    }


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # Serve the preset demo emails so the frontend can load them from one place.
        try:
            p = os.path.join(_ROOT, "web", "demo_emails.json")
            data = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"emails": []}
            self._send(200, data)
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "invalid JSON body"})
        content = (payload.get("content") or "").strip()
        if not content:
            return self._send(400, {"error": "empty content"})
        goal = payload.get("goal") or "Triage and summarize my new email."
        ctype = payload.get("content_type") or "email"
        try:
            _init()
            tr = _DM.process(content, user_goal=goal, content_type=ctype)
            self._send(200, _serialize(tr, _DM))
        except Exception as e:  # pragma: no cover - LLM/env dependent
            self._send(500, {"error": str(e)})

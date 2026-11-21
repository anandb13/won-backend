"""POST /api/ai/chat: conversational gathering of the inputs a WON needs.

Stateless. The browser sends the conversation, the allowed values and the draft collected so far.
The model records what the user said through the update_won_draft tool. After each call the server
merges the patch and replies with what is still missing (rules below mirror src/lib/wonDraft.js),
so the model asks for the right things and cannot declare the WON ready early. The model never
computes cost.

Run:  pip install -r requirements.txt
      export ANTHROPIC_API_KEY=...
      uvicorn main:app --reload --port 8000
"""
import json
import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from guard import guard

router = APIRouter(prefix="/api/ai")

# Provider is chosen by env vars (see README):
#   WON_AI_PROVIDER = openai (default; Ollama, Groq, Gemini, OpenRouter, ...) | anthropic
PROVIDER = os.getenv("WON_AI_PROVIDER", "openai")
MODEL = os.getenv("WON_AI_MODEL", "qwen2.5:7b" if PROVIDER == "openai" else "claude-sonnet-5-5")
BASE_URL = os.getenv("WON_AI_BASE_URL", "http://localhost:11434/v1")  # default: local Ollama
API_KEY = os.getenv("WON_AI_API_KEY", "ollama")  # Ollama ignores it, but the SDK wants a value
MAX_TURNS = 20  # history sent to the model
MAX_TOOL_ROUNDS = 3
OPTIONAL = {"contractId": "Contract ID", "contractValue": "Contract value", "referenceWonNumber": "Reference WON number"}

_client = None


def get_client():
    global _client
    if _client is None:
        if PROVIDER == "anthropic":
            from anthropic import AsyncAnthropic

            _client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        else:
            from openai import AsyncOpenAI

            _client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Category(BaseModel):
    id: str
    title: str = ""


class Context(BaseModel):
    months: list[str]
    categories: list[Category]
    countries: list[str]
    indiaLocations: list[str]
    businessUnits: list[str]
    currencies: list[str]


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    context: Context
    draft: dict[str, Any]


class ChatResponse(BaseModel):
    reply: str
    draft: dict[str, Any]


# ---- draft rules (keep in sync with src/lib/wonDraft.js) -------------------
def normalize_rows(rows: Any, ctx: Context) -> list[dict]:
    cats = {c.id for c in ctx.categories}
    out = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        role, cat = str(r.get("role", "")).strip(), str(r.get("category", "")).upper()
        if not role or cat not in cats:
            continue
        country = r.get("country") if r.get("country") in ctx.countries else "India"
        loc, bu = r.get("workLocation", ""), r.get("businessUnit", "")
        try:
            count = min(max(round(float(r.get("count", 1))), 1), 100)
        except (TypeError, ValueError):
            count = 1
        out.append({
            "role": role[:80], "category": cat, "country": country, "count": count,
            "workLocation": loc if country == "India" and loc in ctx.indiaLocations else "",
            "businessUnit": bu if bu in ctx.businessUnits else "",
        })
    return out


def normalize_draft(d: dict, ctx: Context) -> dict:
    def text(v): return v.strip()[:120] if isinstance(v, str) else ""
    start = d.get("startMonth") if d.get("startMonth") in ctx.months else ""
    end = d.get("endMonth") if d.get("endMonth") in ctx.months else ""
    if start and end and ctx.months.index(end) < ctx.months.index(start):
        end = ""
    value = d.get("contractValue", "")
    try:
        value = float(value) if value not in ("", None) and float(value) >= 0 else ""
    except (TypeError, ValueError):
        value = ""
    return {
        "name": text(d.get("name")),
        "referenceWonNumber": text(d.get("referenceWonNumber")),
        "contractId": text(d.get("contractId")),
        "contractValue": value,
        "contractValueCurrency": d.get("contractValueCurrency") if d.get("contractValueCurrency") in ctx.currencies else "INR",
        "startMonth": start,
        "endMonth": end,
        "rows": normalize_rows(d.get("rows"), ctx),
        "skipped": [k for k in d.get("skipped", []) if k in OPTIONAL] if isinstance(d.get("skipped"), list) else [],
    }


def merge_patch(draft: dict, patch: dict, ctx: Context) -> dict:
    merged = dict(draft)
    for k in ("name", "referenceWonNumber", "contractId", "contractValue", "contractValueCurrency",
              "startMonth", "endMonth", "rows"):
        if k in patch:
            merged[k] = patch[k]
    if "skipped" in patch:
        merged["skipped"] = sorted(set(draft.get("skipped", [])) | set(patch["skipped"]))
    return normalize_draft(merged, ctx)


def missing_inputs(d: dict) -> list[dict]:
    out = []
    if not d["name"]:
        out.append({"key": "name", "label": "WON name", "required": True})
    if not d["startMonth"] or not d["endMonth"]:
        out.append({"key": "dates", "label": "Start and end month", "required": True})
    if not d["rows"]:
        out.append({"key": "rows", "label": "Team: roles, grade, headcount, country", "required": True})
    no_loc = [r["role"] for r in d["rows"] if r["country"] == "India" and not r["workLocation"]]
    if no_loc:
        out.append({"key": "workLocation", "label": "Indian work location for: " + ", ".join(no_loc), "required": True})
    for key, label in OPTIONAL.items():
        if d[key] in ("", None) and key not in d["skipped"]:
            out.append({"key": key, "label": label, "required": False})
    return out


# ---- prompt and tool ------------------------------------------------------
def build_system(ctx: Context) -> str:
    cats = "\n".join(f"- {c.id}: {c.title}" for c in ctx.categories)
    return f"""You help a delivery manager create a Work Order Number (WON). You gather the inputs, then the app builds the WON.

Fields:
- name (required), startMonth and endMonth (required, YYYY-MM, one of: {', '.join(ctx.months)}; end is not before start)
- rows (required): the team. Each row has role, category, country, count; India rows also need workLocation.
  Optional on a row: businessUnit. Every row spans the whole start to end period.
- Optional: contractId, contractValue (+ contractValueCurrency), referenceWonNumber

Allowed values:
Categories (grades):
{cats}
Countries: {', '.join(ctx.countries)}
Indian work locations: {', '.join(ctx.indiaLocations)}
Business units: {', '.join(ctx.businessUnits)}
Currencies: {', '.join(ctx.currencies)}

How to work:
- Call update_won_draft with ONLY the fields the user gave or changed this turn. rows, when sent, is the COMPLETE team
  (start from the current draft rows when the user tweaks it). The tool result lists what is still missing.
- Never invent the WON name, dates, an Indian work location, contract details or a business unit. If the user has not
  said it, it is missing: ask. You may propose a team from a project description, since the user reviews it.
- When items are missing, ask for ALL the required ones together in one short message. Ask about the optional ones once;
  if the user declines ("skip", "don't have it"), send them in skipped.
- Convert relative periods ("8 months from August") to startMonth/endMonth using the allowed months.
- When no required item is missing, summarize in a sentence or two and tell the user to press "Build WON".
  Do not say the WON has been created or saved.
- Never state costs, rates or totals. The app shows the estimate.
- Use values exactly as listed above (months as YYYY-MM). Omit fields you do not know instead of sending empty values.

Tone: talk like a friendly, capable colleague, not a form.
- Open with a short natural reaction to what they said ("Got it, an 8 month mobile build starting August.").
- Use plain sentences, no bullet lists, no field names like "startMonth". Say "start and end month".
- Ask for what is missing in one conversational message, e.g. "Two quick things: what should we call
  this WON, and which Indian location will the team sit in?" Ask the optional items lightly, once.
- When you propose a team, say in a sentence why (e.g. one lead plus a small build crew) and invite changes.
- When everything is in, wrap up warmly in one or two sentences and point to the Build WON button.
- Keep it to 2 to 4 sentences. No emojis, no filler, no repeating their request back in full."""


def build_tool(ctx: Context) -> dict:
    # No enums here on purpose: some providers reject the whole call when the model sends a value
    # slightly off-list. Allowed values are in the system prompt, and normalize_draft() enforces them.
    row = {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "category": {"type": "string", "description": "grade id, e.g. ASE"},
            "country": {"type": "string"},
            "count": {"type": "integer"},
            "workLocation": {"type": "string", "description": "India rows only; omit if unknown"},
            "businessUnit": {"type": "string", "description": "omit unless the user said it"},
        },
        "required": ["role", "category", "country", "count"],
    }
    return {
        "name": "update_won_draft",
        "description": "Record WON inputs the user provided or changed. Include only fields that changed this "
                       "turn. Omit any field you do not know; never send empty strings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "referenceWonNumber": {"type": "string"},
                "contractId": {"type": "string"},
                "contractValue": {"type": "number"},
                "contractValueCurrency": {"type": "string", "description": "e.g. INR"},
                "startMonth": {"type": "string", "description": "YYYY-MM, e.g. 2026-08"},
                "endMonth": {"type": "string", "description": "YYYY-MM, e.g. 2027-03"},
                "rows": {"type": "array", "items": row, "description": "The complete team."},
                "skipped": {"type": "array", "items": {"type": "string"},
                            "description": "optional fields the user declined: contractId, contractValue, referenceWonNumber"},
            },
        },
    }


def fallback_reply(draft: dict) -> str:
    need = [m["label"] for m in missing_inputs(draft) if m["required"]]
    if need:
        return "To build the WON I still need: " + "; ".join(need) + "."
    return 'Everything needed is in. Review the draft and press "Build WON".'


def apply_call(draft: dict, args: Any, ctx: Context) -> tuple[dict, str]:
    """Merge one update_won_draft call and build the tool result the model sees."""
    draft = merge_patch(draft, args if isinstance(args, dict) else {}, ctx)
    return draft, json.dumps({"draft": draft, "missing": missing_inputs(draft)})


async def run_anthropic(system: str, msgs: list[dict], ctx: Context, draft: dict) -> tuple[str, dict]:
    text = ""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = await get_client().messages.create(
            model=MODEL, max_tokens=1500, system=system, tools=[build_tool(ctx)], messages=msgs
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [b for b in resp.content if b.type == "tool_use" and b.name == "update_won_draft"]
        if not calls:
            break
        results = []
        for b in calls:
            draft, out = apply_call(draft, b.input, ctx)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        msgs += [{"role": "assistant", "content": resp.content}, {"role": "user", "content": results}]
        if resp.stop_reason != "tool_use":
            break
        text = ""  # the model has not yet seen the tool result; wait for its next turn
    return text, draft


async def run_openai(system: str, msgs: list[dict], ctx: Context, draft: dict) -> tuple[str, dict]:
    """Any OpenAI-compatible server: Ollama, Groq, Gemini, OpenRouter, ..."""
    tool = build_tool(ctx)
    tools = [{"type": "function", "function": {
        "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}]
    messages = [{"role": "system", "content": system}] + msgs
    text = ""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = await get_client().chat.completions.create(
            model=MODEL, messages=messages, tools=tools, max_tokens=1500
        )
        m = resp.choices[0].message
        text = (m.content or "").strip()
        calls = m.tool_calls or []
        if not calls:
            break
        messages.append({
            "role": "assistant", "content": m.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in calls],
        })
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except ValueError:
                args = {}
            if c.function.name == "update_won_draft":
                draft, out = apply_call(draft, args, ctx)
            else:
                out = json.dumps({"error": "unknown tool"})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        text = ""
    return text, draft


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(guard)])
async def chat(req: ChatRequest) -> ChatResponse:
    ctx = req.context
    draft = normalize_draft(req.draft, ctx)
    msgs: list[dict] = [{"role": m.role, "content": m.content} for m in req.messages[-MAX_TURNS:]]
    if msgs[0]["role"] != "user":  # providers require the first turn to be the user's
        msgs = msgs[1:]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Last message must be from the user.")

    # Current draft goes in on each turn, so the model's view is never stale.
    system = build_system(ctx) + "\n\nCurrent draft:\n" + json.dumps(draft) + \
        "\nStill missing: " + json.dumps(missing_inputs(draft))
    run = run_anthropic if PROVIDER == "anthropic" else run_openai
    try:
        text, draft = await run(system, msgs, ctx, draft)
    except Exception as e:  # network, auth, rate limit, model not pulled
        print(f"AI call failed: {type(e).__name__}: {e}")
        raise HTTPException(502, "The AI service is unavailable. Check the backend terminal.") from e
    return ChatResponse(reply=text or fallback_reply(draft), draft=draft)
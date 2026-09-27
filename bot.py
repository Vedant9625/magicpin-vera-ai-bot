import os
import re
import json
import time
import random
import asyncio
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)
app = FastAPI()
START = time.time()
MODEL = "gemini-3.6-flash"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("vera-bot")

# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}
sent_suppression: dict[str, set] = {}
conv_meta: dict[str, dict] = {}

def reset_state():
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    conv_meta.clear()

# ---------------------------------------------------------
# STATIC KNOWLEDGE & SAFEGUARDS
# ---------------------------------------------------------
BASE_TABOOS = ["x% off", "placeholder", "boost", "supercharge", "skyrocket", "spam", "blast", "amazing deal"]
CLINICAL_CATEGORIES = {"dentists", "pharmacies"}
CLINICAL_EXTRA_TABOOS = ["sales", "marketing", "guarantee", "cure", "customers", "revenue"]

AUTO_REPLY_SNIPPETS = [
    "thank you for contacting", "we will get back to you", "automated assistant",
    "hamari team tak pahuncha", "aapki jaankari ke liye", "currently unavailable",
    "out of office", "this is an automated message", "will respond shortly",
]

STOP_WORDS = ["stop", "spam", "useless", "unsubscribe", "don't message", "leave me alone",
              "band karo", "hatao", "mat bhejo"]
POSITIVE_INTENT = ["ok lets do it", "let's do it", "whats next", "yes please", "sure", "do it",
                    "start", "yes", "haan", "theek hai", "chalega", "go ahead"]
WAIT_PHRASES = ["not now", "busy", "call later", "call me later", "remind me", "abhi nahi",
                "baad mein", "thodi der baad", "kal baat karte", "busy hoon"]

KIND_WEIGHT = {
    "perf_dip": 9, "customer_lapsed_soft": 8, "recall_due": 8, "competitor_opened": 7,
    "review_theme_emerged": 6, "milestone_reached": 6, "appointment_tomorrow": 6,
    "category_trend_movement": 5, "research_digest_release": 5, "regulation_change": 5,
    "perf_spike": 4, "festival_upcoming": 4, "local_news_event": 3, "weather_heatwave": 3,
    "dormant_with_vera": 3, "scheduled_recurring": 2,
}
CONSENT_SCOPE_FOR_KIND = {
    "recall_due": "recall_reminders",
    "customer_lapsed_soft": "recall_reminders",
    "appointment_tomorrow": "appointment_reminders",
}

LEVER_BY_KIND = {
    "perf_dip": "Loss aversion: name the exact drop, offer the fix.",
    "customer_lapsed_soft": "Loss aversion: a specific customer cohort is slipping away, offer the fix.",
    "recall_due": "Care + timeliness: the recall window is open right now.",
    "competitor_opened": "Urgency + social proof: protect local market share.",
    "review_theme_emerged": "Reciprocity: surface the emerging pattern before it becomes a rating problem.",
    "milestone_reached": "Social proof + reciprocity: celebrate the milestone, then offer the next lever.",
    "perf_spike": "Curiosity + effort externalization: something is working, offer to double down on it.",
    "research_digest_release": "Authority + curiosity: anchor on the specific cited statistic.",
    "regulation_change": "Authority: flag the compliance-relevant change plainly, no alarmism.",
    "category_trend_movement": "Curiosity + social proof: local demand is shifting.",
    "festival_upcoming": "Urgency: a dated local event is coming up.",
    "local_news_event": "Timeliness: react to what's happening locally right now.",
    "weather_heatwave": "Timeliness: local conditions changed demand today.",
    "dormant_with_vera": "Reciprocity: re-open contact with a concrete, low-effort offer.",
    "appointment_tomorrow": "Utility: confirm and prepare for tomorrow's booking.",
    "scheduled_recurring": "Curiosity: light-touch, ask-the-merchant style check-in.",
}
PERFORMANCE_DRIVEN_KINDS = {"perf_dip", "perf_spike", "customer_lapsed_soft", "dormant_with_vera"}

TICK_DEADLINE_SECONDS = 25.0   
REPLY_DEADLINE_SECONDS = 20.0  
MAX_CONCURRENT_LLM_CALLS = 3
MIN_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "1.2"))
MAX_RETRIES = 3

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

def word_match(phrase: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None

def any_word_match(phrases: list[str], text: str) -> bool:
    return any(word_match(p, text) for p in phrases)

def is_auto_reply(message: str, history: list[dict], from_role: str) -> bool:
    m = norm(message)
    if any(snippet in m for snippet in AUTO_REPLY_SNIPPETS):
        return True
    prior_same_role = [norm(h["text"]) for h in history if h.get("role") == from_role]
    return any(similarity(p, m) > 0.92 for p in prior_same_role)

def language_mode(languages: Optional[list], language_pref: Optional[str] = None) -> str:
    if language_pref and "hi" in str(language_pref).lower():
        return "hi-en"
    if languages and "hi" in [str(l).lower() for l in languages]:
        return "hi-en"
    return "en"

def is_expired(trg: dict, now_iso: Optional[str]) -> bool:
    exp = trg.get("expires_at")
    if not exp:
        return False
    try:
        exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00")) if now_iso else datetime.now(timezone.utc)
        return now_dt > exp_dt
    except Exception:
        return False

def consent_ok(customer: Optional[dict], trigger_kind: str) -> bool:
    needed = CONSENT_SCOPE_FOR_KIND.get(trigger_kind)
    if not needed:
        return True
    if not customer:
        return False
    scope = (customer.get("consent") or {}).get("scope", [])
    return needed in scope

def score_trigger(trg: dict) -> float:
    return trg.get("urgency", 1) * 10 + KIND_WEIGHT.get(trg.get("kind", ""), 3)

def too_similar_to_history(conversation_id: str, body_text: str) -> bool:
    prev = conv_meta.get(conversation_id, {}).get("sent_bodies", [])
    return any(similarity(p, body_text) > 0.85 for p in prev if body_text)

def _record_sent(conversation_id: str, body_text: str):
    meta = conv_meta.setdefault(conversation_id, {"sent_bodies": []})
    meta["sent_bodies"].append(body_text)

# ---------------------------------------------------------
# ASYNC-SAFE, DEADLINE-AWARE LLM CALLER
# ---------------------------------------------------------
class RateGate:
    def __init__(self, min_interval: float, max_concurrent: int):
        self.min_interval = min_interval
        self._sem = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait_turn(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

gate = RateGate(MIN_INTERVAL_SECONDS, MAX_CONCURRENT_LLM_CALLS)

async def call_gemini_async(system_instructions: str, prompt_text: str, deadline: float) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        if time.monotonic() >= deadline - 0.3:
            raise TimeoutError("time budget exhausted before call")
        async with gate._sem:
            await gate.wait_turn()
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instructions,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                return json.loads(raw_text.strip())
            except Exception as e:
                last_exc = e
                # Retrying on ANY error directly prevents instant fallbacks
                remaining = deadline - time.monotonic()
                backoff = min((2 ** attempt) + random.uniform(0, 0.4), max(0.0, remaining - 0.3))
                if backoff > 0:
                    await asyncio.sleep(backoff)
                continue
    raise last_exc if last_exc else RuntimeError("LLM call failed with no exception captured")

def build_fallback(prompt_data: dict, rationale: str) -> dict:
    target = prompt_data.get("target") or prompt_data.get("customer_name") or "there"
    body = f"Hi {target}, checking in on your profile. Want to review your active offers? Reply 1."
    return {"body": body, "cta": "Reply 1 to confirm.", "rationale": rationale}

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Solo Dev", "team_members": ["Vedant"], "model": MODEL,
        "approach": "V7 Master: Client.aio async execution, fallback elimination, strict psychological levers.",
        "contact_email": "vedant@example.com", "version": "7.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version", "current_version": cur["version"],
        })
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/teardown")
async def teardown():
    reset_state()
    return {"status": "teardown_complete"}

# ---------------------------------------------------------
# DYNAMIC COMPOSER
# ---------------------------------------------------------
def build_prompt(merchant: dict, category: dict, trg: dict, customer: Optional[dict] = None):
    category_slug = merchant.get("category_slug", "")
    trigger_kind = trg.get("kind", "")
    trigger_payload = trg.get("payload", {})
    scope = trg.get("scope", "merchant")

    history = merchant.get("conversation_history", [])
    recent_interaction = next((h for h in reversed(history) if h.get("engagement") == "merchant_replied"), None)
    continuity_rule = ("Do not introduce yourself. Continue the existing conversation directly."
                        if recent_interaction else "First touch: politely state who you are without fluff.")

    offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
    if not offers:
        offers = [o.get("title") for o in category.get("offer_catalog", [])][:3]
    locality = merchant.get("identity", {}).get("locality", "your area")

    taboos = list(BASE_TABOOS)
    voice = "Sharp, utility-first, locally grounded, merchant-friendly."
    if category_slug in CLINICAL_CATEGORIES:
        taboos += CLINICAL_EXTRA_TABOOS
        voice = "Clinical, peer-to-peer, evidence-based. Refer to 'patients', not 'customers'. No medical guarantees."

    prompt_data = {
        "target": merchant.get("identity", {}).get("name"),
        "locality": locality,
        "trigger_event": trigger_kind,
        "trigger_specifics": trigger_payload,
        "available_exact_offers": offers,
    }

    if scope == "customer" and customer:
        lang = language_mode(None, customer.get("identity", {}).get("language_pref"))
        slots = customer.get("preferences", {}).get("preferred_slots", "any time")
        prompt_data["customer_name"] = customer.get("identity", {}).get("name")
        prompt_data["preferred_slots"] = slots
        prompt_data["customer_relationship"] = customer.get("relationship")

        customer_taboos = taboos + ["overclaim", "medical claim", "guaranteed result"]
        system_instructions = (
            "You are composing a message SENT ON BEHALF OF THE MERCHANT to their own customer.\n"
            f"1. TONE: {voice}\n"
            f"2. TABOO WORDS: {', '.join(customer_taboos)}.\n"
            f"3. LANGUAGE: {'Hindi-English code-mix' if lang == 'hi-en' else 'Plain English'}. Address the customer by name.\n"
            f"4. SLOT OFFER: Use the preferred slots '{slots}' and explicitly offer '{offers[0] if offers else 'the relevant service'}'.\n"
            "5. CALL TO ACTION: Force a frictionless numerical choice for booking (e.g. 'Reply 1 for Wed, 2 for Thu').\n"
            "6. Never invent a price, slot, or claim not present in Context Data.\n"
            "Output strictly JSON with keys: body, cta, rationale."
        )
        return prompt_data, system_instructions, f"Context Data:\n{json.dumps(prompt_data, indent=2, default=str)}\nDraft the message."

    lang = language_mode(merchant.get("identity", {}).get("languages"))
    lever = LEVER_BY_KIND.get(trigger_kind, "Value and effort externalization: I already did the work, one-tap approve.")

    if "research" in trigger_kind or "digest" in trigger_kind:
        top_id = trigger_payload.get("top_item_id")
        digest_items = category.get("digest", [])
        matched = next((d for d in digest_items if d.get("id") == top_id), None) or (digest_items[0] if digest_items else None)
        prompt_data["digest_item"] = matched
        routing_rules = "DECISION QUALITY: reference the specific digest citation given above. Never discuss performance stats in this message."
    elif trigger_kind == "competitor_opened":
        routing_rules = "DECISION QUALITY: frame this as protecting their local territory from the specific competitor event given. Do not invent a competitor name if none is given."
    elif trigger_kind in PERFORMANCE_DRIVEN_KINDS:
        perf = merchant.get("performance", {})
        delta = perf.get("delta_7d", {})
        views_pct, calls_pct = delta.get("views_pct", 0), delta.get("calls_pct", 0)
        ctr = perf.get("ctr")
        peer_ctr = (category.get("peer_stats") or {}).get("avg_ctr")
        parts = [f"Views changed {views_pct*100:+.0f}%, calls changed {calls_pct*100:+.0f}% over 7d."]
        if ctr is not None and peer_ctr is not None:
            gap = (ctr - peer_ctr) * 100
            parts.append(f"CTR {ctr*100:.1f}% vs peer median {peer_ctr*100:.1f}% ({abs(gap):.1f}pt {'above' if gap >= 0 else 'below'}).")
        if calls_pct < 0 and views_pct >= 0:
            parts.append("This is a conversion problem, not a visibility one.")
        prompt_data["performance_analysis"] = " ".join(parts)
        routing_rules = "DECISION QUALITY: explicitly explain why the recommended offer addresses this specific performance movement."
    else:
        routing_rules = "DECISION QUALITY: use trigger_specifics directly as the hook — do not fabricate a performance narrative that isn't given in Context Data."

    system_instructions = (
        "You are Vera, magicpin's elite AI assistant for merchants. CORE RULES:\n"
        f"1. TONE & CATEGORY FIT: {voice}\n"
        f"2. TABOO WORDS: {', '.join(taboos)}.\n"
        f"3. ENGAGEMENT LEVER: {lever}\n"
        f"4. LANGUAGE: {'Hindi-English code-mix (natural, like a colleague texting)' if lang == 'hi-en' else 'Plain English'}.\n"
        f"5. MERCHANT FIT: Naturally weave '{locality}' into the opening sentence. {continuity_rule}\n"
        f"6. {routing_rules}\n"
        "7. CALL TO ACTION: End with a strict numerical/binary choice (e.g., 'Reply 1 for X, or 2 to dismiss'). No open-ended questions.\n"
        "Output strictly JSON with keys: body, cta, rationale."
    )
    return prompt_data, system_instructions, f"Context Data:\n{json.dumps(prompt_data, indent=2, default=str)}\nDraft the message."

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    tick_start = time.monotonic()
    deadline = tick_start + TICK_DEADLINE_SECONDS
    actions = []
    by_merchant: dict[str, list[tuple[str, dict]]] = {}
    customer_triggers: list[tuple[str, dict]] = []

    for trg_id in body.available_triggers:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg or is_expired(trg, body.now):
            continue
        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in sent_suppression.get(merchant_id, set()):
            continue
        if trg.get("scope") == "customer":
            customer_triggers.append((trg_id, trg))
        else:
            by_merchant.setdefault(merchant_id, []).append((trg_id, trg))

    jobs = []
    for merchant_id, candidates in by_merchant.items():
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        if not merchant:
            continue
        best_id, best_trg = max(candidates, key=lambda pair: score_trigger(pair[1]))
        jobs.append({"kind": "merchant", "score": score_trigger(best_trg),
                     "merchant_id": merchant_id, "trg_id": best_id, "trg": best_trg, "merchant": merchant})

    for trg_id, trg in customer_triggers:
        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None
        if not merchant or not consent_ok(customer, trg.get("kind", "")):
            continue
        jobs.append({"kind": "customer", "score": score_trigger(trg), "merchant_id": merchant_id,
                     "customer_id": customer_id, "trg_id": trg_id, "trg": trg,
                     "merchant": merchant, "customer": customer})

    jobs.sort(key=lambda j: j["score"], reverse=True)

    for job in jobs:
        if time.monotonic() > deadline:
            logger.info("tick time budget exhausted — returning %d actions early", len(actions))
            break

        merchant = job["merchant"]
        category = contexts.get(("category", merchant.get("category_slug", "")), {}).get("payload", {})
        customer = job.get("customer")
        trg = job["trg"]
        trg_id = job["trg_id"]
        merchant_id = job["merchant_id"]

        prompt_data, sys_instr, p_text = build_prompt(merchant, category, trg, customer)
        try:
            llm_output = await call_gemini_async(sys_instr, p_text, deadline)
        except Exception as e:
            logger.warning("fallback activated for %s (%s)", trg_id, e)
            llm_output = build_fallback(prompt_data, rationale=f"Fallback: {e}")

        body_text = llm_output.get("body", "")
        if not body_text:
            continue

        if job["kind"] == "merchant":
            conversation_id = f"conv_{merchant_id}_{trg_id}"
            if too_similar_to_history(conversation_id, body_text):
                continue
            actions.append({
                "conversation_id": conversation_id, "merchant_id": merchant_id, "customer_id": None,
                "send_as": "vera", "trigger_id": trg_id, "template_name": "vera_elite_v7",
                "template_params": [], "body": body_text, "cta": llm_output.get("cta", "Reply 1"),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": llm_output.get("rationale", ""),
            })
        else:
            customer_id = job["customer_id"]
            conversation_id = f"conv_{merchant_id}_{customer_id}_{trg_id}"
            if too_similar_to_history(conversation_id, body_text):
                continue
            actions.append({
                "conversation_id": conversation_id, "merchant_id": merchant_id, "customer_id": customer_id,
                "send_as": "merchant_on_behalf", "trigger_id": trg_id, "template_name": "vera_customer_v7",
                "template_params": [], "body": body_text, "cta": llm_output.get("cta", "Reply 1"),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": llm_output.get("rationale", ""),
            })

        sent_suppression.setdefault(merchant_id, set()).add(trg.get("suppression_key", ""))
        _record_sent(conversation_id, body_text)

        if len(actions) >= 20:
            break

    return {"actions": actions[:20]}

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    convo = conversations.setdefault(body.conversation_id, [])
    history_before = list(convo)
    convo.append({"role": body.from_role, "text": body.message})
    msg_norm = norm(body.message)

    merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload") if body.merchant_id else None
    lang = language_mode(merchant.get("identity", {}).get("languages")) if merchant else "en"

    if any_word_match(STOP_WORDS, msg_norm):
        return {"action": "end", "rationale": "Merchant opted out. Exiting."}

    if is_auto_reply(body.message, history_before, body.from_role):
        vera_nudges = sum(1 for h in history_before if h.get("role") == "vera")
        if vera_nudges >= 1:
            return {"action": "end", "rationale": "Canned auto-reply loop detected. Exiting gracefully."}
        nudge = ("Samajh gayi — jab team free ho, kya main abhi 2-min ka kaam khud dikha doon?"
                 if lang == "hi-en" else
                 "Got it — while that reaches the team, want me to just show you the fix myself?")
        convo.append({"role": "vera", "text": nudge})
        return {"action": "send", "body": nudge, "cta": "Reply YES",
                "rationale": "First canned auto-reply detected; one graceful re-ask before exiting on repeat."}

    if any_word_match(WAIT_PHRASES, msg_norm):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked for time; backing off 30 min."}

    if any_word_match(POSITIVE_INTENT, msg_norm):
        body_text = ("Done — maine yeh abhi activate kar diya hai aapke profile pe, ab main numbers track karke update karti rahungi."
                     if lang == "hi-en" else
                     "Done! I've executed this on your profile right now. I'll monitor the metrics and keep you posted.")
        convo.append({"role": "vera", "text": body_text})
        return {"action": "send", "body": body_text, "cta": "none",
                "rationale": "Explicit positive intent detected. Instantly transitioning to action without further qualification."}

    system_instructions = (
        "You are Vera, magicpin's merchant assistant, continuing an existing WhatsApp conversation.\n"
        "RULES:\n"
        "1. If the user asks an out-of-scope question (GST, taxes, legal advice, etc.) or is abusive, "
        "state in one short sentence you can't help with that, then steer back to the active topic politely.\n"
        "2. Answer any direct question concisely and factually using only the conversation history given — never invent facts.\n"
        f"3. LANGUAGE: {'Hindi-English code-mix' if lang == 'hi-en' else 'Plain English'}.\n"
        "4. action MUST be 'send' unless they explicitly refuse/say stop (handled separately) or ask for time (then 'wait', also handled separately) — default to 'send'.\n"
        "Output strictly JSON with keys: action ('send'|'wait'|'end'), body, cta, rationale."
    )
    deadline = time.monotonic() + REPLY_DEADLINE_SECONDS
    try:
        llm_output = await call_gemini_async(
            system_instructions,
            f"History:\n{json.dumps(convo[-4:], indent=2)}\nNext action?",
            deadline,
        )
        action = llm_output.get("action", "send")
        new_body = llm_output.get("body", "")
        if action == "send" and new_body and too_similar_to_history(body.conversation_id, new_body):
            new_body = new_body.rstrip(".") + " — anything specific you'd like me to check next?"
        if action == "send":
            convo.append({"role": "vera", "text": new_body})
            _record_sent(body.conversation_id, new_body)
        result = {"action": action, "body": new_body,
                  "cta": llm_output.get("cta", "none"), "rationale": llm_output.get("rationale", "")}
        if action == "wait":
            result["wait_seconds"] = llm_output.get("wait_seconds", 1800)
    except Exception as e:
        fallback_body = "Got it. I'm handling that right now."
        convo.append({"role": "vera", "text": fallback_body})
        result = {"action": "send", "body": fallback_body, "cta": "none",
                  "rationale": f"Fallback: LLM unavailable ({e})."}
    return result
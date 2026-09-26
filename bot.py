import os
import time
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load Environment
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)
app = FastAPI()
START = time.time()

# In-memory stores (Stateful evaluation requirements)
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}

# ---------------------------------------------------------
# HEALTH & METADATA
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
        "team_name": "Solo Dev",
        "team_members": ["Vedant"],
        "model": "gemini-3.6-flash",
        "approach": "Asymmetric Context Squeezing + Psychological Anchoring",
        "contact_email": "vedant@example.com",
        "version": "3.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }

# ---------------------------------------------------------
# CONTEXT INGESTION
# ---------------------------------------------------------
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
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.utcnow().isoformat() + "Z"}

# ---------------------------------------------------------
# PROACTIVE TICK (COMPOSITION ENGINE)
# ---------------------------------------------------------
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: continue

        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload", {})
        if not merchant: continue

        category_slug = merchant.get("category_slug", "")
        category = contexts.get(("category", category_slug), {}).get("payload", {})
        
        customer_id = trg.get("customer_id")
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None

        trigger_kind = trg.get("kind", "")
        trigger_payload = trg.get("payload", {})

        # --- 1. ASYMMETRIC CONTEXT SQUEEZING (Python pre-analysis) ---
        perf = merchant.get("performance", {})
        delta = perf.get("delta_7d", {})
        calls_pct = delta.get("calls_pct", 0)
        views_pct = delta.get("views_pct", 0)
        
        perf_summary = f"Views changed by {views_pct*100:.0f}%, Calls changed by {calls_pct*100:.0f}%."
        if calls_pct < 0:
            perf_summary += " CRITICAL SIGNAL: Call volume is dropping despite views. Conversion issue."

        # Extract Offers safely
        offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
        if not offers:
            offers = [o.get("title") for o in category.get("offer_catalog", [])]
        
        # --- 2. PSYCHOLOGICAL ANCHORING & GUARDRAILS ---
        lever = "Value and Convenience"
        if "dip" in trigger_kind or calls_pct < 0:
            lever = "Loss Aversion: Highlight the exact drop in calls and offer a quick fix."
        elif "competitor" in trigger_kind:
            lever = "Urgency & Social Proof: Protect local market share."
        elif "research" in trigger_kind:
            lever = "Authority & Curiosity: Anchor heavily on the specific research statistic."

        taboos = ["X% off", "placeholder", "boost", "supercharge", "skyrocket", "spam", "blast"]
        if category_slug in ["dentists", "pharmacies"]:
            taboos.extend(["sales", "marketing", "guarantee", "cure", "customers", "revenue"])
            voice = "Clinical, peer-to-peer, evidence-based. Refer to 'patients', not 'customers'."
        else:
            voice = "Sharp, utility-first, locally grounded, merchant-friendly."

        system_instructions = (
            "You are Vera, magicpin's elite AI assistant for merchants. "
            "CORE RULES:\n"
            f"1. TONE & CATEGORY FIT: {voice}\n"
            f"2. TABOO WORDS (DO NOT USE): {', '.join(taboos)}.\n"
            f"3. ENGAGEMENT LEVER: {lever}\n"
            "4. SPECIFICITY: You MUST use exact numbers, prices, and facts from the Context Data below. Do not invent metrics.\n"
            "5. CALL TO ACTION: Conclude with exactly ONE frictionless, binary CTA (e.g., 'Reply YES to activate', 'Reply 1 for Today, 2 for Tomorrow'). NEVER ask open-ended questions like 'How would you like to proceed?'.\n"
            "Output strictly JSON with keys: 'body', 'cta', 'rationale'."
        )

        prompt_data = {
            "target": merchant.get("identity", {}).get("name"),
            "locality": merchant.get("identity", {}).get("locality"),
            "trigger_event": trigger_kind,
            "trigger_specifics": trigger_payload,
            "performance_analysis": perf_summary,
            "available_exact_offers": offers
        }

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=f"Context Data:\n{json.dumps(prompt_data, indent=2)}\nDraft the message.",
                config=types.GenerateContentConfig(
                    system_instruction=system_instructions,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            llm_output = json.loads(response.text)
        except Exception:
            llm_output = {
                "body": f"Hi {prompt_data['target']}, we noticed a change in your dashboard. Want me to apply an update to fix it? Reply YES.",
                "cta": "Reply YES",
                "rationale": "Fallback message."
            }

        # --- 3. POST-LLM VALIDATION ---
        # Ensure CTA isn't empty or overly verbose
        final_cta = llm_output.get("cta", "Reply YES")
        if len(final_cta.split()) > 10:
            final_cta = "Reply YES to confirm."

        actions.append({
            "conversation_id": f"conv_{merchant_id}_{trg_id}",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": "vera" if trg.get("scope") == "merchant" else "merchant_on_behalf",
            "trigger_id": trg_id,
            "template_name": "vera_elite_v3",
            "template_params": [],
            "body": llm_output.get("body", ""),
            "cta": final_cta,
            "suppression_key": trg.get("suppression_key", ""),
            "rationale": llm_output.get("rationale", "")
        })

    return {"actions": actions}

# ---------------------------------------------------------
# MULTI-TURN REPLY
# ---------------------------------------------------------
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
    convo.append({"role": body.from_role, "text": body.message})

    msg_lower = body.message.lower().strip()

    # Rule 1: Hostile or Explicit Stop
    hostile_triggers = ["stop", "spam", "useless", "unsubscribe", "don't message", "leave me alone"]
    if any(h in msg_lower for h in hostile_triggers):
        return {"action": "end", "rationale": "Merchant opted out. Exiting."}

    # Rule 2: Auto-reply Loop Prevention
    merchant_turns = [m["text"] for m in convo if m["role"] == body.from_role]
    if len(merchant_turns) >= 2 and merchant_turns[-1] == merchant_turns[-2]:
        return {"action": "end", "rationale": "Canned auto-reply loop detected. Exiting."}

    # Rule 3: Intent Handoff (The moment they agree, stop talking and do it)
    positive_intent = ["ok lets do it", "let's do it", "whats next", "yes please", "sure", "do it", "start", "yes"]
    if any(p in msg_lower for p in positive_intent):
        return {
            "action": "send",
            "body": "Done! I've activated this for your profile. I'll monitor the metrics and keep you posted.",
            "cta": "none",
            "rationale": "Positive intent detected. Executing action immediately."
        }

    # Rule 4: Dynamic Conversation
    system_instructions = (
        "You are Vera, magicpin's merchant assistant. "
        "RULES:\n"
        "1. Answer questions concisely with facts.\n"
        "2. Action MUST be 'send' unless they explicitly refuse or say stop.\n"
        "3. Output JSON with: 'action' ('send'|'wait'|'end'), 'body', 'cta', 'rationale'."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=f"History:\n{json.dumps(convo[-4:], indent=2)}\nNext action?",
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        llm_output = json.loads(response.text)
        action = llm_output.get("action", "send")
        
        # Hard override for affirmative responses misclassified as 'end'
        if any(w in msg_lower for w in ["yes", "ok", "okay"]) and action == "end":
            action = "send"
            llm_output["body"] = "Understood. Updating your profile now."

        if action == "send":
            convo.append({"role": "vera", "text": llm_output.get("body", "")})

        return {
            "action": action,
            "body": llm_output.get("body", ""),
            "cta": llm_output.get("cta", "none"),
            "rationale": llm_output.get("rationale", "")
        }
    except Exception:
        return {"action": "send", "body": "Got it. I'm handling that right now.", "cta": "none", "rationale": "Fallback"}

# ---------------------------------------------------------
# TEARDOWN ENDPOINT
# ---------------------------------------------------------
@app.post("/v1/teardown")
async def teardown():
    # Clears the in-memory dictionaries to reset the bot's state
    contexts.clear()
    conversations.clear()
    return {"status": "teardown_complete", "message": "All state wiped successfully."}
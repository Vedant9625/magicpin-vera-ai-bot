# Magicpin AI Challenge: Vera Merchant Assistant
**Submitted by:** Vedant

## 1. Core Architecture & Philosophy
Building a bot that successfully passes JSON is the baseline. Building an AI that merchants actually trust requires a high signal-to-noise ratio, deterministic behavior, and zero marketing fluff. 

This implementation prioritizes **merchant psychology and latency constraints**, ensuring every message is highly specific, locally grounded, and strictly bounded by the 30-second `/v1/tick` timeout.

*   **Framework:** FastAPI (Python) for asynchronous, low-latency stateful handling.
*   **LLM Engine:** Gemini 3.6 Flash. Chosen specifically for its sub-second time-to-first-token (TTFT), guaranteeing compliance with the 20 actions/tick limit without dropping requests.
*   **Determinism:** `temperature = 0.0` is strictly enforced. Hallucinations in pricing, performance metrics, or local facts destroy merchant retention.
*   **Deployment:** Cloud-hosted via Render to ensure uninterrupted uptime for the judge harness.

## 2. Approach: Context-Grounded Specialization
Instead of feeding the LLM raw JSON payloads, this architecture uses a **Dynamic Context Router** to pre-process data before it reaches the model.

*   **Asymmetric Context Squeezing:** The Python layer intercepts the `merchant` and `trigger` scopes, calculates the exact deltas (e.g., isolating a negative `calls_pct` while ignoring static views), and synthesizes a dense, readable briefing document for the prompt.
*   **Psychological Anchoring:** Triggers are deterministically mapped to psychological engagement levers. Performance dips trigger *Loss Aversion* (highlighting the drop and offering a quick fix). Competitor actions trigger *Urgency & Social Proof*.
*   **Category-Specific Guardrails:** Taboo words (e.g., "boost", "supercharge", "X% off") are strictly banned. Clinical categories (`dentists`, `pharmacies`) are explicitly barred from using words like "sales" or "revenue," forcing a professional, peer-to-peer tone.

## 3. Defensive Engineering & Multi-Turn State
To ensure the bot survives hostile and out-of-distribution scenarios during the mid-test injections:

*   **O(1) Heuristic Guardrails:** Auto-reply loops and hostile triggers ("stop", "spam", "unsubscribe") are caught by hardcoded string matching. This immediately returns an `"end"` action, saving LLM latency and guaranteeing correct hostile handling.
*   **Instant Intent Handoff:** If a merchant replies with clear positive intent ("let's do it", "start", "yes please"), the system bypasses qualifying questions and instantly executes an action (`action: "send"`).
*   **The 1-Click CTA:** The LLM is hard-prompted to conclude every proactive message with a frictionless, binary Call to Action (e.g., "Reply YES to activate"). Open-ended questions are strictly prohibited.

## 4. State Management & Teardown
Contexts and conversations are maintained in-memory to maximize I/O speed. A `/v1/teardown` endpoint is implemented to clear dictionaries between evaluation runs, preventing state pollution.
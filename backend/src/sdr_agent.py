# freshworks_sdr_agent.py

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)

from livekit.plugins import google, murf, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv(".env.local")
logger = logging.getLogger("freshworks.sdr")
logger.setLevel(logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "FreshworksSDR", "freshworks_faq.json")
LEADS_FILE = os.path.join(BASE_DIR, "FreshworksSDR", "freshworks_leads.json")

SDR_VOICE = "Matthew"

def load_freshworks_faq():
    """Load Freshworks FAQ / product summary JSON."""
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"FAQ file missing at: {FAQ_FILE}")
        return None


def find_freshworks_answer_sync(query: str, kb: dict):
    """Ultra simple keyword-based FAQ retrieval."""
    q = query.lower()

    # Who it is for
    if any(x in q for x in ["who", "for which teams", "audience", "customer"]):
        return kb.get("target_audience", "Freshworks products serve teams of all sizes.")

    # Search structured FAQ
    for entry in kb.get("faq", []):
        if any(k in q for k in entry["keywords"]):
            return entry["answer"]

    return None


def save_lead_sync(lead_data: Dict[str, Any]):
    try:
        # If file does not exist → initialize it
        if not os.path.exists(LEADS_FILE):
            with open(LEADS_FILE, "w", encoding="utf-8") as f:
                json.dump({"leads": [lead_data]}, f, indent=4)
            return

        # If file exists → load safely
        with open(LEADS_FILE, "r+", encoding="utf-8") as f:
            try:
                content = f.read().strip()

                # If file is empty or whitespace
                if not content:
                    data = {"leads": []}
                else:
                    data = json.loads(content)

                # File corrupted OR missing "leads" field
                if "leads" not in data or not isinstance(data["leads"], list):
                    data = {"leads": []}

            except json.JSONDecodeError:
                # File corrupted — reset to fresh JSON
                data = {"leads": []}

            # Append new lead
            data["leads"].append(lead_data)

            # Write back clean JSON
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

        logger.info("Lead successfully saved.")

    except Exception as e:
        logger.error(f"Failed saving lead: {e}", exc_info=True)


@function_tool(name="freshworks_lookup_faq")
async def freshworks_lookup_faq(ctx: RunContext[dict], question: str):
    kb = load_freshworks_faq()
    if not kb:
        return {"response": "Our FAQ system is temporarily unavailable.", "status": "error"}

    # Thread safe blocking search
    answer = await asyncio.to_thread(find_freshworks_answer_sync, question, kb)

    if answer:
        return {"response": answer, "status": "answered"}

    return {
        "response":
            "Great question! I don't have that specific detail in my basic FAQ, "
            "but I can help you get connected to a product specialist. "
            "Would you like to proceed?",
        "status": "unanswered"
    }


@function_tool(name="freshworks_save_lead")
async def freshworks_save_lead(
    ctx: RunContext[dict],
    name: str,
    email: str,
    company: str,
    role: str,
    use_case: str,
    timeline: str,
):
    """Save final lead details once the call ends."""

    lead = {
        "name": name,
        "email": email,
        "company": company,
        "role": role,
        "use_case": use_case,
        "timeline": timeline,
        "timestamp": datetime.now().isoformat(),
    }

    await asyncio.to_thread(save_lead_sync, lead)

    summary = (
        f"Summary: {name} from {company}. "
        f"They’re exploring Freshworks for {use_case}, "
        f"and their timeline is {timeline}. We'll follow up soon!"
    )

    return {"status": "success", "verbal_summary": summary}


class FreshworksSDRAgent(Agent):

    def __init__(self):
        instructions = """
You are Optimus — a friendly, professional Sales Development Representative (SDR) for Freshworks.

Your objectives:
1. Warmly greet the user.
2. Ask what they are working on.
3. Ask follow-up questions to understand their needs.
4. Collect these 6 lead details naturally:
   - name
   - email
   - company
   - role
   - use case
   - timeline
5. When they ask questions:
   - Immediately use the `freshworks_lookup_faq` tool.
6. When user says they are done:
   - Call `freshworks_save_lead` with all collected fields.
   - Speak ONLY the verbal summary returned by the tool.
   - End the call politely.

Behavior:
- Always stay warm, crisp, accurate.
- Never invent product/pricing details.
- Keep Freshworks introduction short: 2–3 sentences max.
"""

        super().__init__(
            instructions=instructions,
            tools=[
                freshworks_lookup_faq,
                freshworks_save_lead,
            ]
        )


async def entrypoint(ctx: JobContext):
    logger.info("Launching Freshworks SDR Agent")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice=SDR_VOICE, style="Conversation", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
    )

    agent = FreshworksSDRAgent()

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    if not os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "w") as f:
            json.dump({"leads": []}, f, indent=4)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )

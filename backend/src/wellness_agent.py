# wellness_agent.py

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, List

from livekit.agents import (
    Agent,
    AgentSession,
    RunContext,
    JobContext,
    JobProcess,
    WorkerOptions,
    RoomInputOptions,
    function_tool,
    cli,
    tokenize,
)
from livekit.plugins import deepgram, google, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel, Field

from dotenv import load_dotenv

logger = logging.getLogger("wellness-agent")

load_dotenv(".env.local")


# ---------------------------------------------------------------------
# SYSTEM INSTRUCTIONS — grounded & wellness-safe
# ---------------------------------------------------------------------

WELLNESS_INSTRUCTIONS = """
You are a supportive, grounded, realistic health & wellness voice companion.
You speak naturally and warmly, like a caring daily check-in partner.

Your goals today:

1. Ask the user how they're feeling emotionally.
2. Ask about their energy level.
3. Ask what 1-3 small intentions or goals they want to set today.
4. Offer simple, realistic advice — NOT medical, NOT diagnostic.
5. At the end, summarize what they told you.
6. Then call the save_checkin tool with:
    - mood
    - energy
    - goals
    - a short summary sentence

Tone guidelines:
- Warm, gentle, concise.
- Never diagnose or imply medical expertise.
- Suggestions must be small, practical, and safe (e.g., a short walk, deep breaths, taking a break).
- Do NOT invent symptoms or conditions.
- Encourage self-awareness, not perfection.
- Use past JSON data if available (previous moods, goals, struggles).

Use past check-ins lightly:
Examples:
- "Last time you said your energy was low — how is it today?"
- "You mentioned stress last week. Has anything changed?"

Never claim to remember exact dates. Just reference themes.
"""

class CheckinData(BaseModel):
    mood: str
    energy: str
    goals: List[str]
    summary: str
    timestamp: str


@dataclass
class Userdata:
    past_checkins: List[CheckinData]

LOG_PATH = "wellness_log.json"


def load_past_checkins() -> List[CheckinData]:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [CheckinData(**entry) for entry in data]
    except Exception:
        return []


def save_checkin_to_file(entry: CheckinData):
    past = load_past_checkins()
    past.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in past], f, indent=4)


class WellnessAgent(Agent):
    def __init__(self, *, userdata: Userdata):
        last_note = ""
        if userdata.past_checkins:
            last = userdata.past_checkins[-1]
            last_note = (
                f"\nThe user's last mood was '{last.mood}' "
                f"and their energy was '{last.energy}'.\n"
                "Gently ask whether today feels similar or different."
            )

        instructions = WELLNESS_INSTRUCTIONS + last_note

        super().__init__(
            instructions=instructions,
            tools=[self.build_save_tool()],
        )

    def build_save_tool(self):
        @function_tool
        async def save_checkin(
            ctx: RunContext[Userdata],
            mood: Annotated[str, Field(description="User's mood today")],
            energy: Annotated[str, Field(description="User's energy level today")],
            goals: Annotated[List[str], Field(description="List of today's goals")],
            summary: Annotated[str, Field(description="Short summary of today's check-in")],
        ):
            entry = CheckinData(
                mood=mood,
                energy=energy,
                goals=goals,
                summary=summary,
                timestamp=datetime.now().isoformat(),
            )

            save_checkin_to_file(entry)
            ctx.userdata.past_checkins.append(entry)

            return "Your check-in has been saved."

        return save_checkin


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    past = load_past_checkins()
    userdata = Userdata(past_checkins=past)

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=WellnessAgent(userdata=userdata),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

# game_master_agent.py

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

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

# ---------------------------------------------------------------------
# Setup & logging
# ---------------------------------------------------------------------

load_dotenv(".env.local")

logger = logging.getLogger("gm.agent")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "gm_data")
os.makedirs(DATA_DIR, exist_ok=True)

STORY_LOG_PATH = os.path.join(DATA_DIR, "story_log.json")


# ---------------------------------------------------------------------
# Optional: simple story session logging (for fun / debugging)
# ---------------------------------------------------------------------

class StorySession(BaseModel):
    session_id: str
    ending_summary: str
    created_at: str


def load_past_sessions() -> List[StorySession]:
    if not os.path.exists(STORY_LOG_PATH):
        return []
    try:
        with open(STORY_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [StorySession(**entry) for entry in data]
    except Exception:
        logger.warning("Story log corrupted or unreadable, starting fresh.")
        return []


def save_story_session(entry: StorySession) -> None:
    past = load_past_sessions()
    past.append(entry)
    with open(STORY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in past], f, indent=4)


# ---------------------------------------------------------------------
# Userdata for this agent
# ---------------------------------------------------------------------

@dataclass
class Userdata:
    current_session_id: str
    past_sessions: List[StorySession]


# ---------------------------------------------------------------------
# SYSTEM INSTRUCTIONS — Game Master persona
# ---------------------------------------------------------------------

GAME_MASTER_INSTRUCTIONS = """
You are AURYN, a Dungeons & Dragons–style Game Master running a voice-first,
single-player adventure in the sci-fi fantasy universe **Shardfall**.

INTRO:
First introduce yourself in a single line: “I’m Auryn, your Game Master in Shardfall.”
Then ask the player to say **"Start"** to begin the adventure.

UNIVERSE:
- Floating islands, arcane tech, ancient crystal ruins.
- The player awakens with glowing crystal scars and no memory.
- Hover-skiffs, crystal-powered weapons, and AI oracles exist here.

TONE:
- Cinematic, adventurous, mysterious but friendly.
- Short, vivid narration suitable for spoken voice.

YOUR ROLE:
- Describe scenes in **2 short sentences max**.
- ALWAYS end with a **single** action prompt: “What do you do?”
- Never reveal game rules, tools, or that you are an AI.

MEMORY:
- Remember player’s chosen name.
- Track their major decisions, items, allies, enemies, places.

GAMEPLAY FLOW:
1️⃣ Introduce the first scene & ask their character name.  
2️⃣ Continue the story based on player responses.  
3️⃣ When a mini-arc completes (e.g., escape a ruin), offer a choice to continue or end.  
4️⃣ NEVER break role.

TOOLS:
- If the player says “restart”, “start over”, or “new game” → call `restart_story`
- If the player says “stop here” or “end session” → narratively wrap up and call `log_story_ending`

PROHIBITED:
- No rules explanations
- No long paragraphs
- No mentioning JSON, tools, or system messages

This is the only questions you asked to user, You will not ask anything else:

1️⃣
GM: “You wake on a floating island, wind roaring below. Your crystal scars pulse faintly.  
What do you do?”
Player: “Look around.”

2️⃣
GM: “A hovering drone flickers to life beside you. A robotic voice asks: ‘Name… please…?’  
What do you say?”
Player: “My name is Kai.”

3️⃣
GM: “A sky-pirate airship descends, its cannons glowing blue. They shout for you to drop your pack.  
Do you run or talk to them?”
Player: “I try to talk.”

4️⃣
GM: “The ground trembles—ancient gears grinding below. A stairway opens leading into darkness.  
Do you go down?”
Player: “Yes, cautiously.”

5️⃣
GM: “A crystal blade is lodged in a stone pedestal, humming with energy.  
Do you pull the weapon free?”
Player: “Yes!”

Always respond with short narration + **one** question:  
🎮 “What do you do?”
"""


# ---------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------

class GameMasterAgent(Agent):
    def __init__(self, *, userdata: Userdata):
        past = userdata.past_sessions
        last_note = ""
        if past:
            last = past[-1]
            last_note = (
                f"\nPrevious session note (for you, the GM): "
                f"the last adventure ended like this: '{last.ending_summary}'. "
                "You may gently reference that the player has adventured here before, "
                "but do not state exact dates or meta details."
            )

        instructions = GAME_MASTER_INSTRUCTIONS + last_note

        super().__init__(
            instructions=instructions,
            tools=[
                self.build_restart_tool(),
                self.build_log_story_ending_tool(),
            ],
        )

    # ---------------------------- tool: restart_story ----------------------------

    def build_restart_tool(self):
        @function_tool
        async def restart_story(
            ctx: RunContext[Userdata],
        ) -> str:
            """
            Reset the current adventure and begin a fresh story in the same universe.
            Use when the player says 'restart', 'new game', or similar.
            """
            # create a new session id
            ctx.userdata.current_session_id = datetime.utcnow().isoformat()
            # Tool result is for the GM (LLM) to see, not user-facing instructions.
            return (
                "The story has been restarted. Begin a brand new opening scene in Shardfall, "
                "as if this is the start of a new adventure. Briefly set the scene and then "
                "ask the player what they do."
            )

        return restart_story

    # ------------------------ tool: log_story_ending ----------------------------

    def build_log_story_ending_tool(self):
        @function_tool
        async def log_story_ending(
            ctx: RunContext[Userdata],
            ending_summary: Annotated[str, Field(description="Short summary of how this mini-arc or session ended.")],
        ) -> str:
            """
            Log a brief ending summary for this session's adventure.
            Use when the player finishes a mini-arc or chooses to end the session.
            """
            entry = StorySession(
                session_id=ctx.userdata.current_session_id,
                ending_summary=ending_summary,
                created_at=datetime.utcnow().isoformat(),
            )
            save_story_session(entry)
            ctx.userdata.past_sessions.append(entry)
            return "Story ending has been logged. You may offer a farewell or offer to restart the adventure."

        return log_story_ending


# ---------------------------------------------------------------------
# Prewarm & Entrypoint
# ---------------------------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    past_sessions = load_past_sessions()
    userdata = Userdata(
        current_session_id=datetime.utcnow().isoformat(),
        past_sessions=past_sessions,
    )

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-nikhil",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=GameMasterAgent(userdata=userdata),
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

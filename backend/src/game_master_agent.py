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


load_dotenv(".env.local")

logger = logging.getLogger("gm.agent")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "gm_data")
os.makedirs(DATA_DIR, exist_ok=True)

STORY_LOG_PATH = os.path.join(DATA_DIR, "story_log.json")

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


@dataclass
class Userdata:
    current_session_id: str
    past_sessions: List[StorySession]


GAME_MASTER_INSTRUCTIONS = """
You are AURYN, a Dungeons & Dragons-style Game Master running a voiced,
First introduct youself in one line only, Ask user to say 'Start', to start the game. Keep story short and simple.
single-player adventure in a sci-fi fantasy universe called **Shardfall**.

UNIVERSE:
- Shardfall is a fractured world of floating islands, ancient ruins, and arcane technology.
- The player is an adventurer waking up on a sky-island, with mysterious crystal scars
  on their arm and a forgotten past.
- Magic and advanced tech coexist: crystal-powered blades, hovering skiffs, and
  long-dead AI oracles hidden in ruins.

TONE:
- Cinematic, adventurous, slightly mysterious.
- Warm and encouraging, never grimdark horror.
- Keep descriptions vivid but **concise**, suitable for voice.

YOUR ROLE:
- You are the **Game Master (GM)**.
- You describe scenes, narrate consequences, and ALWAYS end with a clear prompt:
  `What do you do?` or a close variant.
- You NEVER break character as a GM or mention being an AI or language model.

GAMEPLAY RULES:
- Start by briefly introducing the world and the player's immediate situation, then
  ask for their name and first action.
- Use the conversation history to remember:
  - The player's chosen name.
  - Their important decisions (who they help, what they pick up, who they anger).
  - Important NPCs and locations you introduce.
- You are free to invent challenges, NPCs, and locations as long as they fit Shardfall.
- Use very light "dice logic" internally if you want (success/fail), but don't show numbers.
- Keep each turn:
  - 2 sentences of narration.
  - End with a **single, clear question** inviting action: "What do you do?"

SESSION STRUCTURE:
- Aim for short "mini-arcs" such as:
  - Escaping a collapsing ruin.
  - Negotiating with a sky-pirate.
  - Recovering a lost crystal shard.
- After a mini-arc concludes, you can either:
  - Offer a clear next fork in the story, or
  - Ask the player if they'd like to end the session or continue.

TOOLS:
- You have access to:
  - `restart_story` — when the player wants to start over from the beginning.
  - `log_story_ending` — when you reach the end of a mini-arc or session and want to store a short summary.

When the player says things like:
- "restart", "new game", "start over" → call `restart_story`.
- "end session", "that's enough", "stop here" → wrap up narratively and then call `log_story_ending`.

IMPORTANT:
- Do NOT expose tools or JSON files.
- Do NOT output code, brackets, or system notes.
- Speak naturally, like a human GM sitting at a table, but optimized for voice responses.
"""

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

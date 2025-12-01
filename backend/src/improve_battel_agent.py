# improv_battle_agent.py

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from pydantic import Field

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


# -------------------------------------------------------------------
# Logging + Paths
# -------------------------------------------------------------------

load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("improv")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "improv_data")
os.makedirs(DATA_DIR, exist_ok=True)
SCENARIOS_PATH = os.path.join(DATA_DIR, "improv_scenarios.json")


# -------------------------------------------------------------------
# Default Scenarios
# -------------------------------------------------------------------

DEFAULT_SCENARIOS = [
    "You are a time-travelling tour guide explaining smartphones to someone from 1800s.",
    "You are a barista explaining a latte that leads to another dimension.",
    "You are a waiter explaining that their order escaped the kitchen.",
    "You are a space mechanic using duct tape to fix a spaceship.",
    "You are trying to return a cursed object to a skeptical shopkeeper.",
    "You are a superhero who can only make things slightly warmer.",
]


def load_scenarios() -> List[str]:
    if not os.path.exists(SCENARIOS_PATH):
        with open(SCENARIOS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SCENARIOS, f, indent=2)
        return DEFAULT_SCENARIOS

    try:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(s) for s in data]
    except:
        return DEFAULT_SCENARIOS


# -------------------------------------------------------------------
# State Models
# -------------------------------------------------------------------


@dataclass
class ImprovRoundState:
    scenario: str
    host_reaction: Optional[str] = None


@dataclass
class ImprovState:
    player_name: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 3
    phase: str = "intro"
    rounds: List[ImprovRoundState] = field(default_factory=list)
    ended_at: Optional[str] = None


@dataclass
class Userdata:
    improv_state: ImprovState
    scenarios: List[str]


# -------------------------------------------------------------------
# System Instructions
# -------------------------------------------------------------------

IMPROV_HOST_INSTRUCTIONS = """
You are **HYPE-MIC**, the high-energy host of the TV improv show "Improv Battle"!

Your job:
- Welcome the player using their name if known
- Run 3 improv rounds
- Give the player a fun scenario each round
- After they perform, react differently each time (funny, teasing, praising, etc.)
- Keep it short: 2–3 sentences per message
- Always end with a prompt like “What do you do?” or “Go for it!”

Flow:
1️⃣ Introduction → Ask player name → Explain game
2️⃣ Round N:
    ✔ Announce scenario
    ✔ Encourage performance (“Say your lines!”)
    ✔ After they finish (they say "end scene"):
        - Give unique reaction
        - Call log_round tool
3️⃣ After final round:
    - Give fun summary of their performance
    - Call end_show tool
"""


# -------------------------------------------------------------------
# Agent Implementation
# -------------------------------------------------------------------


class ImprovBattleAgent(Agent):
    def __init__(self, *, userdata: Userdata):
        instructions = (
            IMPROV_HOST_INSTRUCTIONS
            + "\nExample scenarios:\n"
            + "\n".join(f"- {s}" for s in userdata.scenarios[:4])
        )

        super().__init__(
            instructions=instructions,
            tools=[
                self.build_log_round_tool(),
                self.build_end_show_tool(),
            ],
        )

    # Tool: Store reaction + scenario
    def build_log_round_tool(self):
        @function_tool
        async def log_round(
            ctx: RunContext[Userdata],
            round_index: Annotated[int, Field(description="Round number, 0-based")],
            scenario: Annotated[str, Field(description="Scenario text")],
            host_reaction: Annotated[str, Field(description="Short reaction summary")],
        ) -> str:
            state = ctx.userdata.improv_state

            while len(state.rounds) <= round_index:
                state.rounds.append(ImprovRoundState(scenario=""))

            current = state.rounds[round_index]
            current.scenario = scenario
            current.host_reaction = host_reaction

            state.current_round = max(state.current_round, round_index + 1)

            logger.info(f"Round {round_index} logged")

            return "Round captured. Continue gameplay."

        return log_round

    # Tool: End the show
    def build_end_show_tool(self):
        @function_tool
        async def end_show(
            ctx: RunContext[Userdata],
            final_summary: Annotated[str, Field(description="Last reaction summary")],
        ) -> str:
            state = ctx.userdata.improv_state
            state.phase = "done"
            state.ended_at = datetime.now(timezone.utc).isoformat()

            logger.info("Final Summary: %s", final_summary)

            return "Show finished."

        return end_show


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    scenarios = load_scenarios()

    # -----------------------------------------------------------------
    # 1️⃣ Load playerName BEFORE session.start()
    # -----------------------------------------------------------------
    player_name = "Player"
    try:
        # Access metadata safely after connect — but before agent starts responding
        await ctx.connect()  # 👈 Connect early to read metadata
        metadata_json = ctx.room.local_participant.metadata
        print("Metadata JSON: ", metadata_json)

        if metadata_json:
            meta = json.loads(metadata_json)
            player_name = meta.get("playerName", "Player")

    except Exception as e:
        logger.warning(f"Metadata parsing failed: {e}")

    logger.info(f"Player Name Resolved: {player_name}")

    # Initialize full state AFTER we know player name
    improv_state = ImprovState(player_name=player_name)
    userdata = Userdata(
        improv_state=improv_state,
        scenarios=scenarios,
    )

    # -----------------------------------------------------------------
    # 2️⃣ Create and Start Agent Session
    # -----------------------------------------------------------------
    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-Nikhil",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )

    await session.start(
        agent=ImprovBattleAgent(userdata=userdata),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    logger.info("🎤 Improv Battle Agent Ready for Voice Flow!")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

# day4_tutor_agent.py
import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, List, Dict

from pydantic import BaseModel, Field

# LiveKit imports
from livekit.agents import (
    Agent,
    AgentSession,
    RunContext,
    JobContext,
    JobProcess,
    function_tool,
    cli,
    WorkerOptions,
    tokenize,
    RoomInputOptions,
)
from livekit.plugins import deepgram, murf, silero, noise_cancellation, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dotenv import load_dotenv

logger = logging.getLogger("day4-tutor")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_PATH = os.path.join(BASE_DIR, "shared-data", "day4_tutor_content.json")
PROGRESS_PATH = os.path.join(BASE_DIR, "shared-data", "day4_tutor_progress.json")

class Concept(BaseModel):
    id: str
    title: str
    summary: str
    sample_question: str

class ProgressRecord(BaseModel):
    concept_id: str
    timestamp: str
    score: float
    feedback: str

@dataclass
class Userdata:
    content: List[Concept]
    progress: Dict[str, List[ProgressRecord]]

def load_content() -> List[Concept]:
    with open(CONTENT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Concept(**c) for c in raw]

def load_progress() -> Dict[str, List[ProgressRecord]]:
    if not os.path.exists(PROGRESS_PATH):
        return {}
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cid: [ProgressRecord(**r) for r in records] for cid, records in raw.items()}

def save_progress(progress: Dict[str, List[ProgressRecord]]):
    serializable = {cid: [r.dict() for r in records] for cid, records in progress.items()}
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)


def evaluate_teach_back(concept: Concept, user_text: str):
    import re
    def words(s):
        return set(re.findall(r"\b[a-zA-Z0-9]+\b", s.lower()))

    concept_words = words(concept.summary)
    user_words = words(user_text)
    if not concept_words:
        return 0.0, "No summary found for evaluation."

    ratio = len(concept_words.intersection(user_words)) / len(concept_words)
    score = round(ratio * 100, 1)

    if score >= 80:
        feedback = "Great explanation — you covered the key points!"
    elif score >= 50:
        feedback = "Good attempt — you captured several important ideas."
    elif score >= 25:
        feedback = "You covered a few basics but missed important parts."
    else:
        feedback = "Not much overlap — try summarizing the main purpose next time."

    return score, feedback


VOICE_MAP = {
    "learn":  {"voice": "en-US-matthew", "style": "Conversation"},
    "quiz":   {"voice": "en-US-alicia",  "style": "Conversation"},
    "teach":  {"voice": "en-US-ken",     "style": "Conversation"},
}

def switch_voice(session: AgentSession, mode: str):
    """Switches Murf Falcon TTS voice using official update_options()."""
    try:
        voice_info = VOICE_MAP.get(mode)
        if not voice_info:
            logger.warning(f"No voice mapped for mode: {mode}")
            return

        tts_engine = session.tts
        if hasattr(tts_engine, "update_options"):
            tts_engine.update_options(
                voice=voice_info["voice"],
                style=voice_info["style"],
            )
            logger.info(f"✔ Voice switched to {voice_info['voice']} for mode {mode}")
        else:
            logger.warning("TTS engine does not support update_options()")

    except Exception as e:
        logger.error(f"Voice switching failed: {e}")


ROUTER_PROMPT = """
You are Tutor — a friendly teach-the-tutor voice assistant that supports three modes:
- learn  : explain a concept (short summary).
- quiz   : ask a sample question about the concept.
- teach_back : ask the user to explain the concept back, then give feedback and a score.

Flow:
1) Greet the user and ask which mode they'd like: learn, quiz, or teach_back.
2) Ask which concept (by id or title) they want; show the available concepts if needed.
3) In learn mode: present the concept.summary (short).
4) In quiz mode: ask the concept.sample_question.
5) In teach_back mode: ask the user to explain the concept; then call the tool teach_back_evaluate to score and store progress.
6) Allow switching modes anytime. If intent unclear, ask a single clarifying question: "Do you want to learn, be quizzed, or teach back?"

Voice guidance:
- When returning from the learn tool, prefer Murf Falcon voice 'Matthew'.
- When returning from the quiz tool, prefer Murf Falcon voice 'Alicia'.
- When returning from teach_back evaluation, prefer Murf Falcon voice 'Ken'.

Safety:
- Keep explanations concise, clear, and non-technical when possible.
- This is a study helper, not a subject-matter authority.
"""

class Day4TutorAgent(Agent):
    def __init__(self, userdata: Userdata):
        concept_list = "\n".join(f"- {c.id}: {c.title}" for c in userdata.content)
        instructions = ROUTER_PROMPT + "\nAvailable concepts:\n" + concept_list

        super().__init__(instructions=instructions, tools=[
            self.learn_tool(),
            self.quiz_tool(),
            self.teach_back_tool(),
        ])

    # -------------------------
    # Learn mode
    # -------------------------
    def learn_tool(self):
        @function_tool
        async def learn_concept(ctx: RunContext[Userdata], concept_id: Annotated[str, Field(...)]):
            switch_voice(ctx.session, "learn")

            match = next((c for c in ctx.userdata.content if c.id == concept_id), None)
            if not match:
                return "Concept not found."

            ctx.session.say(f"{match.title}: {match.summary}")
            return ""

        return learn_concept

    # -------------------------
    # Quiz mode
    # -------------------------
    def quiz_tool(self):
        @function_tool
        async def quiz_concept(ctx: RunContext[Userdata], concept_id: Annotated[str, Field(...)]):
            switch_voice(ctx.session, "quiz")

            match = next((c for c in ctx.userdata.content if c.id == concept_id), None)
            if not match:
                return "Concept not found."

            ctx.session.say(f"Quiz time! {match.sample_question}")
            return ""

        return quiz_concept

    # -------------------------
    # Teach-back mode
    # -------------------------
    def teach_back_tool(self):
        @function_tool
        async def teach_back_evaluate(
            ctx: RunContext[Userdata], concept_id: str, user_explanation: str
        ):
            switch_voice(ctx.session, "teach")

            match = next((c for c in ctx.userdata.content if c.id == concept_id), None)
            if not match:
                return "Concept not found."

            score, feedback = evaluate_teach_back(match, user_explanation)

            record = ProgressRecord(
                concept_id=match.id,
                timestamp=datetime.now().isoformat(),
                score=score,
                feedback=feedback,
            )
            ctx.userdata.progress.setdefault(match.id, []).append(record)
            save_progress(ctx.userdata.progress)

            ctx.session.say(
                f"Your score for {match.title} is {score}. {feedback}"
            )
            return ""

        return teach_back_evaluate

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    content = load_content()
    progress = load_progress()
    userdata = Userdata(content=content, progress=progress)

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Conversation", model="Falcon"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Day4TutorAgent(userdata),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
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

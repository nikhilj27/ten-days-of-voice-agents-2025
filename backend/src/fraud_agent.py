# fraud_agent.py

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    RoomInputOptions,
    function_tool,
    cli,
    tokenize,
)

from livekit.plugins import deepgram, google, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

logger = logging.getLogger("fraud.agent")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAUD_DB_PATH = os.path.join(BASE_DIR, "fraud_data", "fraud_cases.json")

VOICE_ID = "en-US-matthew"     # Calm, professional voice
VOICE_STYLE = "Conversation"

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------

class FraudCase(BaseModel):
    case_id: str
    username: str                     # username to look up the case
    customer_name: str
    security_id: str                  # fake security identifier
    masked_card: str                  # e.g. "**** 1234"
    transaction_amount: float
    merchant_name: str
    location: str
    timestamp: str
    security_question: str
    security_answer: str
    status: str                       # e.g. "pending_review", "confirmed_safe", "confirmed_fraud", "verification_failed"
    outcome_note: Optional[str] = None


@dataclass
class FraudUserdata:
    cases: List[FraudCase]
    current_case: Optional[FraudCase] = None
    verification_passed: bool = False


# -------------------------------------------------------------------
# Seed DB helper
# -------------------------------------------------------------------

SEED_FRAUD_CASES: List[dict] = [
    {
        "case_id": "FRD-1001",
        "username": "arjun_98",
        "customer_name": "Arjun Mehta",
        "security_id": "SEC-ARJ-9921",
        "masked_card": "**** 4321",
        "transaction_amount": 8299.50,
        "merchant_name": "ElectroHub Online",
        "location": "Bangalore, India",
        "timestamp": "2025-01-15T18:42:00+05:30",
        "security_question": "What is your favourite sport?",
        "security_answer": "cricket",
        "status": "pending_review",
        "outcome_note": None,
    },
    {
        "case_id": "FRD-1002",
        "username": "priya_singh",
        "customer_name": "Priya Singh",
        "security_id": "SEC-PRI-7743",
        "masked_card": "**** 5577",
        "transaction_amount": 14999.00,
        "merchant_name": "Global Travel Mart",
        "location": "Delhi, India",
        "timestamp": "2025-02-03T10:17:00+05:30",
        "security_question": "What city were you born in?",
        "security_answer": "jaipur",
        "status": "pending_review",
        "outcome_note": None,
    },
    {
        "case_id": "FRD-1003",
        "username": "nikhil.j",
        "customer_name": "Nikhil Jadhav",
        "security_id": "SEC-RAH-6610",
        "masked_card": "**** 9012",
        "transaction_amount": 2199.0,
        "merchant_name": "FoodKart",
        "location": "Mumbai, India",
        "timestamp": "2025-02-10T22:05:00+05:30",
        "security_question": "What is your favourite color?",
        "security_answer": "blue",
        "status": "pending_review",
        "outcome_note": None
    },
]


def bootstrap_fraud_db():
    """Create fraud_data/fraud_cases.json with 3 demo cases if missing or invalid."""
    os.makedirs(os.path.dirname(FRAUD_DB_PATH), exist_ok=True)

    if not os.path.exists(FRAUD_DB_PATH) or os.path.getsize(FRAUD_DB_PATH) == 0:
        logger.info("Bootstrapping fraud DB with seed cases...")
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(SEED_FRAUD_CASES, f, indent=4)
        return

    # If file exists but is corrupted / invalid JSON, reset with seed
    try:
        with open(FRAUD_DB_PATH, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        logger.warning("Fraud DB file invalid; resetting with seed cases.")
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(SEED_FRAUD_CASES, f, indent=4)


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def load_all_cases() -> List[FraudCase]:
    try:
        with open(FRAUD_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [FraudCase(**item) for item in raw]
    except FileNotFoundError:
        logger.error("Fraud DB file not found, returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Failed to load fraud DB: {e}", exc_info=True)
        return []


def save_all_cases(cases: List[FraudCase]) -> None:
    try:
        os.makedirs(os.path.dirname(FRAUD_DB_PATH), exist_ok=True)
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([c.dict() for c in cases], f, indent=4)
        logger.info("Fraud DB updated successfully.")
    except Exception as e:
        logger.error(f"Failed to save fraud DB: {e}", exc_info=True)


def update_case_in_db(updated: FraudCase) -> None:
    cases = load_all_cases()
    found = False
    for idx, c in enumerate(cases):
        if c.case_id == updated.case_id:
            cases[idx] = updated
            found = True
            break
    if not found:
        cases.append(updated)
    save_all_cases(cases)


# -------------------------------------------------------------------
# System instructions (persona + flow)
# -------------------------------------------------------------------

FRAUD_AGENT_INSTRUCTIONS = """
You are a calm, professional fraud detection representative for a fictional Indian bank called "SafeGuard Bank".

Your job during this call:

1. Greet the user and clearly introduce yourself as Fred:
   - Mention SafeGuard Bank.
   - Explain you are calling about a suspicious card transaction.
   - Speak calmly, clearly, and be reassuring.

2. Ask for the customer's username early in the call.
   - Once they provide it, call the `load_fraud_case` tool.
   - If no case is found, gently say that you cannot find an active alert and ask them to confirm the username or end the call.

3. After a case is loaded:
   - Perform a simple security verification using the security question from the loaded case.
   - Ask the security question exactly, and when the user answers, call `verify_security_answer`.
   - If verification fails, explain politely that you cannot proceed further and end the call.

4. If verification succeeds:
   - Clearly summarise the suspicious transaction using the loaded case:
     - Merchant
     - Amount
     - Masked card
     - Location
     - Approximate time
   - Ask: "Did you make this transaction?" and wait for a yes/no style answer.

5. When you understand whether the user did or did not make the transaction:
   - Call `record_fraud_decision`:
     - user_confirmed_transaction = true if they say it's legitimate.
     - user_confirmed_transaction = false if they deny it.
     - extra_note: a short summary of what the user said (optional).

6. Speak a short, clear closing message:
   - If confirmed_safe:
       - Explain that the transaction is marked as safe and no block is placed.
   - If confirmed_fraud:
       - Explain that you've blocked the card and started a dispute process (mock).
   - If verification_failed:
       - Explain that you could not verify identity and cannot discuss the transaction further.

Important safety & behavior rules:
- NEVER ask for full card numbers, PINs, OTPs, passwords, or credentials.
- ONLY use the security question from the database as verification.
- Keep answers grounded in the data returned by the tools.
- Keep messages short and conversational.
- If the user seems confused, re-explain in simple language.
"""


# -------------------------------------------------------------------
# Agent implementation
# -------------------------------------------------------------------

class FraudAlertAgent(Agent):
    """Fraud alert voice agent using JSON 'DB' for cases."""

    def __init__(self, *, userdata: FraudUserdata):
        super().__init__(
            instructions=FRAUD_AGENT_INSTRUCTIONS,
            tools=[
                self.build_load_case_tool(),
                self.build_verify_tool(),
                self.build_record_decision_tool(),
            ],
        )

    # -----------------------------
    # Tool: load case by username
    # -----------------------------
    def build_load_case_tool(self):
        @function_tool
        async def load_fraud_case(
            ctx: RunContext[FraudUserdata],
            username: Annotated[str, Field(description="Customer username to look up fraud case")],
        ) -> str:
            uname = username.strip().lower()
            match = next(
                (c for c in ctx.userdata.cases if c.username.lower() == uname),
                None,
            )

            if not match:
                return (
                    "No active fraud case found for this username. "
                    "You should politely say you cannot see any current fraud alerts "
                    "and ask the user to confirm details or contact support."
                )

            ctx.userdata.current_case = match
            ctx.userdata.verification_passed = False

            # Guidance text for the LLM
            return (
                f"Fraud case loaded for {match.customer_name} (username: {match.username}). "
                f"Security question: {match.security_question}. "
                f"Suspicious transaction: INR {match.transaction_amount} at {match.merchant_name} "
                f"in {match.location} on {match.timestamp}, card {match.masked_card}. "
                f"Current status: {match.status}. "
                "Now, ask the user ONLY the security question to verify their identity."
            )

        return load_fraud_case

    # -----------------------------
    # Tool: verify security answer
    # -----------------------------
    def build_verify_tool(self):
        @function_tool
        async def verify_security_answer(
            ctx: RunContext[FraudUserdata],
            user_answer: Annotated[str, Field(description="User's answer to the security question")],
        ) -> str:
            case = ctx.userdata.current_case
            if not case:
                return "No active case is loaded. Ask for a username and call load_fraud_case first."

            expected = case.security_answer.strip().lower()
            given = user_answer.strip().lower()

            if given == expected:
                ctx.userdata.verification_passed = True
                return (
                    "verified: The security answer matches. "
                    "You can now explain the suspicious transaction details and ask if they made it."
                )
            else:
                ctx.userdata.verification_passed = False
                return (
                    "failed: The security answer does not match. "
                    "Politely say you cannot continue discussing the account and end the call."
                )

        return verify_security_answer

    # -----------------------------
    # Tool: record decision & update DB
    # -----------------------------
    def build_record_decision_tool(self):
        @function_tool
        async def record_fraud_decision(
            ctx: RunContext[FraudUserdata],
            user_confirmed_transaction: Annotated[
                bool,
                Field(
                    description=(
                        "True if the user confirms the suspicious transaction is legitimate, "
                        "False if they say they did NOT make this transaction."
                    )
                ),
            ],
            extra_note: Annotated[
                Optional[str],
                Field(
                    description=(
                        "Optional short note summarizing what the customer said about the transaction."
                    )
                ),
            ] = None,
        ) -> str:
            case = ctx.userdata.current_case
            if not case:
                return "No active case to update."

            # If verification never passed, mark as verification_failed
            if not ctx.userdata.verification_passed:
                case.status = "verification_failed"
                case.outcome_note = "Verification failed; could not confirm customer identity."
                update_case_in_db(case)
                logger.info("Case %s marked verification_failed", case.case_id)
                return (
                    "verification_failed: Inform the customer that you could not verify their identity "
                    "and cannot discuss the transaction further."
                )

            if user_confirmed_transaction:
                case.status = "confirmed_safe"
                base_note = "Customer confirmed transaction as legitimate. No block placed."
            else:
                case.status = "confirmed_fraud"
                base_note = (
                    "Customer denied the transaction. Card blocked and dispute process (mock) started."
                )

            if extra_note:
                case.outcome_note = f"{base_note} Extra note: {extra_note}"
            else:
                case.outcome_note = base_note

            update_case_in_db(case)
            logger.info("Case %s updated to %s", case.case_id, case.status)

            return (
                f"{case.status}: {base_note} "
                "You should give a brief closing message summarizing this action to the customer."
            )

        return record_fraud_decision


# -------------------------------------------------------------------
# Prewarm & entrypoint
# -------------------------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Ensure DB exists & is valid
    bootstrap_fraud_db()
    cases = load_all_cases()
    userdata = FraudUserdata(cases=cases)

    session = AgentSession[FraudUserdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_ID,
            style=VOICE_STYLE,
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    agent = FraudAlertAgent(userdata=userdata)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    # Make sure directory exists even before worker starts
    bootstrap_fraud_db()

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

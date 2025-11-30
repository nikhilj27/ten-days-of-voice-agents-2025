# acp_shopping_agent.py

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Annotated, List, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    RoomInputOptions,
    function_tool,
    cli,
    tokenize,
)
from livekit.plugins import deepgram, murf, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

load_dotenv(".env.local")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "acp_data")
os.makedirs(DATA_DIR, exist_ok=True)

CATALOG_PATH = os.path.join(DATA_DIR, "shopping_catalog.json")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.json")


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------


class Product(BaseModel):
    id: str
    name: str
    price: float
    currency: str
    category: str
    color: Optional[str] = None
    size: Optional[List[str]] = None


class Order(BaseModel):
    order_id: str
    items: List[Dict]
    total: float
    currency: str
    created_at: str


@dataclass
class Userdata:
    catalog: List[Product]
    orders: List[Order]


# ---------------------------------------------------------------------
# DB Helpers
# ---------------------------------------------------------------------


def load_catalog() -> List[Product]:
    with open(CATALOG_PATH, "r") as f:
        raw = json.load(f)
    return [Product(**p) for p in raw]


def load_orders() -> List[Order]:
    if not os.path.exists(ORDERS_PATH):
        return []
    try:
        with open(ORDERS_PATH, "r") as f:
            raw = json.load(f).get("orders", [])
        return [Order(**o) for o in raw]
    except:
        return []


def save_orders(orders: List[Order]):
    with open(ORDERS_PATH, "w") as f:
        json.dump({"orders": [o.dict() for o in orders]}, f, indent=4)


# ---------------------------------------------------------------------
# Conversation Instructions
# ---------------------------------------------------------------------

ACP_INSTRUCTIONS = """
You are KARTY — a friendly AI shopping assistant.

Help users browse items and place orders.
Respond naturally, never mention tools, JSON, or backend logic.
Always ask if they want anything else after helping.
"""


# ---------------------------------------------------------------------
# ACP Shopping Agent
# ---------------------------------------------------------------------


class ACPShoppingAgent(Agent):
    def __init__(self, *, userdata: Userdata):
        super().__init__(
            instructions=ACP_INSTRUCTIONS,
            tools=[
                self.tool_list_products(),
                self.tool_create_order(),
                self.tool_get_last_order(),
            ],
        )

    # -----------------------------------------------------------------
    # Tool: List / filter product catalog
    # -----------------------------------------------------------------

    def tool_list_products(self):
        @function_tool
        async def list_products(
            ctx: RunContext[Userdata],
            name: Annotated[Optional[str], Field(default=None, description="Search match in product name")] = None,
            category: Annotated[Optional[str], Field(default=None, description="Category filter")] = None,
            max_price: Annotated[Optional[float], Field(default=None, description="Max price filter")] = None,
            color: Annotated[Optional[str], Field(default=None, description="Color filter")] = None,
        ) -> str:

            products = ctx.userdata.catalog

            # Basic filtering
            if name:
                query = (name or "").strip().lower()
                products = [
                    p
                    for p in products
                    if query in p.name.lower()
                    or any(query in t.lower() for t in (p.tags or []))
                ]
            if category:
                products = [
                    p for p in products if p.category.lower() == category.lower()
                ]
            if max_price:
                products = [p for p in products if p.price <= max_price]
            if color:
                products = [
                    p for p in products if p.color and p.color.lower() == color.lower()
                ]

            if not products:
                return "No products match — want to try different filters?"

            # Short voice-friendly output
            out = ", ".join(f"{p.name} (ID:{p.id}) ₹{p.price}" for p in products[:5])
            return f"I found: {out}. Want one of these?"

        return list_products

    # -----------------------------------------------------------------
    # Tool: Create Order (fixed schema!)
    # -----------------------------------------------------------------

    def tool_create_order(self):
        @function_tool
        async def create_order(
            ctx: RunContext[Userdata],
            product_id: Annotated[str, Field(description="Product ID")],
            quantity: Annotated[int, Field(description="Quantity requested")],
        ) -> str:

            # Manual validation instead of schema enforcement
            if quantity <= 0:
                return "Quantity must be at least one!"

            product = next(
                (p for p in ctx.userdata.catalog if p.id == product_id), None
            )
            if not product:
                return "That product seems unavailable now!"

            total = product.price * quantity
            order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"

            order = Order(
                order_id=order_id,
                items=[{"id": product_id, "qty": quantity}],
                total=total,
                currency="INR",
                created_at=datetime.utcnow().isoformat(),
            )

            ctx.userdata.orders.append(order)
            save_orders(ctx.userdata.orders)

            return f"Order placed! {quantity} × {product.name} for ₹{total}. Anything else you'd like?"

        return create_order

    # -----------------------------------------------------------------
    # Tool: Retrieve last order
    # -----------------------------------------------------------------

    def tool_get_last_order(self):
        @function_tool
        async def get_last_order(ctx: RunContext[Userdata]) -> str:
            if not ctx.userdata.orders:
                return "No recent shopping yet!"
            last = ctx.userdata.orders[-1]
            qty = last.items[0].get("qty", 1)
            return f"Latest order {last.order_id}: {qty} item(s) worth ₹{last.total}."

        return get_last_order


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    userdata = Userdata(
        catalog=load_catalog(),
        orders=load_orders(),
    )

    session = AgentSession(
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
        vad=silero.VAD.load(),
        preemptive_generation=True,
    )

    await session.start(
        agent=ACPShoppingAgent(userdata=userdata),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

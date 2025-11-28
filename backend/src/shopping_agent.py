import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, List, Dict, Optional

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
)

from livekit.plugins import deepgram, google, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("shoppint_agent.agent")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "shopping_data")

os.makedirs(DATA_DIR, exist_ok=True)

CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.json")

class CatalogItem(BaseModel):
    id: str
    name: str
    category: str
    price: float
    unit: str
    tags: List[str] = []


class CartItem(BaseModel):
    item_id: str
    name: str
    quantity: float
    unit: str
    price_per_unit: float
    line_total: float


class Order(BaseModel):
    order_id: str
    items: List[CartItem]
    total_amount: float
    status: str
    created_at: str
    updated_at: str


@dataclass
class Userdata:
    catalog: List[CatalogItem]
    cart: List[CartItem]
    orders: List[Order]
    active_order_id: Optional[str] = None


def load_catalog() -> List[CatalogItem]:
    if not os.path.exists(CATALOG_PATH):
        return []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [CatalogItem(**item) for item in raw]


def load_orders() -> List[Order]:
    if not os.path.exists(ORDERS_PATH):
        return []
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Order(**entry) for entry in raw]
    except:
        logger.warning("Orders file corrupted. Starting fresh.")
        return []


def save_orders(orders: List[Order]) -> None:
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump([o.dict() for o in orders], f, indent=4)


def generate_order_id(existing: List[Order]) -> str:
    if not existing:
        return "QK-1001"
    last = sorted(existing, key=lambda o: o.order_id)[-1]
    try:
        num = int(last.order_id.split("-")[-1])
        return f"QK-{num+1}"
    except:
        return f"QK-{len(existing)+1001}"


def compute_status(order: Order) -> str:
    try:
        created = datetime.fromisoformat(order.created_at)
    except:
        return order.status

    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    elapsed = (now - created).total_seconds()

    if elapsed < 5:
        return "placed"
    elif elapsed < 10:
        return "preparing"
    elif elapsed < 20:
        return "out_for_delivery"
    else:
        return "delivered"


QUICKMART_INSTRUCTIONS = """
You are Fred, QuickMart's Shopping Assistant — a friendly grocery shopping voice assistant.
Overview user about who you are, what can you do. and provide list of popular items.

Help the user add products to their cart and place orders.
After placing an order, they may ask “Where is my order?” anytime.
Keep responses short and conversational.
"""

class QuickMartAgent(Agent):
    def __init__(self, *, userdata: Userdata):
        preview = ", ".join(item.name for item in userdata.catalog[:6])
        super().__init__(
            instructions=QUICKMART_INSTRUCTIONS + f"\nPopular items: {preview}.",
            tools=[
                self.add_to_cart_tool(),
                self.list_popular_items_tool(),
                self.place_order_tool(),
                self.get_order_status_tool(),
            ],
        )

    @staticmethod
    def _find_item(catalog, q):
        q = q.lower()
        for item in catalog:
            if q in item.name.lower() or any(q in t.lower() for t in item.tags):
                return item
        return None

    # ---------------- tool: Add Item ----------------
    def add_to_cart_tool(self):
        @function_tool
        async def add_to_cart(ctx: RunContext[Userdata], item_query: str, quantity: float):
            if quantity <= 0:
                return "Quantity must be greater than zero."

            item = QuickMartAgent._find_item(ctx.userdata.catalog, item_query)
            if not item:
                return "That item isn’t available now. Try asking for milk, bread, bananas…"

            cost = item.price * quantity
            ctx.userdata.cart.append(
                CartItem(
                    item_id=item.id,
                    name=item.name,
                    quantity=quantity,
                    unit=item.unit,
                    price_per_unit=item.price,
                    line_total=round(cost, 2),
                )
            )
            return f"Added {quantity} {item.unit} of {item.name}."

        return add_to_cart

    # ---------------- tool: Popular Items ----------------
    def list_popular_items_tool(self):
        @function_tool
        async def list_popular_items(ctx: RunContext[Userdata]):
            items = ctx.userdata.catalog[:6]
            return "Popular picks: " + ", ".join(i.name for i in items)
        return list_popular_items

    # ---------------- tool: Place Order ----------------
    def place_order_tool(self):
        @function_tool
        async def place_order(ctx: RunContext[Userdata]):
            if not ctx.userdata.cart:
                return "Your cart is empty!"

            order_id = generate_order_id(ctx.userdata.orders)
            total = round(sum(i.line_total for i in ctx.userdata.cart), 2)
            now = datetime.now(timezone.utc).isoformat()

            ctx.userdata.orders.append(Order(
                order_id=order_id,
                items=list(ctx.userdata.cart),
                total_amount=total,
                status="placed",
                created_at=now,
                updated_at=now,
            ))
            ctx.userdata.active_order_id = order_id
            ctx.userdata.cart.clear()
            save_orders(ctx.userdata.orders)

            return f"Order {order_id} placed successfully! Total: ₹{total}. You can ask for status anytime."

        return place_order

    # ---------------- tool: Check Status ----------------
    def get_order_status_tool(self):
        @function_tool
        async def get_order_status(
            ctx: RunContext[Userdata],
            order_id: Optional[str] = None
        ) -> Dict[str, str]:

            orders = ctx.userdata.orders
            if not orders:
                await ctx.session.say(
                    "I don’t see any orders on your account yet. You can add items to your cart and place a new order anytime!"
                )
                return {"status": "no_orders"}

            if not order_id:
                order_id = ctx.userdata.active_order_id

            order = next((o for o in orders if o.order_id == order_id), None)
            if not order:
                await ctx.session.say(f"Sorry, I couldn't locate order ID {order_id}. Please check once again.")
                return {"status": "not_found"}

            # Auto-update order status progression
            new_status = compute_status(order)
            if new_status != order.status:
                logger.info(f"Order {order.order_id} status changed: {order.status} → {new_status}")
                order.status = new_status
                order.updated_at = datetime.now(timezone.utc).isoformat()
                save_orders(orders)

            # Safe-length voice responses
            if new_status == "placed":
                msg = (
                    f"Order {order.order_id} has been placed successfully. "
                    "Our team has started preparing it for packing. I’ll keep you posted!"
                )
            elif new_status == "preparing":
                msg = (
                    f"Order {order.order_id} is currently being packed. "
                    "Just a little bit longer and it will be on its way to you!"
                )
            elif new_status == "out_for_delivery":
                msg = (
                    f"Order {order.order_id} is out for delivery now. "
                    "A delivery partner is heading to your location with your items!"
                )
            else:  # delivered
                msg = (
                    f"Order {order.order_id} has been delivered successfully. "
                    "Hope everything reached fresh and right on time! Enjoy your order and feel free to shop again anytime!"
                )

            await ctx.session.say(msg)

            return {"status": new_status, "order_id": order_id}

        return get_order_status


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    userdata = Userdata(
        catalog=load_catalog(),
        cart=[],
        orders=load_orders(),
    )

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        # FIXED TTS — no tokenizer or text_pacing
        tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        preemptive_generation=False,
    )

    await session.start(agent=QuickMartAgent(userdata=userdata),
                        room=ctx.room,
                        room_input_options=RoomInputOptions(
                            noise_cancellation=noise_cancellation.BVC()
                        ))

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

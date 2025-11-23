import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    ToolError,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from database import MenuItem, COMMON_INSTRUCTIONS, load_coffee_menu, OrderItem
from typing import Annotated, Literal
from pydantic import Field
from dataclasses import dataclass
import os
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")

orders_dir = "orders"
os.makedirs(orders_dir, exist_ok=True)


@dataclass
class Userdata:
    drinks: list[MenuItem]
    milks: list[MenuItem]
    extras: list[MenuItem]
    order_items: dict


class Assistant(Agent):
    def __init__(self, *, userdata: Userdata) -> None:
        instructions = (
            COMMON_INSTRUCTIONS
            + "\n\nCoffee Menu:\n"
            + "\n".join(f"- {item.id}: {item.name}" for item in userdata.drinks)
            + "\n\nMilk Options:\n"
            + "\n".join(f"- {item.id}: {item.name}" for item in userdata.milks)
            + "\n\nExtras:\n"
            + "\n".join(f"- {item.id}: {item.name}" for item in userdata.extras)
        )

        super().__init__(
            instructions=instructions,
            # Add new tool here to accept order details
            tools=[
                self.build_order_coffee_tool(
                    userdata.drinks,
                    userdata.milks,
                    userdata.extras,
                ),
                self.build_list_order_items_tool(),
                self.build_remove_order_item_tool(),
            ],
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

    def build_order_coffee_tool(
        self,
        drinks: list[MenuItem],
        milks: list[MenuItem],
        extras_list: list[MenuItem],
    ):
        drink_ids = {item.id for item in drinks}
        milk_ids = {item.id for item in milks}
        extras_ids = {item.id for item in extras_list}

        @function_tool
        async def order_coffee(
            ctx: RunContext[Userdata],
            drinkType: Annotated[
                str,
                Field(
                    description="Coffee type",
                    json_schema_extra={"enum": list(drink_ids)},
                ),
            ],
            size: Annotated[
                str,
                Field(
                    description="Size: S, M or L",
                    json_schema_extra={"enum": ["S", "M", "L"]},
                ),
            ],
            milk: Annotated[
                str,
                Field(
                    description="Milk type", json_schema_extra={"enum": list(milk_ids)}
                ),
            ],
            extras: Annotated[
                str,
                Field(
                    description="Optional extra",
                    json_schema_extra={"enum": list(extras_ids)},
                ),
            ] = "none",
            name: Annotated[
                str,
                Field(description="Customer name"),
            ] = "Guest",
        ):
            """
            Called when user clearly specifies all coffee fields.
            """

            # Validate IDs exist
            if drinkType not in drink_ids:
                raise ToolError(f"Unknown drink type: {drinkType}")

            if milk not in milk_ids:
                raise ToolError(f"Unknown milk option: {milk}")

            if extras not in extras_ids and extras != "none":
                raise ToolError(f"Unknown extra: {extras}")

            item = OrderItem(
                drinkType=drinkType,
                size=size,
                milk=milk,
                extras=extras,
                name=name,
            )

            order_id = f"order-{len(ctx.userdata.order_items) + 1}"
            ctx.userdata.order_items[order_id] = item

            file_path = os.path.join(orders_dir, f"{order_id}.json")

            order_data = {
                "order_id": order_id,
                "drinkType": item.drinkType,
                "size": item.size,
                "milk": item.milk,
                "extras": item.extras,
                "customer_name": item.name,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(order_data, f, indent=4)

            print(f"Order saved to {file_path}")

            return (
                f"{name}, your {size} {milk} {drinkType} with {extras} has been added."
            )

        return order_coffee

    def build_list_order_items_tool(self):
        @function_tool
        async def list_order_items(ctx: RunContext[Userdata]) -> str:
            if not ctx.userdata.order_items:
                return "Your order is empty."

            lines = []
            for oid, item in ctx.userdata.order_items.items():
                lines.append(f"{oid}: {item.model_dump_json()}")

            return "\n".join(lines)

        return list_order_items

    def build_remove_order_item_tool(self):
        @function_tool
        async def remove_order_item(
            ctx: RunContext[Userdata],
            order_id: Annotated[
                str,
                Field(description="The order_id to remove"),
            ],
        ) -> str:
            if order_id not in ctx.userdata.order_items:
                return "No such item."

            removed = ctx.userdata.order_items.pop(order_id)
            return f"Removed: {removed.model_dump_json()}"

        return remove_order_item


async def new_userdata() -> Userdata:
    menu = await load_coffee_menu()

    return Userdata(
        drinks=menu["drinks"],
        milks=menu["milks"],
        extras=menu["extras"],
        order_items={},
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    userdata = await new_userdata()
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession[Userdata](
        userdata=userdata,
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(userdata=userdata),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

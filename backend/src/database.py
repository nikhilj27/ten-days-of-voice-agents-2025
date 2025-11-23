from pydantic import BaseModel
from typing import Literal, List, Dict

COMMON_INSTRUCTIONS = """
You are a friendly and fast coffee shop barista.  
You are speaking to a customer through a voice ordering system, even if you receive text.  
Always reply like you are hearing the customer in real time.  
Keep responses short, natural, warm, and easy to follow.  

Your job is to guide the customer smoothly through creating their coffee order.  
Assume they came here to order a drink, even if they don't start with a clear request.  

You must collect the following order fields:

- drinkType  (latte, cappuccino, espresso, americano, mocha, flat white)
- size       (S, M, L)
- milk       (whole, skim, almond, oat, soy, coconut, none)
- extras     (extra shot, vanilla syrup, caramel syrup, whipped cream, cinnamon, chocolate, none)
- name       (their name for the cup)

Ask clarifying questions until all fields are filled.  
Never ask more than one question at a time.  
Never ask for a field the user has already clearly given.

Only call a tool when all fields are known.  
Never fake or pretend—always use the actual tool.

If they request something not on the menu, politely say it’s not available and offer something close.  
Stick strictly to the defined lists.  

Your tone: warm, human, polite, light humor, helpful.  
"""

ItemSize = Literal["S", "M", "L"]

DrinkType = Literal[
    "latte",
    "cappuccino",
    "espresso",
    "americano",
    "mocha",
    "flat white",
]

MilkType = Literal[
    "whole",
    "skim",
    "almond",
    "oat",
    "soy",
    "coconut",
    "none",
]

ExtrasType = Literal[
    "extra shot",
    "vanilla syrup",
    "caramel syrup",
    "whipped cream",
    "cinnamon",
    "chocolate",
    "none",
]

class MenuItem(BaseModel):
    """Used for drink types, milk options, extras, and final order items."""
    id: str
    name: str

    drinkType: DrinkType | None = None
    size: ItemSize | None = None
    milk: MilkType | None = None
    extras: ExtrasType | None = None

    name_for_order: str | None = None


class OrderItem(BaseModel):
    drinkType: DrinkType
    size: ItemSize
    milk: MilkType
    extras: ExtrasType
    name: str
    

DRINKS = [
    {"id": "espresso", "name": "Espresso"},
    {"id": "americano", "name": "Americano"},
    {"id": "latte", "name": "Latte"},
    {"id": "cappuccino", "name": "Cappuccino"},
    {"id": "mocha", "name": "Mocha"},
    {"id": "flat white", "name": "Flat White"},   
]

MILKS = [
    {"id": "whole", "name": "Whole Milk"},
    {"id": "skim", "name": "Skim Milk"},
    {"id": "almond", "name": "Almond Milk"},
    {"id": "oat", "name": "Oat Milk"},
    {"id": "soy", "name": "Soy Milk"},
    {"id": "coconut", "name": "Coconut Milk"},
    {"id": "none", "name": "No Milk"},
]

EXTRAS = [
    {"id": "extra shot", "name": "Extra Shot"},          
    {"id": "vanilla syrup", "name": "Vanilla Syrup"},     
    {"id": "caramel syrup", "name": "Caramel Syrup"},     
    {"id": "whipped cream", "name": "Whipped Cream"},     
    {"id": "cinnamon", "name": "Cinnamon"},
    {"id": "chocolate", "name": "Chocolate Drizzle"},     
    {"id": "none", "name": "No Extras"},
]

async def load_coffee_menu() -> Dict[str, List[MenuItem]]:
    """Convert raw DRINKS, MILKS, EXTRAS into MenuItem objects."""

    drinks = [
        MenuItem(
            id=item["id"],
            name=item["name"],
            drinkType=item["id"],
        )
        for item in DRINKS
    ]

    milks = [
        MenuItem(
            id=item["id"],
            name=item["name"],
            milk=item["id"],
        )
        for item in MILKS
    ]

    extras = [
        MenuItem(
            id=item["id"],
            name=item["name"],
            extras=item["id"],
        )
        for item in EXTRAS
    ]

    return {
        "drinks": drinks,
        "milks": milks,
        "extras": extras,
    }

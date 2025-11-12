import os
import re
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Travel Agency Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    slots: Dict[str, Any]
    reply: str
    follow_up: Optional[str] = None


class IntentDef(BaseModel):
    name: str
    description: str
    sample_utterances: List[str]
    required_slots: List[str] = []


# ---------- Intent Catalog ----------
INTENTS: List[IntentDef] = [
    IntentDef(
        name="greeting",
        description="User says hello or starts a conversation",
        sample_utterances=["hi", "hello", "hey there", "good morning"],
    ),
    IntentDef(
        name="search_flights",
        description="Find flights between two cities on a date",
        sample_utterances=[
            "Find flights from NYC to Paris on June 5",
            "Any flights to Tokyo next Friday?",
            "I need a flight from London to Rome",
        ],
        required_slots=["from", "to", "date"],
    ),
    IntentDef(
        name="book_hotel",
        description="Book a hotel in a city for given dates",
        sample_utterances=[
            "Book a hotel in Paris from July 1 to July 5",
            "Need a hotel in Tokyo next weekend",
            "Find me a 4-star hotel in Rome for two nights",
        ],
        required_slots=["location", "check_in", "check_out"],
    ),
    IntentDef(
        name="cancel_booking",
        description="Cancel an existing booking by ID",
        sample_utterances=[
            "Cancel booking ABC123",
            "I want to cancel reservation 9XZ45",
        ],
        required_slots=["booking_id"],
    ),
    IntentDef(
        name="weather",
        description="Get destination weather for a date",
        sample_utterances=[
            "What's the weather in Paris tomorrow?",
            "Forecast for Tokyo on June 10",
        ],
        required_slots=["location", "date"],
    ),
    IntentDef(
        name="thanks",
        description="User expresses gratitude",
        sample_utterances=["thanks", "thank you", "cheers"],
    ),
    IntentDef(
        name="help",
        description="User asks what the assistant can do",
        sample_utterances=["help", "what can you do?", "options"],
    ),
]


# ---------- Simple NLU ----------
CITY_PATTERN = re.compile(r"\b(paris|tokyo|rome|london|new york|nyc|madrid|berlin|sydney|dubai)\b", re.I)
BOOKING_ID_PATTERN = re.compile(r"\b([A-Z0-9]{5,8})\b")
DATE_WORDS = [
    "today",
    "tomorrow",
    "tonight",
    "next monday",
    "next tuesday",
    "next wednesday",
    "next thursday",
    "next friday",
    "next saturday",
    "next sunday",
]
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.I,
)


def extract_date(text: str) -> Optional[str]:
    # Look for explicit dates first
    m = DATE_PATTERN.search(text)
    if m:
        return m.group(0)
    # Look for natural language date words
    for w in DATE_WORDS:
        if w in text.lower():
            return w
    return None


def extract_cities(text: str) -> List[str]:
    return [m.group(0).lower() for m in CITY_PATTERN.finditer(text)]


def guess_intent(message: str) -> ChatResponse:
    text = message.strip()
    low = text.lower()

    # Greetings
    if re.search(r"\b(hi|hello|hey|good\s+(morning|afternoon|evening))\b", low):
        return ChatResponse(intent="greeting", confidence=0.95, slots={}, reply="Hi! Where would you like to travel? I can search flights and hotels, or check the weather.")

    # Thanks
    if re.search(r"\b(thanks|thank you|cheers|appreciate it)\b", low):
        return ChatResponse(intent="thanks", confidence=0.95, slots={}, reply="You're welcome! Anything else I can help with?")

    # Help
    if re.search(r"\b(help|what can you do|options)\b", low):
        return ChatResponse(
            intent="help",
            confidence=0.9,
            slots={},
            reply="I can: 1) find flights, 2) book hotels, 3) check weather, 4) cancel bookings if you have an ID.",
        )

    # Cancel booking
    if re.search(r"\b(cancel|void)\b", low) and re.search(r"\b(booking|reservation)\b", low):
        bid = None
        m = BOOKING_ID_PATTERN.search(text)
        if m:
            bid = m.group(1)
        reply = "I can help cancel that. What's the booking ID?" if not bid else f"Okay, I can cancel booking {bid}. Do you want me to proceed?"
        return ChatResponse(intent="cancel_booking", confidence=0.85, slots={"booking_id": bid} if bid else {}, reply=reply)

    # Book hotel
    if re.search(r"\b(hotel|stay|accommodation)\b", low) and re.search(r"\b(book|reserve|find)\b", low):
        cities = extract_cities(text)
        date1 = extract_date(text)
        date2 = None
        # attempt check-out by looking for two dates in text
        dates = list(DATE_PATTERN.finditer(text))
        if len(dates) >= 2:
            date1 = dates[0].group(0)
            date2 = dates[1].group(0)
        reply = "Sure, what city and dates are you looking at?"
        if cities and date1 and date2:
            reply = f"Got it. Searching hotels in {cities[0].title()} from {date1} to {date2}..."
        elif cities and date1:
            reply = f"Great. What's your check-out date for {cities[0].title()} after {date1}?"
        elif cities:
            reply = f"Great. What dates would you like in {cities[0].title()}?"
        return ChatResponse(
            intent="book_hotel",
            confidence=0.8,
            slots={"location": cities[0] if cities else None, "check_in": date1, "check_out": date2},
            reply=reply,
            follow_up="Please provide missing details if any.",
        )

    # Search flights
    if re.search(r"\b(flight|flights|plane)\b", low) and re.search(r"\b(from|to)\b", low):
        cities = extract_cities(text)
        date = extract_date(text)
        slots: Dict[str, Any] = {"from": None, "to": None, "date": date}
        if len(cities) >= 2:
            slots["from"], slots["to"] = cities[0], cities[1]
        elif len(cities) == 1:
            # Try to infer direction based on wording
            if "from" in low:
                slots["from"] = cities[0]
            elif "to" in low:
                slots["to"] = cities[0]
        reply = "Looking for flights"
        parts = []
        if slots["from"]:
            parts.append(f"from {slots['from'].title()}")
        if slots["to"]:
            parts.append(f"to {slots['to'].title()}")
        if date:
            parts.append(f"on {date}")
        if parts:
            reply = " ".join(["Got it."] + parts) + "..."
        else:
            reply = "Sure — what's your origin, destination, and date?"
        return ChatResponse(intent="search_flights", confidence=0.82, slots=slots, reply=reply)

    # Weather
    if re.search(r"\b(weather|forecast)\b", low):
        cities = extract_cities(text)
        date = extract_date(text)
        reply = "I can get the forecast. Which city and date?"
        if cities and date:
            reply = f"Forecast for {cities[0].title()} on {date}: 23°C, partly cloudy (sample)."
        return ChatResponse(intent="weather", confidence=0.78, slots={"location": cities[0] if cities else None, "date": date}, reply=reply)

    # Fallback
    return ChatResponse(
        intent="fallback",
        confidence=0.4,
        slots={},
        reply="I'm not sure I understood. Try asking me to find flights, book a hotel, check weather, or cancel a booking.",
    )


# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Travel Agency Chatbot Backend Running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/intents", response_model=List[IntentDef])
def list_intents():
    return INTENTS


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return guess_intent(req.message)


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class TurnAnnotation(BaseModel):
    conversation_id: str
    turn_id: int
    speaker: Literal["caller", "recipient"]
    text: str
    labels: List[Literal["none", "urgency", "isolation", "authority", "emotional_destabilization", "channel_control"]]
    synthetic: bool = True
    source_type: Literal["synthetic"] = "synthetic"
    source: str
    difficulty: Literal["easy", "medium", "hard", "none"] = "none"

class Conversation(BaseModel):
    conversation_id: str
    conversation_level_label: Literal["fraud", "benign"]
    turns: List[TurnAnnotation]

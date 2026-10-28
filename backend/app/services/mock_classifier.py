from typing import Dict

class MockClassifier:
    """
    Mock classifier for the MVP to simulate tactic detection without ML.
    In real implementation, this would be a HuggingFace Transformer model.
    """
    def predict(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        tactics = {
            "urgency": 0.0,
            "isolation": 0.0,
            "authority": 0.0,
            "emotional_destabilization": 0.0,
            "channel_control": 0.0
        }

        # Mocking logic based on keywords for demonstration purposes
        if any(word in text_lower for word in ["bank", "police", "officer", "government", "microsoft", "department"]):
            tactics["authority"] = 0.85
            
        if any(word in text_lower for word in ["arrest", "trouble", "compromised", "attacked", "lose"]):
            tactics["emotional_destabilization"] = 0.75
            
        if any(word in text_lower for word in ["now", "immediately", "minutes", "time", "today"]):
            tactics["urgency"] = 0.80
            
        if any(word in text_lower for word in ["don't tell", "between us", "don't call", "nobody"]):
            tactics["isolation"] = 0.90
            
        if any(word in text_lower for word in ["stay on", "do not hang up", "guide you"]):
            tactics["channel_control"] = 0.88

        return tactics

import asyncio
from typing import Any
from app.services.mock_classifier import MockClassifier
from app.services.risk_engine import RiskEngine

class DemoPipeline:
    def __init__(self, session_id: str, ws_manager: Any):
        self.session_id = session_id
        self.ws_manager = ws_manager
        self.classifier = MockClassifier()
        self.risk_engine = RiskEngine()
        
        # Simulated bank impersonation conversation
        self.script = [
            {"speaker": "caller", "text": "Hello, I'm calling from your bank's fraud department.", "delay": 2},
            {"speaker": "recipient", "text": "Oh, hello. Is everything alright?", "delay": 2},
            {"speaker": "caller", "text": "We have detected suspicious activity on your account. Your account has been compromised.", "delay": 3},
            {"speaker": "recipient", "text": "Oh my god, what should I do?", "delay": 2},
            {"speaker": "caller", "text": "You need to resolve this within the next ten minutes or you could lose everything.", "delay": 3},
            {"speaker": "recipient", "text": "I should call my daughter, she helps me with this.", "delay": 2},
            {"speaker": "caller", "text": "Please don't call your daughter because this could compromise the investigation. Keep this between us.", "delay": 4},
            {"speaker": "recipient", "text": "Okay... I won't.", "delay": 2},
            {"speaker": "caller", "text": "Good. Now stay on this line and I'll guide you through the transfer to a secure account.", "delay": 3}
        ]

    async def run(self):
        time_elapsed = 0
        for turn in self.script:
            await asyncio.sleep(turn["delay"])
            time_elapsed += turn["delay"]
            
            # Send Transcript
            await self.ws_manager.send_message({
                "type": "transcript",
                "speaker": turn["speaker"],
                "text": turn["text"],
                "time": time_elapsed
            }, self.session_id)
            
            # Only process caller text for risk
            if turn["speaker"] == "caller":
                tactics = self.classifier.predict(turn["text"])
                
                # Send Tactics
                await self.ws_manager.send_message({
                    "type": "tactic_detection",
                    "tactics": tactics
                }, self.session_id)
                
                # Calculate Risk
                risk_data = self.risk_engine.calculate_risk(tactics)
                
                # Send Risk Update
                await self.ws_manager.send_message({
                    "type": "risk_update",
                    "score": risk_data["score"],
                    "level": risk_data["level"],
                    "explanation": risk_data["explanation"]
                }, self.session_id)
                
                # Send Intervention if high
                if risk_data["level"] in ["HIGH", "CRITICAL"]:
                    message = "Potential manipulation detected. Consider pausing and independently verifying the caller."
                    if risk_data["level"] == "CRITICAL":
                        message = "High-risk manipulation pattern detected. Do not provide financial information or transfer money. Consider ending the call and contacting the organization using a verified number."
                        
                    await self.ws_manager.send_message({
                        "type": "intervention",
                        "severity": risk_data["level"],
                        "message": message
                    }, self.session_id)

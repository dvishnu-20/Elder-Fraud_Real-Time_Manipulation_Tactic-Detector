from typing import List, Dict
from app.config import settings

class RiskEngine:
    def __init__(self):
        self.tactic_history: List[Dict[str, float]] = []
        self.current_score: float = 0.0

    def calculate_risk(self, current_tactics: Dict[str, float]) -> dict:
        """
        Calculates composite risk score.
        MVP Implementation: Additive with progression multipliers.
        """
        self.tactic_history.append(current_tactics)
        
        # Simple tactic density
        active_tactics = sum(1 for v in current_tactics.values() if v > 0.5)
        
        # Progression logic (e.g., Authority followed by Urgency increases score more)
        progression_multiplier = 1.0
        if len(self.tactic_history) > 1:
            prev = self.tactic_history[-2]
            # Pattern: Authority -> Urgency/Emotional
            if prev.get("authority", 0) > 0.5 and (current_tactics.get("urgency", 0) > 0.5 or current_tactics.get("emotional_destabilization", 0) > 0.5):
                progression_multiplier = 1.5
            # Pattern: Isolation -> Channel Control
            if prev.get("isolation", 0) > 0.5 and current_tactics.get("channel_control", 0) > 0.5:
                progression_multiplier = 1.8

        # Base score from current turn
        base_score = sum(v for v in current_tactics.values() if v > 0.5) * 10
        
        # Accumulate risk
        step_risk = base_score * progression_multiplier * (1 + 0.1 * active_tactics)
        self.current_score += step_risk
        
        # Cap at 100
        final_score = min(int(self.current_score), 100)
        
        level = self._get_risk_level(final_score)
        
        return {
            "score": final_score,
            "level": level,
            "explanation": self._generate_explanation(current_tactics, progression_multiplier)
        }

    def _get_risk_level(self, score: int) -> str:
        if score >= settings.RISK_THRESHOLD_CRITICAL:
            return "CRITICAL"
        elif score >= settings.RISK_THRESHOLD_HIGH:
            return "HIGH"
        elif score >= settings.RISK_THRESHOLD_GUARDED:
            return "GUARDED"
        return "LOW"
        
    def _generate_explanation(self, tactics: dict, multiplier: float) -> str:
        reasons = []
        for t, v in tactics.items():
            if v > 0.5:
                reasons.append(f"{t.replace('_', ' ').title()} detected.")
                
        if multiplier > 1.0:
            reasons.append("Multiple manipulation tactics are occurring in a progressive sequence.")
            
        return " ".join(reasons) if reasons else "Normal conversation pattern."

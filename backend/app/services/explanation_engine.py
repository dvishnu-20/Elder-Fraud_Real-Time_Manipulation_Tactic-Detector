"""
Phase 7 — Explanation Engine
Generates human-readable risk explanations from tactic probabilities and temporal risk.

Uses scientifically defensible wording:
- "High-risk manipulation pattern detected" NOT "The caller is definitely a scammer"
- Distinguishes TACTIC EVIDENCE from FRAUD CONFIDENCE
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Tactic display names
TACTIC_NAMES = {
    "urgency": "Urgency",
    "isolation": "Isolation",
    "authority": "Authority Impersonation",
    "emotional_destabilization": "Emotional Destabilization",
    "channel_control": "Channel Control"
}

# Tactic descriptions for explanations
TACTIC_DESCRIPTIONS = {
    "urgency": "creating artificial time pressure",
    "isolation": "discouraging contact with trusted parties",
    "authority": "impersonating an authority figure",
    "emotional_destabilization": "using emotional manipulation",
    "channel_control": "controlling the communication channel"
}


def generate_explanation(
    tactics: Dict[str, float],
    temporal_risk: float,
    threshold: float,
    tactic_thresholds: Dict[str, float],
    risk_state: str,
    turn_id: int,
    risk_history: List[float] = None
) -> dict:
    """
    Generate a structured explanation for the current risk assessment.
    
    Returns:
        {
            "summary": str,
            "active_tactics": [...],
            "risk_trend": str,
            "alert_level": str,
            "alert_message": str or None
        }
    """
    # Identify active tactics (above their calibrated thresholds)
    active_tactics = []
    for tactic, prob in tactics.items():
        t_threshold = tactic_thresholds.get(tactic, 0.5)
        if prob >= t_threshold:
            active_tactics.append({
                "tactic": tactic,
                "display_name": TACTIC_NAMES.get(tactic, tactic),
                "probability": round(prob, 4),
                "threshold": t_threshold,
                "description": TACTIC_DESCRIPTIONS.get(tactic, "")
            })
    
    # Sort by probability descending
    active_tactics.sort(key=lambda x: x["probability"], reverse=True)
    
    # Determine risk trend
    risk_trend = "stable"
    if risk_history and len(risk_history) >= 2:
        prev = risk_history[-2]
        curr = risk_history[-1]
        delta = curr - prev
        if delta > 0.05:
            risk_trend = "increasing"
        elif delta < -0.05:
            risk_trend = "decreasing"
    
    # Generate summary
    if risk_state == "NORMAL":
        summary = "Normal conversation pattern. No manipulation indicators detected."
    elif risk_state == "WATCH":
        summary = "Low-level indicators present. Monitoring conversation."
    elif risk_state == "ELEVATED":
        if active_tactics:
            tactic_names = [t["display_name"] for t in active_tactics[:3]]
            summary = f"Manipulation pattern detected. Indicators: {', '.join(tactic_names)}."
        else:
            summary = "Temporal risk has crossed the decision threshold."
    elif risk_state == "HIGH_RISK":
        if active_tactics:
            tactic_names = [t["display_name"] for t in active_tactics[:3]]
            summary = f"High-risk manipulation pattern detected. Active tactics: {', '.join(tactic_names)}."
        else:
            summary = "High-risk manipulation pattern detected based on conversation progression."
    elif risk_state == "CRITICAL":
        summary = "Very high-confidence manipulation pattern detected. Multiple manipulation tactics identified in sequential escalation."
    else:
        summary = f"Risk state: {risk_state}"
    
    if risk_trend == "increasing" and risk_state in ("ELEVATED", "HIGH_RISK", "CRITICAL"):
        summary += " Risk is escalating."
    
    # Generate alert message if threshold crossed
    alert_message = None
    if temporal_risk >= threshold:
        if risk_state == "CRITICAL":
            alert_message = (
                "HIGH-RISK MANIPULATION PATTERN DETECTED. "
                "Do not provide financial information or transfer money. "
                "Consider ending the call and independently verifying the caller "
                "using a known, trusted phone number."
            )
        elif risk_state == "HIGH_RISK":
            alert_message = (
                "Potential manipulation detected. "
                "Consider pausing and independently verifying the caller's identity."
            )
        elif risk_state == "ELEVATED":
            alert_message = (
                "Elevated risk indicators detected. "
                "Exercise caution with any requests for personal or financial information."
            )
    
    return {
        "summary": summary,
        "active_tactics": active_tactics,
        "risk_trend": risk_trend,
        "alert_level": risk_state,
        "alert_message": alert_message,
        "temporal_risk": round(temporal_risk, 4),
        "turn_id": turn_id
    }

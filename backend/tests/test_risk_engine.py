from app.services.risk_engine import RiskEngine

def test_risk_engine_baseline():
    engine = RiskEngine()
    
    # Empty tactics should return LOW risk and score 0
    empty_tactics = {
        "urgency": 0.0,
        "isolation": 0.0,
        "authority": 0.0,
        "emotional_destabilization": 0.0,
        "channel_control": 0.0
    }
    
    result = engine.calculate_risk(empty_tactics)
    assert result["score"] == 0
    assert result["level"] == "LOW"

def test_risk_engine_escalation():
    engine = RiskEngine()
    
    # Step 1: Authority
    res1 = engine.calculate_risk({
        "urgency": 0.0, "isolation": 0.0, "authority": 0.8, 
        "emotional_destabilization": 0.0, "channel_control": 0.0
    })
    
    # Step 2: Urgency (Progression should trigger multiplier)
    res2 = engine.calculate_risk({
        "urgency": 0.9, "isolation": 0.0, "authority": 0.0, 
        "emotional_destabilization": 0.0, "channel_control": 0.0
    })
    
    assert res1["score"] > 0
    assert res2["score"] > res1["score"] * 2  # Shows progression multiplier impact
    
def test_risk_engine_critical_threshold():
    engine = RiskEngine()
    
    # Hit it with all tactics
    res = engine.calculate_risk({
        "urgency": 0.9, "isolation": 0.9, "authority": 0.9, 
        "emotional_destabilization": 0.9, "channel_control": 0.9
    })
    
    assert res["level"] in ["HIGH", "CRITICAL"]
    assert res["score"] > 50

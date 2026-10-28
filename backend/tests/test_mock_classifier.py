from app.services.mock_classifier import MockClassifier

def test_mock_classifier_tactics():
    classifier = MockClassifier()
    
    # Test Authority
    result = classifier.predict("I am calling from the bank's security department.")
    assert result["authority"] > 0.5
    assert result["urgency"] == 0.0
    
    # Test Urgency
    result = classifier.predict("You must act immediately, we only have ten minutes.")
    assert result["urgency"] > 0.5
    
    # Test Isolation
    result = classifier.predict("Please keep this between us and don't tell anyone.")
    assert result["isolation"] > 0.5
    
    # Test Emotional Destabilization
    result = classifier.predict("Your account is compromised and you are in trouble.")
    assert result["emotional_destabilization"] > 0.5
    
    # Test Channel Control
    result = classifier.predict("Do not hang up, stay on the line.")
    assert result["channel_control"] > 0.5
    
    # Test multiple tactics
    result = classifier.predict("I'm an officer. Don't tell anyone and stay on this line immediately.")
    assert result["authority"] > 0.5
    assert result["isolation"] > 0.5
    assert result["channel_control"] > 0.5
    assert result["urgency"] > 0.5

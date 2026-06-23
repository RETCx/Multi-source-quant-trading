import os
import json
import numpy as np
from src.models.self_reflection import AgenticJournal

def test_veto_logic():
    print("--- Testing Agentic Self-Reflection VETO Logic ---")
    
    # Setup mock journal
    journal_path = "data/models/test_trade_journal.json"
    if os.path.exists(journal_path):
        os.remove(journal_path)
        
    journal = AgenticJournal(journal_path=journal_path, similarity_threshold=0.95)
    
    # Mock Features (e.g., top 5 indicators)
    features_mistake = {
        "RSI14": 75.0,
        "MACD": 0.5,
        "VIX_Close": 20.0,
        "SMA20_ratio": 1.05,
        "SPY_Return": -0.01
    }
    
    # Record a mistake
    print("\n1. Recording a past mistake...")
    journal.analyze_and_record_mistake(
        date="2025-05-10",
        predicted_action="BUY",
        actual_action="SELL",
        features_dict=features_mistake,
        gemini_api_key=None # Skip LLM call for testing
    )
    
    # Mock Current Features (Very similar to the mistake)
    features_current_similar = {
        "RSI14": 74.5,
        "MACD": 0.51,
        "VIX_Close": 19.8,
        "SMA20_ratio": 1.04,
        "SPY_Return": -0.012
    }
    
    print("\n2. Checking VETO on similar market state (Proposing BUY)...")
    is_vetoed, reason = journal.check_veto(features_current_similar, "BUY")
    print(f"Vetoed? {is_vetoed}")
    print(f"Reason: {reason}")
    assert is_vetoed == True, "VETO should have triggered for highly similar state!"
    
    print("\n3. Checking VETO on similar market state (Proposing SELL)...")
    is_vetoed, reason = journal.check_veto(features_current_similar, "SELL")
    print(f"Vetoed? {is_vetoed}")
    print(f"Reason: {reason}")
    assert is_vetoed == False, "VETO should NOT trigger if proposing a different action than the mistaken one!"
    
    # Mock Current Features (Different state - like a bearish state)
    features_current_different = {
        "RSI14": -2.0, # Scaled values
        "MACD": -1.5,
        "VIX_Close": 3.0,
        "SMA20_ratio": -0.8,
        "SPY_Return": -2.0
    }
    
    print("\n4. Checking VETO on different market state (Proposing BUY)...")
    is_vetoed, reason = journal.check_veto(features_current_different, "BUY")
    print(f"Vetoed? {is_vetoed}")
    print(f"Reason: {reason}")
    assert is_vetoed == False, "VETO should NOT trigger for a different market state!"
    
    print("\n[+] All VETO Logic Tests Passed Successfully!")

if __name__ == "__main__":
    test_veto_logic()

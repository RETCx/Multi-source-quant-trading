import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
try:
    import google.generativeai as genai
except ImportError:
    pass # Will be handled if needed

class AgenticJournal:
    def __init__(self, journal_path="data/models/trade_journal.json", similarity_threshold=0.90):
        self.journal_path = journal_path
        self.similarity_threshold = similarity_threshold
        self.journal = self._load_journal()
        
    def _load_journal(self):
        if os.path.exists(self.journal_path):
            with open(self.journal_path, 'r') as f:
                return json.load(f)
        return []
        
    def _save_journal(self):
        with open(self.journal_path, 'w') as f:
            json.dump(self.journal, f, indent=4)
            
    def analyze_and_record_mistake(self, date, predicted_action, actual_action, features_dict, gemini_api_key=None):
        """
        Uses LLM to reflect on why a mistake happened and saves the feature snapshot to the journal.
        """
        explanation = "Market reversed unexpectedly."
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                You are an expert quantitative trader.
                Yesterday, our LSTM model predicted {predicted_action}.
                However, the actual market outcome was {actual_action}.
                
                Here were the key market features:
                {json.dumps(features_dict, indent=2)}
                
                Why did the model likely fail? Provide a 1-2 sentence explanation.
                Focus on technical indicator contradictions or market context (like SPY/VIX).
                """
                response = model.generate_content(prompt)
                explanation = response.text.strip()
            except Exception as e:
                print(f"[Warning] LLM reflection failed: {e}")
                
        # Record the mistake
        entry = {
            "date": str(date),
            "predicted": predicted_action,
            "actual": actual_action,
            "explanation": explanation,
            "features_snapshot": features_dict
        }
        self.journal.append(entry)
        self._save_journal()
        print(f"[Journal] Mistake recorded for {date}: {explanation}")
        
    def check_veto(self, current_features_dict, proposed_action):
        """
        Checks if the current market state is very similar to a past mistake.
        If it is, and we are about to make the same mistake, we VETO it.
        """
        if not self.journal:
            return False, ""
            
        current_vec = np.array(list(current_features_dict.values()))
        
        for entry in self.journal:
            # Only veto if we are about to repeat the same action that failed
            if entry['predicted'] != proposed_action:
                continue
                
            past_vec = np.array(list(entry['features_snapshot'].values()))
            
            # Compute Cosine Similarity
            if np.linalg.norm(current_vec) == 0 or np.linalg.norm(past_vec) == 0:
                continue
                
            cos_sim = np.dot(current_vec, past_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(past_vec))
            
            if cos_sim >= self.similarity_threshold:
                reason = f"VETO: State is {cos_sim*100:.1f}% similar to a past mistake on {entry['date']}. Reflection: {entry['explanation']}"
                return True, reason
                
        return False, ""

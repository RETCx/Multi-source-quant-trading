import os
from datetime import datetime

class ExperimentManager:
    """
    ระบบจัดการการเซฟผลลัพธ์ (Artifacts) ของการทดลองแต่ละครั้ง
    จะสร้างโฟลเดอร์รันที่มี Timestamp กำกับให้โดยอัตโนมัติ และเซฟไฟล์ต่างๆ ลงไปในนั้น
    """
    def __init__(self, base_dir="data/models", prefix="run", ticker="UNKNOWN"):
        import shutil
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"{prefix}_{ticker}_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"[Experiment] Created run directory: {self.run_dir}")
        
    def save_dataframe(self, df, filename):
        """เซฟ DataFrame เป็น CSV"""
        path = os.path.join(self.run_dir, filename)
        df.to_csv(path, index=False)
        print(f"  -> Saved CSV: {filename}")
        
    def save_config(self, config_path="config.yaml"):
        """เซฟตั้งค่า Config เพื่อบันทึกว่ารันด้วยค่าแบบไหน"""
        import shutil
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(self.run_dir, "config_used.yaml"))
            print(f"  -> Saved Config: config_used.yaml")
            
    def save_model(self, model, filename="model.pt"):
        """เซฟ Model Weights (.pt)"""
        import torch
        path = os.path.join(self.run_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"  -> Saved Model: {filename}")
        
    def save_text(self, text, filename):
        """เซฟข้อความหรือ Log เป็น Text File"""
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  -> Saved Text: {filename}")

    def save_dict(self, data_dict, filename):
        """เซฟ Dictionary เป็นไฟล์ JSON (เช่น ไฟล์เก็บ Metrics หรือ Hyperparameters)"""
        import json
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)
        print(f"  -> Saved JSON: {filename}")
        
    def save_figure(self, fig, filename):
        """เซฟกราฟ Matplotlib (เช่น Equity Curve, Loss History)"""
        path = os.path.join(self.run_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"  -> Saved Plot: {filename}")

    def save_pickle(self, obj, filename):
        """เซฟ Object ใดๆ เป็นไฟล์ Pickle (เช่น Object ของ WalkForwardCV)"""
        import pickle
        path = os.path.join(self.run_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  -> Saved Pickle: {filename}")

    def save_file(self, source_path, dest_filename=None):
        """ก็อปปี้ไฟล์ดื้อๆ เลย เช่น รูปภาพ CSV อื่นๆ หรือไฟล์อะไรก็ได้ที่มีอยู่แล้ว"""
        import shutil
        if not dest_filename:
            dest_filename = os.path.basename(source_path)
        if os.path.exists(source_path):
            path = os.path.join(self.run_dir, dest_filename)
            shutil.copy(source_path, path)
            print(f"  -> Copied File: {dest_filename}")

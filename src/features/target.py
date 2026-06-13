import numpy as np
import pandas as pd

def get_label_percentile(df, h=5, window=252):
    """
    Rolling Percentile - 2-Class (Binary Classification)
    ถ้าผลตอบแทนในอนาคต h วัน ชนะค่ากลาง (Median) ย้อนหลัง 1 ปี ให้เป็น 1 (น่าเทรด), นอกนั้น 0
    """
    # ใช้ pct_change(h) เพื่อหาผลตอบแทนสะสม h วัน แล้วดึงค่าจากอนาคตกลับมา (shift -h)
    future_return = df['adjClose'].pct_change(periods=h).shift(-h)
    
    # หาค่า Median ย้อนหลัง (ต้อง shift h เพื่อไม่ให้รวมข้อมูลอนาคตตอนคำนวณ ป้องกัน Look-ahead Bias)
    # ใช้ min_periods=1 เพื่อให้รองรับกรณีข้อมูลมีขนาดเล็กกว่า window (เช่น 1 ปี ~ 251 แถว)
    rolling_median = future_return.shift(h).rolling(window=window, min_periods=1).median()
    
    label = pd.Series(np.nan, index=df.index)
    valid = future_return.notna() & rolling_median.notna()
    
    # แปลงเป็น 1.0 (ชนะตลาด) หรือ 0.0 (แพ้ตลาด)
    label[valid] = (future_return[valid] > rolling_median[valid]).astype(float)
    
    return label

def create_multi_horizon_targets(df, horizons=[1, 3, 5, 7, 9], window=252):
    """
    วนลูปสร้าง Target ให้ครบทั้ง 5 ระยะเวลา 
    """
    for h in horizons:
        col_name = f'Target_{h}D'
        df[col_name] = get_label_percentile(df, h=h, window=window)
        
    return df
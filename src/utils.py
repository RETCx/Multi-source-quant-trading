import os
from datetime import datetime

def needs_download(filepath):
    """
    Checks if the file exists and if it was modified today.
    Returns True if it needs to be downloaded, False otherwise.
    """
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        last_mod_date = datetime.fromtimestamp(mtime).date()
        today_date = datetime.today().date()
        
        if last_mod_date == today_date:
            return False
    return True
"""
Download SMS Spam Collection dataset from UCI ML Repository.
Run: python src/download_data.py
"""
import os
import urllib.request
import zipfile
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")
URL      = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="latin-1")
        if len(df) > 100:
            logger.info(f"Dataset already exists ({len(df)} rows). Skipping.")
            return
        else:
            logger.warning("Existing file appears empty. Re-downloading...")
            os.remove(CSV_PATH)

    zip_path = os.path.join(DATA_DIR, "sms_spam.zip")
    logger.info(f"Downloading from {URL}...")
    urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)

    raw_path = os.path.join(DATA_DIR, "SMSSpamCollection")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, sep="\t", header=None, names=["v1", "v2"])
        df.to_csv(CSV_PATH, index=False)
        os.remove(raw_path)
        logger.info(f"Saved {len(df)} rows → {CSV_PATH}")
    else:
        raise FileNotFoundError("Extraction failed. Please download manually from Kaggle.")


if __name__ == "__main__":
    download()

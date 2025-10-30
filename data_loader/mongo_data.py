import pandas as pd
import pymongo

# --- MongoDB connection ---
url = "mongodb+srv://urvirodi_db_user:1234@cluster0.7vtogdj.mongodb.net/?appName=Cluster0"
client = pymongo.MongoClient(url)

database = client["AIML"]
collection = database["fraud_detection"]

# --- Load CSV ---
csv_path = "D:\\CSV Files\\fraud_detection_dataset.csv"
df = pd.read_csv(csv_path)
print(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Insert into MongoDB ---
data_dict = df.to_dict("records")
if data_dict:
    collection.insert_many(data_dict)
    print("💾 CSV data inserted into MongoDB!")
else:
    print("⚠️ No records to insert.")

# --- Verify upload ---
cursor = collection.find()
df1 = pd.DataFrame(list(cursor))
print("\n✅ First few rows from MongoDB:")
print(df1.head())


from pymongo import MongoClient
import pandas as pd
import os

# -------------------------------
# ⚡ User Configuration
# -------------------------------
MONGO_URI = "mongodb+srv://urvirodi_db_user:1234@cluster0.7vtogdj.mongodb.net/" # <-- replace with your URI
DATABASE_NAME = "AIML"       # <-- replace with your DB name
COLLECTION_NAME = "fraud_detection"   # <-- replace with your Collection name

# Output CSV file
OUTPUT_PATH = "data/raw_data.csv"

# -------------------------------
# Ensure data folder exists
# -------------------------------
if not os.path.exists("data"):
    os.makedirs("data")

# -------------------------------
# MongoDB Connection & Data Load
# -------------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Loaded {len(df)} records and saved to {OUTPUT_PATH}")
except Exception as e:
    print(f"❌ Error: {e}")

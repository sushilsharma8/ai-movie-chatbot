import faiss
import numpy as np
import pickle
from sqlalchemy.orm import Session
from main import SessionLocal, MovieScript, get_embedding

# Initialize FAISS index
INDEX_FILE = "faiss_index.bin"
DB_FILE = "database.pkl"

# Load or create FAISS index
try:
    index = faiss.read_index(INDEX_FILE)
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)
    print(f"🔄 Loaded existing FAISS index with {index.ntotal} entries.")
except:
    index = faiss.IndexFlatIP(1536)  # Assuming 1536-dim embeddings
    database = []
    print("🆕 No existing FAISS index found. Creating a new one...")

# Open DB session
session = SessionLocal()

try:
    scripts = session.query(MovieScript).all()

    if scripts:
        for script in scripts:
            embedding = get_embedding(script.dialogue)

            if embedding is not None:
                index.add(np.array([embedding]))  # Add to FAISS
                database.append(script.dialogue)  # Store text
                
        # Save FAISS index & database
        faiss.write_index(index, INDEX_FILE)
        with open(DB_FILE, "wb") as f:
            pickle.dump(database, f)

        print(f"✅ FAISS index populated with {index.ntotal} dialogues.")
    else:
        print("❌ No movie scripts found in NeonDB. Please add some first.")

except Exception as e:
    print(f"⚠️ Error populating FAISS: {e}")

finally:
    session.close()

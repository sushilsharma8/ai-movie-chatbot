import faiss
import numpy as np
import pickle
from main import SessionLocal, MovieScript, get_embedding, index, database

session = SessionLocal()
scripts = session.query(MovieScript).all()

if scripts:
    for script in scripts:
        embedding = get_embedding(script.dialogue)
        index.add(np.array([embedding]))  # Add vector to FAISS
        database.append(script.dialogue)  # Store dialogue text

    print(f"✅ FAISS index populated with {index.ntotal} dialogues.")

    # Save FAISS index & database to a file
    faiss.write_index(index, "faiss_index.bin")
    with open("database.pkl", "wb") as f:
        pickle.dump(database, f)

    print("✅ FAISS data saved to disk.")

else:
    print("❌ No movie scripts found in NeonDB. Please add some first.")

session.close()

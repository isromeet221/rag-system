import sys
import os
from pathlib import Path

# Add partb to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from partb.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_PROPS, COLLECTION_SECTIONS,
    MONGO_URI, MONGO_DB
)

def clear_neo4j():
    try:
        from neo4j import GraphDatabase
        print("Clearing Neo4j...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_lifetime=200, keep_alive=True)
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("Neo4j cleared.")
    except Exception as e:
        print(f"Error clearing Neo4j: {e}")

def clear_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        print("Clearing Qdrant...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        for col in [COLLECTION_PROPS, COLLECTION_SECTIONS]:
            try:
                client.delete(
                    collection_name=col,
                    points_selector=models.Filter()
                )
                print(f"Deleted all points from {col}")
            except Exception as e:
                print(f"Could not clear {col}, maybe it doesn't exist? Error: {e}")
        print("Qdrant cleared.")
    except Exception as e:
        print(f"Error clearing Qdrant: {e}")

def clear_mongo():
    try:
        from pymongo import MongoClient
        print("Clearing MongoDB...")
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collections = db.list_collection_names()
        for col in collections:
            if col != "users":
                db[col].drop()
                print(f"Dropped MongoDB collection: {col}")
            else:
                print(f"Skipping MongoDB collection: {col} (preserved user data)")
        client.close()
        print("MongoDB cleared.")
    except Exception as e:
        print(f"Error clearing MongoDB: {e}")

if __name__ == "__main__":
    clear_neo4j()
    clear_qdrant()
    clear_mongo()
    print("All requested data cleared successfully.")

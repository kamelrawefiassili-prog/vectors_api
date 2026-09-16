import os
import base64
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Visual Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/embeddings"

DB_INDEX = []

def get_jina_embedding(image_input: dict) -> List[float]:
    if not JINA_API_KEY:
        raise Exception("JINA_API_KEY is not configured in Vercel Environment Variables.")
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": "jina-clip-v1",
        "input": [image_input]
    }
    res = requests.post(JINA_URL, headers=headers, json=payload, timeout=15)
    if res.status_code == 200:
        return res.json()["data"][0]["embedding"]
    raise Exception(f"Jina API Error ({res.status_code}): {res.text}")

class IndexRequest(BaseModel):
    product_id: str
    image_urls: List[str]

@app.get("/")
@app.get("/api")
def read_root():
    return {"status": "online", "message": "Visual Search API is ready"}

@app.post("/index-product")
@app.post("/api/index-product")
async def index_product(req: IndexRequest):
    global DB_INDEX
    try:
        DB_INDEX = [item for item in DB_INDEX if item["product_id"] != str(req.product_id)]
        indexed = 0
        for url in req.image_urls:
            vec = get_jina_embedding({"url": url})
            DB_INDEX.append({
                "product_id": str(req.product_id),
                "url": url,
                "embedding": vec
            })
            indexed += 1
        return {"status": "success", "product_id": req.product_id, "indexed_images": indexed}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search-by-image")
@app.post("/api/search-by-image")
async def search_by_image(file: UploadFile = File(...)):
    try:
        if not DB_INDEX:
            return {"status": "success", "decision": "no_confident_match", "candidates": [], "note": "Database index is empty"}
            
        img_bytes = await file.read()
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        query_vec = np.array(get_jina_embedding({"bytes": base64_img}))
        
        candidates = []
        for item in DB_INDEX:
            db_vec = np.array(item["embedding"])
            sim = float(np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec)))
            candidates.append({"product_id": item["product_id"], "similarity": round(sim, 4)})
            
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top = candidates[0] if candidates else None
        decision = "matched" if top and top["similarity"] > 0.80 else "no_confident_match"
        
        return {
            "status": "success",
            "decision": decision,
            "matched_product_id": top["product_id"] if decision == "matched" else None,
            "candidates": candidates[:5]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

import os
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ذاكرة مؤقتة للمتجهات (أو يمكن ربطها بـ Qdrant Cloud)
DB_INDEX = []

def get_clip_embedding(image_bytes: bytes) -> List[float]:
    """استدعاء Hugging Face للحصول على متجه الصورة"""
    response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
    return response.json()

class IndexRequest(BaseModel):
    product_id: str
    image_urls: List[str]

@app.post("/api/index-product")
async def index_product(req: IndexRequest):
    global DB_INDEX
    # حذف البيانات القديمة للمنتج
    DB_INDEX = [item for item in DB_INDEX if item["product_id"] != str(req.product_id)]
    
    indexed = 0
    for url in req.image_urls:
        try:
            img_res = requests.get(url, timeout=10)
            if img_res.status_code == 200:
                vec = get_clip_embedding(img_res.content)
                DB_INDEX.append({
                    "product_id": str(req.product_id),
                    "url": url,
                    "embedding": vec
                })
                indexed += 1
        except Exception as e:
            print(f"Error: {e}")
            
    return {"status": "success", "product_id": req.product_id, "indexed_images": indexed}

@app.post("/api/search-by-image")
async def search_by_image(file: UploadFile = File(...)):
    if not DB_INDEX:
        return {"status": "success", "decision": "no_confident_match", "candidates": []}
        
    img_bytes = await file.read()
    query_vec = np.array(get_clip_embedding(img_bytes))
    
    # حساب التشابه
    candidates = []
    for item in DB_INDEX:
        db_vec = np.array(item["embedding"])
        sim = float(np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec)))
        candidates.append({"product_id": item["product_id"], "similarity": round(sim, 4)})
        
    # ترتيب حسب التشابه
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    top = candidates[0] if candidates else None
    decision = "matched" if top and top["similarity"] > 0.82 else "no_confident_match"
    
    return {
        "status": "success",
        "decision": decision,
        "matched_product_id": top["product_id"] if decision == "matched" else None,
        "candidates": candidates[:5]
    }

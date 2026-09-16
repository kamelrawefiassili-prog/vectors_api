import os
import base64
import requests
import numpy as np
import urllib3
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# تعطيل تحذيرات شهادات SSL غير الموثوقة لاستضافات Awardspace المجانية
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
GET_VECTORS_URL = os.getenv("GET_VECTORS_URL", "https://codfroud.atwebpages.com/get-vectors.php")

CUSTOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def get_jina_embedding(image_input: dict) -> List[float]:
    if not JINA_API_KEY:
        raise Exception("JINA_API_KEY غير مضاف في متغيرات Vercel")
        
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
    raise Exception(f"خطأ Jina AI: {res.text}")

class EmbeddingRequest(BaseModel):
    url: str

@app.get("/")
@app.get("/api")
def read_root():
    return {"status": "online", "message": "Visual Search API is ready"}

@app.post("/get-embedding")
@app.post("/api/get-embedding")
async def get_embedding(req: EmbeddingRequest):
    try:
        vec = get_jina_embedding({"url": req.url})
        return {"status": "success", "embedding": vec}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search-by-image")
@app.post("/api/search-by-image")
async def search_by_image(file: UploadFile = File(...)):
    try:
        # verify=False يتجاوز مشكلة شهادة SSL في Awardspace
        try:
            db_res = requests.get(GET_VECTORS_URL, headers=CUSTOM_HEADERS, verify=False, timeout=10)
            
            if db_res.status_code != 200:
                return {
                    "status": "error", 
                    "message": f"Awardspace rejected request (Code: {db_res.status_code})"
                }
                
            db_json = db_res.json()
            db_data = db_json.get("data", [])
        except Exception as parse_err:
            return {
                "status": "error", 
                "message": f"Response parse error from Awardspace: {str(parse_err)}"
            }

        if not db_data:
            return {
                "status": "success",
                "decision": "no_confident_match",
                "candidates": [],
                "note": "قاعدة البيانات لا تحتوي على أي منتجات بها متجهات"
            }

        img_bytes = await file.read()
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        query_vec = np.array(get_jina_embedding({"bytes": base64_img}))
        
        candidates = []
        for item in db_data:
            db_vec = np.array(item["embedding"])
            sim = float(np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec)))
            candidates.append({"product_id": str(item["product_id"]), "similarity": round(sim, 4)})
            
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

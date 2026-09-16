import os
import json
import base64
import requests
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

JINA_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

# 1. تفعيل ترويسات CORS لإتاحة الاتصال من المتصفح
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

def process_image_to_base64(img_url):
    """تنزيل الصورة في السيرفر وتحويلها لـ Base64 لتجاوز حظر Jina AI أو الاستضافة"""
    if not img_url:
        return None
    if img_url.startswith("data:image"):
        return img_url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        res = requests.get(img_url, headers=headers, timeout=15)
        if res.status_code == 200:
            content_type = res.headers.get('Content-Type', 'image/jpeg')
            if 'image' not in content_type:
                content_type = 'image/jpeg'
            b64_data = base64.b64encode(res.content).decode('utf-8')
            return f"data:{content_type};base64,{b64_data}"
    except Exception as e:
        print(f"Error downloading image {img_url}: {e}")
    
    return img_url

@app.route('/api/get-embedding', methods=['POST', 'OPTIONS'])
def get_embedding():
    # الاستجابة لطلبات Preflight الخفيفة من المتصفح
    if request.method == 'OPTIONS':
        return make_response('', 200)

    try:
        data = request.get_json(force=True) or {}
        raw_image = data.get('image') or data.get('url')

        if not raw_image:
            return jsonify({'status': 'error', 'message': 'بيانات أو رابط الصورة مفقود'}), 400

        # تحويل الصورة إلى Base64 بواسطة Vercel
        image_data = process_image_to_base64(raw_image)

        payload = {
            'model': 'jina-clip-v1',
            'input': [
                {'image': image_data}
            ]
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }

        response = requests.post(JINA_URL, headers=headers, json=payload, timeout=30)
        res_data = response.json()

        if response.status_code == 200 and 'data' in res_data and len(res_data['data']) > 0:
            embedding = res_data['data'][0]['embedding']
            return jsonify({'status': 'success', 'embedding': embedding})
        else:
            return jsonify({'status': 'error', 'message': f"خطأ Jina AI: {json.dumps(res_data, ensure_ascii=False)}"}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()

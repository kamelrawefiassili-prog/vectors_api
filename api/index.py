import os
import json
import base64
import requests
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

JINA_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

def build_cors_response(data, status_code=200):
    """دالة مخصصة لضمان إرفاق ترويسات CORS دائماً حتى في حالات الخطأ"""
    response = make_response(jsonify(data), status_code)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/get-embedding', methods=['POST', 'OPTIONS'])
def get_embedding():
    # 1. المعالجة الفورية لطلبات Preflight الخفيفة من المتصفح
    if request.method == 'OPTIONS':
        return build_cors_response({'status': 'ok'}, 200)

    try:
        req_data = request.get_json(force=True, silent=True) or {}
        raw_input = req_data.get('image') or req_data.get('url')

        if not raw_input:
            return build_cors_response({'status': 'error', 'message': 'بيانات أو رابط الصورة مفقود'}, 400)

        image_payload = None

        # 2. تحديد نوع البيانات الممررة (Base64 أو رابط خارجي)
        if raw_input.startswith("data:image"):
            image_payload = {"image": raw_input}
        else:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                res = requests.get(raw_input, headers=headers, timeout=12)
                if res.status_code == 200 and 'image' in res.headers.get('Content-Type', ''):
                    b64 = base64.b64encode(res.content).decode('utf-8')
                    mime = res.headers.get('Content-Type', 'image/jpeg')
                    image_payload = {"image": f"data:{mime};base64,{b64}"}
                else:
                    image_payload = {"url": raw_input}
            except Exception:
                image_payload = {"url": raw_input}

        # 3. إرسال الطلب إلى Jina AI
        jina_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        jina_payload = {
            'model': 'jina-clip-v1',
            'input': [image_payload]
        }

        jina_res = requests.post(JINA_URL, headers=jina_headers, json=jina_payload, timeout=25)
        jina_json = jina_res.json()

        if jina_res.status_code == 200 and 'data' in jina_json and len(jina_json['data']) > 0:
            embedding = jina_json['data'][0]['embedding']
            return build_cors_response({'status': 'success', 'embedding': embedding})
        else:
            return build_cors_response({
                'status': 'error', 
                'message': f"خطأ Jina AI ({jina_res.status_code}): {json.dumps(jina_json, ensure_ascii=False)}"
            }, 500)

    except Exception as e:
        return build_cors_response({'status': 'error', 'message': f"خطأ غير متوقع في الخادم: {str(e)}"}, 500)

if __name__ == '__main__':
    app.run()

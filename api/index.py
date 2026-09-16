import os
import json
import base64
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# مفتاح Jina AI الخاص بك من بيئة العمل أو تعيينه هنا
JINA_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

def prepare_image_input(img_source):
    """تحضير الصورة وتجاوز حظر استضافات مثل Awardspace بتنزيلها بـ User-Agent"""
    if not img_source:
        return None
    
    if img_source.startswith("http://") or img_source.startswith("https://"):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            resp = requests.get(img_source, headers=headers, timeout=10)
            if resp.status_code == 200:
                b64_img = base64.b64encode(resp.content).decode('utf-8')
                mime = resp.headers.get('Content-Type', 'image/jpeg')
                return {"image": f"data:{mime};base64,{b64_img}"}
        except Exception as e:
            print(f"Error fetching image: {e}")
            
    return {"image": img_source}

@app.route('/api/get-embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.get_json(force=True) or {}
        image_url = data.get('url') or data.get('image')

        if not image_url:
            return jsonify({'status': 'error', 'message': 'رابط الصورة مطلوب'}), 400

        image_obj = prepare_image_input(image_url)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }

        payload = {
            'model': 'jina-clip-v1',
            'input': [image_obj]
        }

        response = requests.post(JINA_URL, headers=headers, json=payload, timeout=20)
        res_data = response.json()

        if response.status_code == 200 and 'data' in res_data:
            embedding = res_data['data'][0]['embedding']
            return jsonify({'status': 'success', 'embedding': embedding})
        else:
            return jsonify({'status': 'error', 'message': f"خطأ Jina AI: {json.dumps(res_data, ensure_ascii=False)}"}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()

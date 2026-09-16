import os
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# مفتاح Jina AI
JINA_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

@app.route('/api/get-embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.get_json(force=True) or {}
        image_data = data.get('image') or data.get('url')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'بيانات الصورة مفقودة أو فارغة'}), 400

        # تجهيز الطلب لـ Jina CLIP v1
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

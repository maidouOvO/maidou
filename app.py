from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pages')
def get_pages():
    output_dir = 'output_images'
    pages = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    return jsonify(pages)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('output_images', filename)

@app.route('/text_data')
def get_text_data():
    text_path = os.path.join('output_images', 'text_data', 'text_coordinates.json')
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Text data not found"})

@app.route('/delete_text', methods=['POST'])
def delete_text():
    try:
        data = request.json
        page_id = data['page']
        text_id = int(data['text_id'])

        text_path = os.path.join('output_images', 'text_data', 'text_coordinates.json')

        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        # Remove the specified text block
        if page_id in text_data['text']:
            text_data['text'][page_id].pop(text_id)

        # Save updated data
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

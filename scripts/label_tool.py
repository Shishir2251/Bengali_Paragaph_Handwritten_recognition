"""
Simple web-based labeling tool for Bangla OCR
View images and add text labels
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from pathlib import Path

app = Flask(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_SPLITS = ['train', 'val', 'test']

def load_annotations(split):
    """Load annotations for a split"""
    ann_file = PROJECT_ROOT / 'data' / split / 'annotations.json'
    if ann_file.exists():
        with open(ann_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_annotations(split, annotations):
    """Save annotations for a split"""
    ann_file = PROJECT_ROOT / 'data' / split / 'annotations.json'
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('labeling.html')

@app.route('/api/get_image/<split>/<int:index>')
def get_image(split, index):
    """Get image info for labeling"""
    annotations = load_annotations(split)
    
    if index >= len(annotations):
        return jsonify({'done': True})
    
    item = annotations[index]
    img_path = f'/images/{split}/{item["image"]}'
    
    return jsonify({
        'done': False,
        'index': index,
        'total': len(annotations),
        'image': img_path,
        'current_text': item['text'],
        'filename': item['image']
    })

@app.route('/api/save_label', methods=['POST'])
def save_label():
    """Save a label"""
    data = request.json
    split = data['split']
    index = data['index']
    text = data['text']
    
    annotations = load_annotations(split)
    annotations[index]['text'] = text
    save_annotations(split, annotations)
    
    return jsonify({'success': True})

@app.route('/api/stats')
def stats():
    """Get labeling statistics"""
    stats = {}
    for split in DATA_SPLITS:
        annotations = load_annotations(split)
        labeled = sum(1 for a in annotations if a['text'] and a['text'] != 'LABEL_NEEDED')
        stats[split] = {
            'total': len(annotations),
            'labeled': labeled,
            'unlabeled': len(annotations) - labeled
        }
    return jsonify(stats)

@app.route('/images/<split>/<filename>')
def serve_image(split, filename):
    """Serve image files"""
    from flask import send_file
    img_path = PROJECT_ROOT / 'data' / split / filename
    return send_file(img_path)

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create HTML template
    html_content = '''
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla OCR Labeling Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #333; margin-bottom: 20px; text-align: center; }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .stat-card {
            flex: 1;
            min-width: 200px;
            background: #f0f2ff;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .stat-card h3 { color: #667eea; font-size: 1rem; margin-bottom: 10px; }
        .stat-card .number { font-size: 2rem; font-weight: bold; color: #333; }
        .split-selector {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        .split-btn {
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
        }
        .split-btn.active {
            background: #667eea;
            color: white;
        }
        .labeling-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 20px;
        }
        .image-preview {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background: #f9f9f9;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 400px;
        }
        .image-preview img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
        }
        .label-input-area {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .progress {
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
        }
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1.2rem;
            font-family: 'Kalpurush', 'SolaimanLipi', sans-serif;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .info { background: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .info strong { color: #667eea; }
        .buttons {
            display: flex;
            gap: 10px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-secondary {
            background: #f0f2ff;
            color: #667eea;
        }
        .btn-secondary:hover { background: #e8ebff; }
        .shortcuts {
            background: #fff9e6;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .shortcuts h3 { color: #f39c12; margin-bottom: 10px; }
        .shortcuts ul { list-style: none; }
        .shortcuts li { padding: 5px 0; }
        .shortcuts kbd {
            background: #333;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇧🇩 Bangla OCR Labeling Tool</h1>
        
        <div class="stats" id="stats"></div>
        
        <div class="split-selector">
            <button class="split-btn active" data-split="train">Train</button>
            <button class="split-btn" data-split="val">Validation</button>
            <button class="split-btn" data-split="test">Test</button>
        </div>
        
        <div class="progress">
            <div class="progress-bar" id="progress">0 / 0</div>
        </div>
        
        <div class="labeling-area">
            <div class="image-preview">
                <img id="image" src="" alt="Loading...">
            </div>
            
            <div class="label-input-area">
                <div class="info">
                    <strong>File:</strong> <span id="filename">-</span><br>
                    <strong>Index:</strong> <span id="current-index">0</span> / <span id="total-count">0</span>
                </div>
                
                <label><strong>Bangla Text:</strong></label>
                <textarea id="text-input" placeholder="আপনার বাংলা টেক্সট এখানে লিখুন..."></textarea>
                
                <div class="buttons">
                    <button class="btn-secondary" onclick="previous()">⬅ Previous</button>
                    <button class="btn-primary" onclick="saveAndNext()">Save & Next ➡</button>
                </div>
                
                <button class="btn-secondary" onclick="skip()">Skip (Keep Current)</button>
            </div>
        </div>
        
        <div class="shortcuts">
            <h3> Keyboard Shortcuts</h3>
            <ul>
                <li><kbd>Ctrl</kbd> + <kbd>Enter</kbd> - Save and Next</li>
                <li><kbd>Ctrl</kbd> + <kbd>←</kbd> - Previous</li>
                <li><kbd>Ctrl</kbd> + <kbd>→</kbd> - Skip</li>
            </ul>
        </div>
    </div>
    
    <script>
        let currentSplit = 'train';
        let currentIndex = 0;
        
        async function loadStats() {
            const res = await fetch('/api/stats');
            const stats = await res.json();
            
            const statsHtml = Object.entries(stats).map(([split, data]) => `
                <div class="stat-card">
                    <h3>${split.toUpperCase()}</h3>
                    <div class="number">${data.labeled} / ${data.total}</div>
                    <small>${data.unlabeled} unlabeled</small>
                </div>
            `).join('');
            
            document.getElementById('stats').innerHTML = statsHtml;
        }
        
        async function loadImage(index) {
            const res = await fetch(`/api/get_image/${currentSplit}/${index}`);
            const data = await res.json();
            
            if (data.done) {
                alert(`All ${currentSplit} images labeled! `);
                await loadStats();
                return;
            }
            
            currentIndex = data.index;
            document.getElementById('image').src = data.image;
            document.getElementById('filename').textContent = data.filename;
            document.getElementById('text-input').value = data.current_text;
            document.getElementById('current-index').textContent = data.index + 1;
            document.getElementById('total-count').textContent = data.total;
            
            const progress = ((data.index + 1) / data.total) * 100;
            document.getElementById('progress').style.width = progress + '%';
            document.getElementById('progress').textContent = `${data.index + 1} / ${data.total}`;
            
            document.getElementById('text-input').focus();
        }
        
        async function saveLabel() {
            const text = document.getElementById('text-input').value;
            
            await fetch('/api/save_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    split: currentSplit,
                    index: currentIndex,
                    text: text
                })
            });
            
            await loadStats();
        }
        
        async function saveAndNext() {
            await saveLabel();
            await loadImage(currentIndex + 1);
        }
        
        async function previous() {
            if (currentIndex > 0) {
                await loadImage(currentIndex - 1);
            }
        }
        
        async function skip() {
            await loadImage(currentIndex + 1);
        }
        
        // Split selector
        document.querySelectorAll('.split-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.split-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentSplit = btn.dataset.split;
                loadImage(0);
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                saveAndNext();
            }
            if (e.ctrlKey && e.key === 'ArrowLeft') {
                e.preventDefault();
                previous();
            }
            if (e.ctrlKey && e.key === 'ArrowRight') {
                e.preventDefault();
                skip();
            }
        });
        
        // Initialize
        loadStats();
        loadImage(0);
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / 'labeling.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("="*70)
    print("  Bangla OCR Labeling Tool")
    print("="*70)
    print()
    print("Starting server...")
    print("Open: http://localhost:5001")
    print()
    print("Instructions:")
    print("  1. View each image")
    print("  2. Type the Bangla text you see")
    print("  3. Click 'Save & Next'")
    print("  4. Repeat for all images")
    print()
    print("Keyboard Shortcuts:")
    print("  Ctrl+Enter - Save & Next")
    print("  Ctrl+← - Previous")
    print("  Ctrl+→ - Skip")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
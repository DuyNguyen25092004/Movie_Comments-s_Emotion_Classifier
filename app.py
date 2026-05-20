from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sqlite3

app = Flask(__name__)
# Cho phép frontend từ GitHub Pages có thể gọi tới backend này
CORS(app)

# Tải mô hình AI của bạn
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('svm_model.pkl')

DB_FILE = 'reviews.db'

# Dữ liệu mẫu ban đầu để nạp nếu database trống
SEED_DATA = {
    1: [
        {"text": "Mind-blowing concept! The layers of dreams are incredibly creative.", "label": 1, "confidence": 0.97},
        {"text": "Too confusing and convoluted. I lost track of what was real.", "label": 0, "confidence": 0.82},
        {"text": "A masterpiece of modern cinema. Christopher Nolan at his finest!", "label": 1, "confidence": 0.99},
    ],
    2: [
        {"text": "Heath Ledger's Joker is one of the greatest performances ever.", "label": 1, "confidence": 0.99},
        {"text": "Overrated. The hype doesn't match the actual experience.", "label": 0, "confidence": 0.78},
    ],
    3: [
        {"text": "Visually stunning and emotionally powerful. Science is fascinating.", "label": 1, "confidence": 0.95},
        {"text": "The ending was a huge disappointment that ruined the whole film.", "label": 0, "confidence": 0.88},
        {"text": "One of the most beautiful movies I've seen in IMAX.", "label": 1, "confidence": 0.96},
    ],
    4: [
        {"text": "Bong Joon-ho crafted something truly unique and thought-provoking.", "label": 1, "confidence": 0.97},
        {"text": "Absolutely deserved every award. Social commentary at its best.", "label": 1, "confidence": 0.98},
    ],
    5: [
        {"text": "Denis Villeneuve has created an epic sci-fi masterpiece!", "label": 1, "confidence": 0.97},
        {"text": "Zendaya and Timothée Chalamet are absolutely mesmerizing together.", "label": 1, "confidence": 0.93},
    ],
    6: [
        {"text": "Cillian Murphy delivers a career-defining performance.", "label": 1, "confidence": 0.96},
        {"text": "Three hours of pure cinematic brilliance. Nolan's best work.", "label": 1, "confidence": 0.95},
        {"text": "Overly long and self-indulgent. Could be cut by an hour.", "label": 0, "confidence": 0.81},
    ],
    7: [
        {"text": "Emma Stone is absolutely extraordinary. Bizarre yet brilliant.", "label": 1, "confidence": 0.94},
        {"text": "Deeply disturbing content wrapped in a pretentious package.", "label": 0, "confidence": 0.85},
    ],
    8: [
        {"text": "Haunting and profound. A necessary film about the banality of evil.", "label": 1, "confidence": 0.91},
        {"text": "Slow and hard to watch, but that's precisely the point.", "label": 1, "confidence": 0.72},
    ]
}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            text TEXT,
            label INTEGER,
            confidence REAL,
            reason TEXT
        )
    ''')
    
    # Nếu database chưa có dữ liệu, tiến hành nạp SEED_DATA
    cursor.execute('SELECT COUNT(*) FROM reviews')
    if cursor.fetchone()[0] == 0:
        for movie_id, reviews in SEED_DATA.items():
            for r in reviews:
                cursor.execute('''
                    INSERT INTO reviews (movie_id, text, label, confidence, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (movie_id, r['text'], r['label'], r['confidence'], 'Mẫu hệ thống'))
        conn.commit()
    conn.close()

# Khởi tạo database
init_db()

@app.route('/api/reviews', methods=['GET'])
def get_all_reviews():
    """Endpoint trả về toàn bộ bình luận được nhóm theo movie_id để Frontend đồng bộ cấu trúc"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT movie_id, text, label, confidence, reason FROM reviews')
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            m_id = str(row[0])
            if m_id not in result:
                result[m_id] = []
            result[m_id].append({
                "text": row[1],
                "label": row[2],
                "confidence": row[3],
                "reason": row[4]
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint nhận bình luận mới, phân loại bằng AI và lưu thẳng vào SQLite"""
    try:
        data = request.json
        text = data.get('text', '')
        movie_id = data.get('movie_id')
        
        if not text or movie_id is None:
            return jsonify({"error": "Thiếu nội dung bình luận hoặc ID phim"}), 400
            
        # Xử lý văn bản qua mô hình AI của bạn
        vec_text = vectorizer.transform([text])
        prediction = int(model.predict(vec_text)[0])
        
        try:
            confidence = float(max(model.predict_proba(vec_text)[0]))
        except:
            confidence = 0.85
            
        reason = "Phân loại bằng mô hình học máy cục bộ (LinearSVC)"
        
        # Lưu vào SQLite Database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reviews (movie_id, text, label, confidence, reason)
            VALUES (?, ?, ?, ?, ?)
        ''', (int(movie_id), text, prediction, confidence, reason))
        conn.commit()
        conn.close()
        
        return jsonify({
            "label": prediction,
            "confidence": confidence,
            "reason": reason
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

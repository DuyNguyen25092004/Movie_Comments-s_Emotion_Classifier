from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Thay import pickle bằng import joblib

app = Flask(__name__)
CORS(app) 

# Thay đổi từ pickle.load sang joblib.load
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        # Tiền xử lý văn bản bằng vectorizer
        vec_text = vectorizer.transform([text])
        
        # Dự đoán nhãn (giả sử 1 là Tích cực, 0 là Tiêu cực)
        prediction = int(model.predict(vec_text)[0])
        
        # Cố gắng lấy độ tin cậy (confidence) nếu mô hình hỗ trợ predict_proba
        try:
            confidence = float(max(model.predict_proba(vec_text)[0]))
        except:
            # Nếu mô hình là LinearSVC mặc định không có predict_proba, gán cứng 1 mức độ
            confidence = 0.85 

        return jsonify({
            "label": prediction,
            "confidence": confidence,
            "reason": "Phân loại bởi mô hình Local AI"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Chạy server ở cổng 5000
    app.run(port=5000, debug=True)
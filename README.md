# 🎬 Movie Review Sentiment Analysis (FastAPI + Machine Learning)

## 📖 Overview
This project performs **sentiment analysis on movie reviews**, predicting whether a review is **positive or negative** using multiple machine learning and deep learning models.

The models were trained in **Google Colab** using several techniques and exported for use in a **FastAPI backend**.

---

## 🧠 Models Used
- **Logistic Regression**
- **Naive Bayes**
- **Multi-Layer Perceptron (Keras)**
- **BERT**, **DistilBERT**, and **RoBERT** (Transformers)


---

## ⚙️ Tech Stack
**Backend:** FastAPI (Python)   
**Environment:** Google Colab, Visual Studio Code  


---

## 🚀 How It Works
1. The trained models (stored locally in `backend/models/`) classify movie reviews as positive or negative.
2. The backend exposes a REST API built with **FastAPI**.
3. Clients (like a frontend React app or Colab notebook) can send reviews via POST requests.
4. The API returns the sentiment prediction with confidence score.

---


---

## 🧠 Training & Model Export
Training was performed in **Google Colab** using datasets of labeled movie reviews.  
After training, each model was saved locally using:
```python
joblib.dump(model, 'model_name.pkl')



🏃‍♂️ How to Run the Project Locally
🔧 Backend (FastAPI)

Open a terminal and navigate to the backend folder:

cd C:\Users\Kela\Desktop\ml-webapp\backend


Activate the virtual environment:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Users\Kela\Desktop\ml-webapp\venv\Scripts\Activate.ps1"


Run the FastAPI backend server:

uvicorn app:app --reload


Once running, open your browser and go to:
👉 http://127.0.0.1:8000/docs

to access the interactive Swagger API documentation.




🌐 Frontend (React or Static HTML)

In a new terminal window, navigate to the frontend folder:

cd C:\Users\Kela\Desktop\ml-webapp\frontend


Run a simple development server:

python -m http.server 5500


Open your browser and visit:
👉 http://127.0.0.1:5500

Your frontend will now communicate with the backend running on port 8000.

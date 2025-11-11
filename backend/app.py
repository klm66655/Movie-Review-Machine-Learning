import pickle
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 🔹 Hugging Face modeli
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import torch
import torch.nn.functional as F


# ================================
# 🔹 FastAPI app + CORS
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
  # frontend adresa
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# 🔹 Učitaj tradicionalne modele
# ================================
models = {
    "naive_bayes": pickle.load(open("models/naive_bayes_model.pkl", "rb")),
    "logistic": pickle.load(open("models/logistic_pipeline.pkl", "rb")),
    "mlp": load_model("models/mlp_model.keras"),
}

tfidf = pickle.load(open("models/tfidf.pkl", "rb"))


# ================================
# 🔹 Učitaj Transformers modele
# ================================
# BERT
bert_path = "models/bert_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()

# DistilBERT
distilbert_path = "models/distilbert_model"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
distilbert_model.eval()

# ✅ RoBERTa (novi model)
roberta_path = "models/roberta_sentiment_model"
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_path)
roberta_model.eval()


# ================================
# 🔹 Pydantic input model
# ================================
class ReviewInput(BaseModel):
    model: str
    review: str


# ================================
# 🔹 API rute
# ================================
@app.get("/")
def home():
    return {"message": "API is running!"}


@app.get("/models")
def get_models():
    return {"available_models": list(models.keys()) + ["bert", "distilbert", "roberta"]}


@app.post("/predict")
def predict(data: ReviewInput):
    text = [data.review]
    model_choice = data.model.lower()

    # --- NAIVE BAYES ---
    if model_choice == "naive_bayes":
        pred_class = models["naive_bayes"].predict(text)[0]
        pred_probs = models["naive_bayes"].predict_proba(text)[0]
        class_mapping = {0: "negative", 1: "positive"}
        pred_text = class_mapping[int(pred_class)]
        recommendation = (
            "Positive review – recommended!"
            if pred_text == "positive"
            else "Negative review – not recommended."
        )

        return {
            "model": model_choice,
            "review": data.review,
            "prediction": pred_text,
            "probability_negative": round(float(pred_probs[0]), 4),
            "probability_positive": round(float(pred_probs[1]), 4),
            "recommendation": recommendation,
        }

    # --- LOGISTIC REGRESSION ---
    elif model_choice == "logistic":
        pred_probs = models["logistic"].predict_proba(text)[0]
        pred_class = models["logistic"].predict(text)[0]
        pred_text = "positive" if int(pred_class) == 1 else "negative"
        probability_positive = float(pred_probs[1])
        probability_negative = float(pred_probs[0])
        recommendation = (
            "Positive review – recommended!"
            if pred_text == "positive"
            else "Negative review – not recommended."
        )

        return {
            "model": model_choice,
            "review": data.review,
            "prediction": pred_text,
            "probability_positive": round(probability_positive, 4),
            "probability_negative": round(probability_negative, 4),
            "recommendation": recommendation,
        }

    # --- MLP MODEL ---
    elif model_choice == "mlp":
        X = tfidf.transform([data.review]).toarray()
        pred_prob = float(models["mlp"].predict(X)[0][0])
        pred_text = "positive" if pred_prob >= 0.5 else "negative"
        recommendation = (
            "Positive review – recommended!"
            if pred_text == "positive"
            else "Negative review – not recommended."
        )

        return {
            "model": model_choice,
            "review": data.review,
            "prediction": pred_text,
            "probability_positive": round(pred_prob, 4),
            "probability_negative": round(1 - pred_prob, 4),
            "recommendation": recommendation,
        }

    # --- BERT MODEL ---
    elif model_choice == "bert":
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

        pred_class = int(probs.argmax())
        pred_text = "positive" if pred_class == 1 else "negative"
        recommendation = "Positive review – recommended!" if pred_text == "positive" else "Negative review – not recommended."

        return {
            "model": "bert",
            "review": data.review,
            "prediction": pred_text,
            "probability_negative": round(float(probs[0]), 4),
            "probability_positive": round(float(probs[1]), 4),
            "recommendation": recommendation,
        }

    # --- DISTILBERT MODEL ---
    elif model_choice == "distilbert":
        inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

        pred_class = int(probs.argmax())
        pred_text = "positive" if pred_class == 1 else "negative"
        recommendation = "Positive review – recommended!" if pred_text == "positive" else "Negative review – not recommended."

        return {
            "model": "distilbert",
            "review": data.review,
            "prediction": pred_text,
            "probability_negative": round(float(probs[0]), 4),
            "probability_positive": round(float(probs[1]), 4),
            "recommendation": recommendation,
        }

    # ✅ --- ROBERT MODEL ---
    elif model_choice == "roberta":
        inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

        pred_class = int(probs.argmax())
        pred_text = "positive" if pred_class == 1 else "negative"
        recommendation = "Positive review – recommended!" if pred_text == "positive" else "Negative review – not recommended."

        return {
            "model": "roberta",
            "review": data.review,
            "prediction": pred_text,
            "probability_negative": round(float(probs[0]), 4),
            "probability_positive": round(float(probs[1]), 4),
            "recommendation": recommendation,
        }

    # --- NEPOZNAT MODEL ---
    else:
        return {"error": f"Unknown model '{model_choice}'. Use one of: {list(models.keys()) + ['bert', 'distilbert', 'roberta']}"}

from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from models import db, Analysis
from datetime import datetime
import pickle
import re

dashboard_bp = Blueprint('dashboard', __name__)

# Load ML model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    print("Warning: Model files not found. Please run train_model.py first.")
    model = None
    vectorizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

@dashboard_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None
    
    if request.method == 'POST':
        text = request.form.get("text")
        if text and model and vectorizer:
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            result = model.predict(vectorized)[0]
            
            if result == 1:
                prediction = "Positive Mental State 😊"
                sentiment = "positive"
            elif result == 0:
                prediction = "Neutral Mental State 😐"
                sentiment = "neutral"
            else:
                prediction = "Negative Mental State 😔"
                sentiment = "negative"
            
            # Save analysis to database
            analysis = Analysis(
                user_id=current_user.id,
                text=text[:500],  # Store first 500 chars
                prediction=prediction,
                sentiment=sentiment
            )
            db.session.add(analysis)
            db.session.commit()
        elif not model or not vectorizer:
            flash("ML model not loaded. Please check if model.pkl and vectorizer.pkl exist.", "error")
    
    # Get user's analysis history
    analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.created_at.desc())\
        .limit(20)\
        .all()
    
    return render_template('dashboard.html', 
                         prediction=prediction, 
                         analyses=analyses)
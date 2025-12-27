from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from models import db, Analysis
from datetime import datetime
import re

# Import model and vectorizer from ml_model (NO circular import)
from ml_model import model, vectorizer

dashboard_bp = Blueprint('dashboard', __name__)

# ==============================
# Utility
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# ==============================
# Routes
# ==============================
@dashboard_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None

    if request.method == 'POST':
        text = request.form.get("text")

        # Validate input
        if not text or not text.strip():
            flash("Please enter text to analyze.", "error")

        elif model is None or vectorizer is None:
            flash(
                "ML model not loaded. Please check model.pkl and vectorizer.pkl.",
                "error"
            )

        else:
            cleaned = clean_text(text)
            try:
                vectorized = vectorizer.transform([cleaned])
                result = model.predict(vectorized)[0]
            except Exception as e:
                print(f"Prediction error: {e}")
                flash(
                    "An error occurred while analyzing text. Please try again.",
                    "error"
                )
                result = None

            if result is not None:
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
                    text=text[:500],
                    prediction=prediction,
                    sentiment=sentiment
                )

                try:
                    db.session.add(analysis)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"DB save error: {e}")
                    flash(
                        "Failed to save analysis. Please try again.",
                        "error"
                    )

    # Fetch analysis history
    analyses = (
        Analysis.query
        .filter_by(user_id=current_user.id)
        .order_by(Analysis.created_at.desc())
        .limit(20)
        .all()
    )

    return render_template(
        'dashboard.html',
        prediction=prediction,
        analyses=analyses
    )

import os
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, current_user
from models import db, User
from auth import auth_bp
from dashboard import dashboard_bp
import pickle
import re

app = Flask(__name__)

# Railway PostgreSQL configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Get database URL from Railway environment variable
database_url = os.environ.get('DATABASE_URL')

if database_url:
    # Fix for Railway PostgreSQL URL format
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Fallback for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(dashboard_bp)

# Load ML model and vectorizer
def load_ml_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_ml_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    # Redirect to dashboard if user is logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    # For non-logged in users, show the original index with analysis
    prediction = ""
    if request.method == "POST" and model and vectorizer:
        text = request.form.get("text")
        if text:
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            result = model.predict(vectorized)[0]

            if result == 1:
                prediction = "Positive Mental State 😊"
            elif result == 0:
                prediction = "Neutral Mental State 😐"
            else:
                prediction = "Negative Mental State 😔"
    
    return render_template("index.html", prediction=prediction)

# Create database tables on app startup
with app.app_context():
    db.create_all()
    print("Database tables created/verified")

if __name__ == "__main__":
    # Get port from Railway environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
import os
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, current_user
from models import db, User
from auth import auth_bp
from dashboard import dashboard_bp
import pickle
import re

app = Flask(__name__)

# Render.com PostgreSQL configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Get database URL from Render environment variable
database_url = os.environ.get('DATABASE_URL')

if database_url:
    # Fix for Render PostgreSQL URL format
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print("Using PostgreSQL database from DATABASE_URL")
else:
    # Fallback for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'
    print("Using SQLite database for local development")

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

# Load ML model and vectorizer (keep using pickle)
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("ML model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model, vectorizer = None, None

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
    try:
        db.create_all()
        print("Database tables created/verified")
    except Exception as e:
        print(f"Database initialization error: {e}")

if __name__ == "__main__":
    # Render automatically sets PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
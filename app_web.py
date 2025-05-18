from flask import Flask, request, render_template_string, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
from spacy.language import Language
from transformers import pipeline
from spacy.tokens import Doc

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Flask-Login manager
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Sentiment model linked to user
class SentimentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(20), nullable=False)
    score = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# spaCy + transformers setup
sentiment_analyzer = pipeline("sentiment-analysis")

@Language.component("sentiment_component")
def sentiment_component(doc):
    result = sentiment_analyzer(doc.text)[0]
    doc._.sentiment = result
    return doc

Doc.set_extension("sentiment", default=None)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentiment_component", last=True)

# HTML Templates (use triple quotes for readability)
BASE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>{% block title %}Sentiment Analysis{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
<div class="container mt-5">
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-info" role="alert">
          {{ messages[0] }}
        </div>
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</div>
</body>
</html>
'''

HOME_HTML = '''
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<h1 class="mb-4">Welcome, {{ current_user.username }}!</h1>
<a href="{{ url_for('logout') }}" class="btn btn-danger mb-3">Logout</a>
<form method="POST" action="{{ url_for('analyze') }}">
    <div class="mb-3">
        <textarea class="form-control" name="text" rows="6" placeholder="Type your text here...">{{ text }}</textarea>
    </div>
    <button type="submit" class="btn btn-primary">Analyze & Save</button>
</form>

{% if sentiment %}
<div class="mt-4">
    <h4>Result:</h4>
    <p><strong>Text:</strong> {{ text }}</p>
    <p><strong>Sentiment:</strong> {{ sentiment['label'] }} (Confidence: {{ "%.2f"|format(sentiment['score']) }})</p>
</div>
{% endif %}

<hr>
<h4>Your Saved Analyses:</h4>
{% if saved_texts %}
    <ul class="list-group">
    {% for item in saved_texts %}
        <li class="list-group-item">
            <strong>Text:</strong> {{ item.text }}<br>
            <strong>Sentiment:</strong> {{ item.label }} ({{ "%.2f"|format(item.score) }})
        </li>
    {% endfor %}
    </ul>
{% else %}
    <p>No saved analyses yet.</p>
{% endif %}
{% endblock %}
'''

LOGIN_HTML = '''
{% extends "base.html" %}
{% block title %}Login{% endblock %}
{% block content %}
<h2>Login</h2>
<form method="POST" action="{{ url_for('login') }}">
  <div class="mb-3">
    <input type="text" class="form-control" name="username" placeholder="Username" required>
  </div>
  <div class="mb-3">
    <input type="password" class="form-control" name="password" placeholder="Password" required>
  </div>
  <button type="submit" class="btn btn-primary">Login</button>
</form>
<p class="mt-3">Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
{% endblock %}
'''

REGISTER_HTML = '''
{% extends "base.html" %}
{% block title %}Register{% endblock %}
{% block content %}
<h2>Register</h2>
<form method="POST" action="{{ url_for('register') }}">
  <div class="mb-3">
    <input type="text" class="form-control" name="username" placeholder="Username" required>
  </div>
  <div class="mb-3">
    <input type="password" class="form-control" name="password" placeholder="Password" required>
  </div>
  <button type="submit" class="btn btn-success">Register</button>
</form>
<p class="mt-3">Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
{% endblock %}
'''

from flask import render_template_string
from flask_login import login_user, logout_user, login_required, current_user

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        username = request.form['username'].strip()
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
            return redirect(url_for('register'))
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))
    return render_template_string(REGISTER_HTML)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        username = request.form['username'].strip()
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully!")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))
    return render_template_string(LOGIN_HTML)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route("/", methods=["GET"])
@login_required
def home():
    saved_texts = SentimentRecord.query.filter_by(user_id=current_user.id).order_by(SentimentRecord.id.desc()).all()
    return render_template_string(HOME_HTML, text="", sentiment=None, saved_texts=saved_texts)

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Please enter some text to analyze.")
        return redirect(url_for("home"))
    doc = nlp(text)
    sentiment = doc._.sentiment

    record = SentimentRecord(text=text, label=sentiment['label'], score=sentiment['score'], user_id=current_user.id)
    db.session.add(record)
    db.session.commit()

    flash("Analysis saved successfully!")
    saved_texts = SentimentRecord.query.filter_by(user_id=current_user.id).order_by(SentimentRecord.id.desc()).all()
    return render_template_string(HOME_HTML, text=text, sentiment=sentiment, saved_texts=saved_texts)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

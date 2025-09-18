import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Flask for web interface
from flask import Flask, render_template, request, jsonify, send_from_directory
import sqlite3
from werkzeug.security import generate_password_hash

# OpenAI integration
import openai
from openai import OpenAI

# Env loader
from dotenv import load_dotenv
load_dotenv()

# ML models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("Warning: Some NLTK resources couldn't be downloaded")

class EnhancedTicketClassifier:
    """
    Enhanced AI Support Ticket Classification System with OpenAI Integration
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            
        self.category_model = None
        self.priority_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.priority_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Enhanced categories
        self.categories = {
            'authentication': {
                'keywords': ['password', 'login', 'signin', 'authenticate', 'credentials', 'access denied', 'locked out'],
                'priority_default': 'high',
                'description': 'Login issues, password resets, authentication problems'
            },
            'hr_query': {
                'keywords': ['leave', 'balance', 'vacation', 'hr', 'payroll', 'benefits', 'holidays', 'sick leave'],
                'priority_default': 'low',
                'description': 'HR-related questions, leave balance, payroll inquiries'
            },
            'technical_support': {
                'keywords': ['system', 'error', 'bug', 'crash', 'software', 'application', 'not working', 'broken'],
                'priority_default': 'medium',
                'description': 'Technical issues, software bugs, system errors'
            },
            'access_request': {
                'keywords': ['access', 'permission', 'role', 'rights', 'privileges', 'authorization', 'vpn'],
                'priority_default': 'medium',
                'description': 'Access permissions, role changes, VPN requests'
            },
            'hardware_support': {
                'keywords': ['laptop', 'desktop', 'printer', 'keyboard', 'mouse', 'monitor', 'hardware'],
                'priority_default': 'medium',
                'description': 'Hardware issues, equipment requests, device problems'
            },
            'network_connectivity': {
                'keywords': ['internet', 'network', 'wifi', 'connection', 'connectivity', 'slow internet'],
                'priority_default': 'high',
                'description': 'Network issues, connectivity problems, internet speed'
            },
            'general_inquiry': {
                'keywords': ['question', 'help', 'support', 'inquiry', 'information', 'how to'],
                'priority_default': 'low',
                'description': 'General questions, information requests, guidance'
            }
        }
        
        # Sample training data
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self) -> List[Dict]:
        """Generate comprehensive sample training data"""
        return [
            # Authentication issues
            {"text": "I forgot my password and can't login to the system", "category": "authentication", "priority": "high"},
            {"text": "Password reset not working, please help", "category": "authentication", "priority": "high"},
            {"text": "My account is locked after multiple failed attempts", "category": "authentication", "priority": "high"},
            {"text": "Can't sign in with my credentials", "category": "authentication", "priority": "medium"},
            {"text": "Login page shows error message", "category": "authentication", "priority": "medium"},
            
            # HR queries
            {"text": "How to check my leave balance?", "category": "hr_query", "priority": "low"},
            {"text": "When will I receive my payslip?", "category": "hr_query", "priority": "low"},
            {"text": "Need information about health benefits", "category": "hr_query", "priority": "low"},
            {"text": "How many vacation days do I have left?", "category": "hr_query", "priority": "low"},
            {"text": "Sick leave policy clarification needed", "category": "hr_query", "priority": "low"},
            
            # Technical support
            {"text": "Application crashes when I try to save files", "category": "technical_support", "priority": "high"},
            {"text": "Software is running very slowly today", "category": "technical_support", "priority": "medium"},
            {"text": "Getting error message in the system", "category": "technical_support", "priority": "medium"},
            {"text": "Bug in the reporting module", "category": "technical_support", "priority": "medium"},
            {"text": "System not responding properly", "category": "technical_support", "priority": "high"},
            
            # Access requests
            {"text": "Need access to shared folder", "category": "access_request", "priority": "medium"},
            {"text": "Request permission for new project database", "category": "access_request", "priority": "medium"},
            {"text": "VPN access required for remote work", "category": "access_request", "priority": "medium"},
            {"text": "Role change - need admin privileges", "category": "access_request", "priority": "high"},
            {"text": "Access denied to customer portal", "category": "access_request", "priority": "medium"},
            
            # Hardware support
            {"text": "My laptop screen is flickering", "category": "hardware_support", "priority": "medium"},
            {"text": "Printer not working in conference room", "category": "hardware_support", "priority": "low"},
            {"text": "Need new keyboard, current one broken", "category": "hardware_support", "priority": "low"},
            {"text": "Monitor display issues", "category": "hardware_support", "priority": "medium"},
            {"text": "Mouse not responding", "category": "hardware_support", "priority": "low"},
            
            # Network connectivity
            {"text": "Internet connection is very slow", "category": "network_connectivity", "priority": "high"},
            {"text": "Can't connect to office WiFi", "category": "network_connectivity", "priority": "high"},
            {"text": "Network keeps disconnecting", "category": "network_connectivity", "priority": "high"},
            {"text": "No internet access on my workstation", "category": "network_connectivity", "priority": "high"},
            {"text": "VPN connection unstable", "category": "network_connectivity", "priority": "medium"},
            
            # General inquiries
            {"text": "How to use the new CRM system?", "category": "general_inquiry", "priority": "low"},
            {"text": "Need help with file sharing process", "category": "general_inquiry", "priority": "low"},
            {"text": "General question about company policies", "category": "general_inquiry", "priority": "low"},
            {"text": "Information about upcoming system maintenance", "category": "general_inquiry", "priority": "low"},
            {"text": "How to submit expense reports?", "category": "general_inquiry", "priority": "low"},
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stop_words]
        
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        except:
            pass
        
        return ' '.join(tokens)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features using TF-IDF vectorization"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)
    
    def train_models(self):
        """Train classification models"""
        print("Training AI models...")
        
        df = pd.DataFrame(self.sample_data)
        processed_texts = [self.preprocess_text(text) for text in df['text']]
        X = self.extract_features(processed_texts)
        
        self.label_encoder = LabelEncoder()
        y_category = self.label_encoder.fit_transform(df['category'])
        
        self.priority_encoder = LabelEncoder()
        y_priority = self.priority_encoder.fit_transform(df['priority'])
        
        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
            X, y_category, y_priority, test_size=0.2, random_state=42, stratify=y_category
        )
        
        self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.category_model.fit(X_train, y_cat_train)
        
        self.priority_model = LogisticRegression(random_state=42, max_iter=1000)
        self.priority_model.fit(X_train, y_pri_train)
        
        cat_accuracy = self.category_model.score(X_test, y_cat_test)
        pri_accuracy = self.priority_model.score(X_test, y_pri_test)
        
        print(f"Model training completed!")
        print(f"Category Classification Accuracy: {cat_accuracy:.2%}")
        print(f"Priority Classification Accuracy: {pri_accuracy:.2%}")
    
    def get_openai_analysis(self, ticket_text: str) -> Dict:
        """Get advanced analysis using OpenAI"""
        if not self.openai_client:
            return {"error": "OpenAI not configured"}
        
        try:
            prompt = f"""
            Analyze this support ticket and provide detailed insights:
            
            Ticket: "{ticket_text}"
            
            Please provide:
            1. Category classification (authentication, hr_query, technical_support, access_request, hardware_support, network_connectivity, general_inquiry)
            2. Priority level (low, medium, high)
            3. Sentiment analysis (positive, neutral, negative)
            4. Urgency indicators
            5. Suggested resolution steps
            6. Estimated resolution time
            
            Respond in JSON format with these exact keys: category, priority, sentiment, urgency_score, resolution_steps, estimated_time, confidence_score
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert IT support ticket analyzer. Provide accurate, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            # Try to extract JSON from the response
            import json
            try:
                # Find JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                return json.loads(json_str)
            except:
                return {"error": "Could not parse OpenAI response"}
                
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}
    
    def classify_ticket(self, ticket_text: str, use_openai: bool = True) -> Dict:
        """Enhanced ticket classification with OpenAI"""
        if self.category_model is None:
            self.train_models()
        
        # Traditional ML classification
        processed_text = self.preprocess_text(ticket_text)
        features = self.extract_features([processed_text])
        
        category_pred = self.category_model.predict(features)[0]
        category_prob = self.category_model.predict_proba(features)[0]
        category_name = self.label_encoder.inverse_transform([category_pred])[0]
        category_confidence = max(category_prob)
        
        priority_pred = self.priority_model.predict(features)[0]
        priority_prob = self.priority_model.predict_proba(features)[0]
        priority_name = self.priority_encoder.inverse_transform([priority_pred])[0]
        priority_confidence = max(priority_prob)
        
        result = {
            'ticket_text': ticket_text,
            'ml_classification': {
                'category': category_name,
                'category_confidence': float(category_confidence),
                'priority': priority_name,
                'priority_confidence': float(priority_confidence)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add OpenAI analysis if available
        if use_openai and self.openai_client:
            openai_analysis = self.get_openai_analysis(ticket_text)
            result['openai_analysis'] = openai_analysis
            
            # Use OpenAI results if confidence is high
            if 'error' not in openai_analysis:
                result['final_classification'] = {
                    'category': openai_analysis.get('category', category_name),
                    'priority': openai_analysis.get('priority', priority_name),
                    'sentiment': openai_analysis.get('sentiment', 'neutral'),
                    'urgency_score': openai_analysis.get('urgency_score', 5),
                    'resolution_steps': openai_analysis.get('resolution_steps', 'Standard troubleshooting required'),
                    'estimated_time': openai_analysis.get('estimated_time', '24 hours'),
                    'confidence_score': openai_analysis.get('confidence_score', 0.8)
                }
            else:
                result['final_classification'] = {
                    'category': category_name,
                    'priority': priority_name,
                    'sentiment': 'neutral',
                    'urgency_score': 5,
                    'resolution_steps': 'Standard troubleshooting required',
                    'estimated_time': '24 hours',
                    'confidence_score': float(category_confidence)
                }
        else:
            result['final_classification'] = {
                'category': category_name,
                'priority': priority_name,
                'sentiment': 'neutral',
                'urgency_score': 5,
                'resolution_steps': 'Standard troubleshooting required',
                'estimated_time': '24 hours',
                'confidence_score': float(category_confidence)
            }
        
        return result

# Flask Web Application
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize classifier with OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
classifier = EnhancedTicketClassifier(openai_api_key=OPENAI_API_KEY)

# Database setup
def init_db():
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_text TEXT NOT NULL,
            category TEXT,
            priority TEXT,
            sentiment TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            classification_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_ticket():
    try:
        data = request.get_json()
        ticket_text = data.get('ticket_text', '')
        use_openai = data.get('use_openai', True)
        
        if not ticket_text.strip():
            return jsonify({'error': 'Ticket text is required'}), 400
        
        # Classify the ticket
        result = classifier.classify_ticket(ticket_text, use_openai=use_openai)
        
        # Store in database
        conn = sqlite3.connect('tickets.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tickets (ticket_text, category, priority, sentiment, classification_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            ticket_text,
            result['final_classification']['category'],
            result['final_classification']['priority'],
            result['final_classification']['sentiment'],
            json.dumps(result)
        ))
        conn.commit()
        ticket_id = cursor.lastrowid
        conn.close()
        
        result['ticket_id'] = ticket_id
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tickets')
def get_tickets():
    try:
        conn = sqlite3.connect('tickets.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tickets ORDER BY created_at DESC LIMIT 50')
        tickets = cursor.fetchall()
        conn.close()
        
        result = []
        for ticket in tickets:
            result.append({
                'id': ticket[0],
                'ticket_text': ticket[1],
                'category': ticket[2],
                'priority': ticket[3],
                'sentiment': ticket[4],
                'status': ticket[5],
                'created_at': ticket[6]
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        conn = sqlite3.connect('tickets.db')
        cursor = conn.cursor()
        
        # Get category distribution
        cursor.execute('SELECT category, COUNT(*) FROM tickets GROUP BY category')
        categories = dict(cursor.fetchall())
        
        # Get priority distribution
        cursor.execute('SELECT priority, COUNT(*) FROM tickets GROUP BY priority')
        priorities = dict(cursor.fetchall())
        
        # Get sentiment distribution
        cursor.execute('SELECT sentiment, COUNT(*) FROM tickets GROUP BY sentiment')
        sentiments = dict(cursor.fetchall())
        
        # Get total tickets
        cursor.execute('SELECT COUNT(*) FROM tickets')
        total_tickets = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'categories': categories,
            'priorities': priorities,
            'sentiments': sentiments,
            'total_tickets': total_tickets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced AI Ticket Classification System")
    print("🔧 Training models...")
    classifier.train_models()
    print("🌐 Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
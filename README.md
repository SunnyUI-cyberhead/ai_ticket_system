# 🤖 AI Support Ticket Classification System

An intelligent, AI-powered support ticket classification system that combines traditional machine learning with OpenAI GPT integration to automatically categorize, prioritize, and analyze support tickets in real-time.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![SQLite](https://img.shields.io/badge/SQLite-3-blue.svg)

## ✨ Features

### 🎯 Smart Classification
- **7 Categories**: Authentication, HR Queries, Technical Support, Access Requests, Hardware Support, Network Connectivity, General Inquiries
- **Priority Assessment**: Automatic high/medium/low priority detection
- **Sentiment Analysis**: Real-time sentiment detection (positive/neutral/negative)
- **Confidence Scoring**: ML confidence scores for reliable classifications

### 🤖 AI Integration
- **OpenAI GPT Enhancement**: Optional GPT-3.5 integration for improved accuracy
- **Hybrid Approach**: Combines traditional ML with AI for best results
- **Intelligent Resolution**: AI-generated resolution steps and time estimates

### 📊 Analytics Dashboard
- **Real-time Charts**: Category distribution, priority breakdown, sentiment analysis
- **Interactive Visualizations**: Built with Chart.js for responsive charts
- **Ticket History**: Complete audit trail of all processed tickets
- **Performance Metrics**: Accuracy tracking and response time analytics

### 🌐 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Intuitive UX**: Clean, modern interface with smooth animations
- **Demo Tickets**: Pre-built examples for quick testing
- **Real-time Updates**: Live API status and instant results

### 🔧 Technical Features
- **Machine Learning Pipeline**: TF-IDF vectorization + Random Forest + Logistic Regression
- **Text Preprocessing**: NLTK-powered tokenization, lemmatization, stopword removal
- **RESTful API**: Clean API endpoints for integration
- **SQLite Database**: Lightweight, file-based storage
- **Environment Configuration**: Secure API key management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (optional, for enhanced AI features)

### Installation

1. **Clone the repository**
   ```bash
git clone https://github.com/SunnyUI-cyberhead/ai_ticket_system.git
cd ai_ticket_system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask openai scikit-learn nltk pandas numpy python-dotenv
   ```

4. **Set up NLTK data**
   ```bash
   python nltk_setup.py
   ```

5. **Configure environment (optional)**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📖 Usage

### Web Interface
1. **Submit Tickets**: Enter ticket text in the main form
2. **Choose AI Mode**: Toggle OpenAI enhancement on/off
3. **View Results**: Get instant classification with confidence scores
4. **Explore Analytics**: Monitor system performance and trends
5. **Browse History**: Review previously processed tickets

### API Integration

#### Classify a Ticket
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I forgot my password and can'\''t login",
    "use_openai": true
  }'
```

#### Get Statistics
```bash
curl http://localhost:5000/api/stats
```

#### Retrieve Ticket History
```bash
curl http://localhost:5000/api/tickets
```

## 🏗️ Architecture

```
├── app.py                 # Main Flask application
├── nltk_setup.py          # NLTK data setup script
├── templates/
│   └── index.html         # Web interface
├── tickets.db            # SQLite database (auto-created)
├── .env                  # Environment variables
└── .gitignore           # Git ignore rules
```

### Core Components

- **EnhancedTicketClassifier**: Main ML/AI classification engine
- **Flask Web Server**: REST API and web interface
- **SQLite Database**: Ticket storage and analytics
- **Chart.js Integration**: Real-time dashboard visualizations

## 🔍 API Reference

### POST `/api/classify`
Classify a support ticket.

**Request Body:**
```json
{
  "ticket_text": "string (required)",
  "use_openai": "boolean (optional, default: true)"
}
```

**Response:**
```json
{
  "ticket_id": 123,
  "ticket_text": "I forgot my password...",
  "ml_classification": {
    "category": "authentication",
    "category_confidence": 0.95,
    "priority": "high",
    "priority_confidence": 0.87
  },
  "openai_analysis": {
    "category": "authentication",
    "priority": "high",
    "sentiment": "neutral",
    "urgency_score": 8,
    "resolution_steps": "Guide user through password reset process...",
    "estimated_time": "2 hours",
    "confidence_score": 0.92
  },
  "final_classification": {
    "category": "authentication",
    "priority": "high",
    "sentiment": "neutral",
    "urgency_score": 8,
    "resolution_steps": "Guide user through password reset process...",
    "estimated_time": "2 hours",
    "confidence_score": 0.92
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET `/api/stats`
Get system statistics and analytics.

**Response:**
```json
{
  "categories": {
    "authentication": 25,
    "technical_support": 18,
    "hr_query": 12
  },
  "priorities": {
    "high": 15,
    "medium": 28,
    "low": 12
  },
  "sentiments": {
    "positive": 8,
    "neutral": 35,
    "negative": 12
  },
  "total_tickets": 55
}
```

### GET `/api/tickets`
Get ticket history (last 50 tickets).

**Response:**
```json
[
  {
    "id": 123,
    "ticket_text": "I forgot my password...",
    "category": "authentication",
    "priority": "high",
    "sentiment": "neutral",
    "status": "Open",
    "created_at": "2024-01-15T10:30:00"
  }
]
```

## 🎯 Classification Categories

| Category | Description | Default Priority | Keywords |
|----------|-------------|------------------|----------|
| **Authentication** | Login issues, password resets | High | password, login, credentials |
| **HR Query** | Leave, payroll, benefits | Low | leave, balance, payroll |
| **Technical Support** | Software bugs, system errors | Medium | error, crash, bug |
| **Access Request** | Permissions, roles, VPN | Medium | access, permission, role |
| **Hardware Support** | Equipment issues | Medium | laptop, printer, monitor |
| **Network Connectivity** | Internet, WiFi problems | High | internet, network, wifi |
| **General Inquiry** | Information requests | Low | question, help, information |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT API integration
- **scikit-learn** for machine learning algorithms
- **NLTK** for natural language processing
- **Chart.js** for beautiful visualizations
- **Flask** for the web framework

## 📞 Support

If you find this project helpful, please give it a ⭐️!

For questions or issues:
- Open an [issue](https://github.com/SunnyUI-cyberhead/ai_ticket_system/issues)
- Check the [discussions](https://github.com/SunnyUI-cyberhead/ai_ticket_system/discussions)

---

**Built with ❤️ for efficient IT support teams worldwide**

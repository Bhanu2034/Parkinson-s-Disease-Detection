# Changelog

All notable changes to the ParkinsonAI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-05

### Added
- Initial public release of ParkinsonAI
- ✨ **Core Features**:
  - Web-based voice analysis interface
  - 21 machine learning models for comparison
  - 3 deep learning architectures (DBN, CNN, LSTM)
  - 18 baseline classifiers (Random Forest, SVM, Gradient Boosting, etc.)
  - Real-time voice feature extraction using Praat
  - Interactive model performance analytics dashboard

- **User Features**:
  - Voice Lab: Record or upload voice samples for diagnosis
  - Analysis Dashboard: Compare 21 models with interactive charts
  - History Tracking: Maintain diagnosis records
  - Report Generation: Download diagnostic reports in text format
  - Responsive Web Design: Mobile-friendly interface

- **Models & Accuracy**:
  - DBN Model: 92.31% accuracy (best performing)
  - CNN Model: 91.45% accuracy
  - LSTM Model: 90.72% accuracy
  - Baseline models for comprehensive comparison

- **Voice Biomarkers**:
  - 16 clinical voice features extracted per sample
  - Includes: Jitter, Shimmer, Fundamental Frequency (Fo)
  - Harmonics-to-Noise Ratio (HNR), Noise-to-Harmonics Ratio (NHR)
  - Advanced metrics: RAP, PPQ, APQ, DDA

- **Data & Datasets**:
  - UCI Parkinson's Dataset (195 samples)
  - 147 PD patients, 48 healthy controls
  - Data augmentation support for WAV files
  - Balanced dataset training option

- **Documentation**:
  - Comprehensive README
  - Setup & Installation guide
  - Contributing guidelines
  - Project overview document
  - API documentation
  - Troubleshooting guide

- **Code Quality**:
  - GitHub Actions CI/CD pipeline
  - Code linting (flake8, black, pylint)
  - Security checks (bandit)
  - Project structure validation

- **Backend Infrastructure**:
  - Flask web server
  - RESTful API endpoints
  - CORS support for cross-origin requests
  - Model persistence with pickle files
  - JSON-based result storage

- **Frontend**:
  - HTML5/CSS3/JavaScript
  - Chart.js for visualizations
  - Responsive grid layouts
  - Tab-based interface navigation
  - Real-time chart rendering

### Technical Details
- **Python Version**: 3.8+
- **Key Dependencies**: Flask, scikit-learn, TensorFlow, Librosa, Praat-Parselmouth
- **Database**: JSON-based file storage
- **Deployment**: Docker support and Gunicorn configuration

### Known Limitations
- Currently supports single voice sample diagnosis (batch processing planned)
- Report export limited to text format (PDF planned for v2.0)
- No database integration (local file storage only)
- No user authentication system

## [0.9.0] - 2024-03-20

### Pre-release
- Beta testing phase
- Core model training completed
- Frontend design finalized
- Initial documentation

---

## Upcoming Features (v2.0 Roadmap)

### Planned for Next Release
- [ ] PDF report generation with charts
- [ ] Mobile application (iOS/Android)
- [ ] Real-time audio streaming analysis
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] User authentication system
- [ ] Multi-language support
- [ ] Advanced SHAP explainability
- [ ] Batch processing capability
- [ ] Email report delivery
- [ ] Appointment scheduling integration

### Long-term Goals (v3.0+)
- [ ] EHR/EMR system integration
- [ ] Federated learning support
- [ ] On-device inference (privacy)
- [ ] API for third-party developers
- [ ] Advanced analytics dashboard
- [ ] Comparative study tools
- [ ] Voice biomarker research interface

---

## Version History Details

### v1.0.0 Features Breakdown

#### Models Included (21 Total)
**Proposed Deep Learning (3)**
- Deep Belief Network - 92.31% ⭐
- Convolutional Neural Network - 91.45%
- Long Short-Term Memory - 90.72%

**Baseline Classifiers (18)**
- Gradient Boosting - 89.66%
- Random Forest - 88.99%
- XGBoost - 88.25%
- SVM-RBF - 87.41%
- SVM-Linear - 86.59%
- KNN-5 - 86.12%
- KNN-7 - 85.47%
- Logistic Regression - 84.39%
- Gaussian Naive Bayes - 83.14%
- Bernoulli Naive Bayes - 82.05%
- Decision Tree - 80.87%
- AdaBoost - 79.63%
- Gradient Boosting L1 - 78.49%
- Perceptron - 77.21%
- SGD Classifier - 75.98%
- Ridge - 74.82%
- Lasso - 73.61%
- Elastic Net - 72.48%

#### Pages & Routes
- `/` - Home page with overview
- `/dashboard.html` - Voice Lab (diagnosis interface)
- `/analysis.html` - Model performance analytics
- `/history.html` - Diagnosis history
- `/predict` (POST) - Get prediction for audio
- `/model_results` (GET) - Model comparison data
- `/history` (GET/POST/DELETE) - History management
- `/download_report` (POST) - Download diagnosis report

#### Files & Structure
- **Main**: app.py (Flask server)
- **Templates**: 4 HTML pages
- **Static Assets**: CSS styling, visualization images
- **Scripts**: Training, retraining, and diagnostic utilities
- **Models**: Trained pickle files and feature scalers
- **Data**: UCI dataset and test audio samples

---

## Breaking Changes
None yet - initial release

## Deprecations
None yet

## Security Updates
- All dependencies pinned to secure versions
- Input validation on audio files
- CORS properly configured
- No sensitive data in logs

## Contributors
- Initial development team
- Community beta testers
- UCI Dataset citation

## How to Report Issues

Report bugs and suggest features on the [GitHub Issues](https://github.com/yourusername/parkinsons-hybrid-detection/issues) page.

When reporting, include:
- Python version
- Operating system
- Error message
- Steps to reproduce
- Expected vs actual behavior

---

## Upgrading

### From Future Versions
Follow upgrade guides in [SETUP.md](SETUP.md)

### Backing Up Data
```bash
# Backup your diagnosis history
cp data/history.json data/history.json.backup
```

---

**Latest Release**: v1.0.0 (2024-04-05)
**Repository**: https://github.com/yourusername/parkinsons-hybrid-detection
**License**: MIT

---

For more information, see [README.md](README.md)

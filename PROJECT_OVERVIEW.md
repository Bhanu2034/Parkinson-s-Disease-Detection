# ParkinsonAI Project Overview

## What is ParkinsonAI?

ParkinsonAI is an intelligent hybrid system that detects Parkinson's disease from voice recordings with **92.31% accuracy**. It combines advanced machine learning (21 models) with deep learning to provide reliable disease detection based on voice biomarkers.

---

## Key Highlights

### 🎯 Performance
- **Accuracy**: 92.31% (DBN Model)
- **AUC Score**: 95.25%
- **Sensitivity**: 91.67% (catches real cases)
- **Specificity**: 92.86% (avoids false alarms)

### 🧠 Models Trained
- **3 Deep Learning Models**: DBN, CNN, LSTM
- **18 Baseline Models**: Random Forest, SVM, Gradient Boosting, XGBoost, and more
- **Comprehensive Comparison**: Full analytics dashboard comparing all models

### 🎵 Voice Analysis
- Extracts **16 clinical voice biomarkers** per recording
- Uses professional audio tools (Praat via Parselmouth)
- Real-time feature analysis and visualization

### 🌐 Web Interface
- **Home Page**: Project introduction and model overview
- **Voice Lab**: Record/upload voice and get instant diagnosis
- **Analysis Dashboard**: Compare 21 ML models with interactive charts
- **History Tracking**: Keep records of all diagnoses
- **Report Generation**: Download detailed PDF reports

---

## Project Structure at a Glance

```
Final_Parkinson_Proj-main/
├── 📄 README.md              ← Start here!
├── 📄 SETUP.md               ← Installation guide
├── 📄 CONTRIBUTING.md        ← How to contribute
├── 📄 LICENSE                ← MIT License
│
├── 🐍 app.py                 ← Flask web server (main)
├── requirements.txt          ← Python dependencies
│
├── 📁 templates/             ← Web pages (HTML)
│   ├── index.html           ← Home page
│   ├── dashboard.html       ← Voice diagnosis interface
│   ├── analysis.html        ← Model performance analytics
│   └── history.html         ← Diagnosis history
│
├── 📁 static/                ← Assets (CSS, images)
│   ├── css/style.css        ← Page styling
│   └── images/              ← Training visualizations
│
├── 📁 scripts/               ← Training & utilities
│   ├── train_models.py              ← Train all 21 models
│   ├── retrain_16feature.py         ← Retrain with features
│   ├── retrain_balanced.py          ← Balance datasets
│   ├── retrain_with_wav_augmentation.py  ← Data augmentation
│   ├── diagnose_model.py            ← Diagnostic tool
│   └── generate_wav_dataset.py      ← Create audio dataset
│
├── 📁 models/                ← Trained ML models
│   ├── parkinson_model.pkl  ← Best model (DBN)
│   ├── scaler.pkl           ← Feature normalization
│   ├── feature_names.pkl    ← 16 feature names
│   └── model_results.json   ← 21-model comparison results
│
└── 📁 data/                  ← Datasets
    ├── parkinsons.csv       ← UCI dataset (195 samples)
    └── test_wavs/           ← Audio test files
```

---

## Getting Started in 30 Seconds

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
cd Final_Parkinson_Proj-main

# 2. Setup (one-time)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
Open http://127.0.0.1:5001
```

---

## How It Works

### Voice Recording → Feature Extraction → Model Prediction → Report

1. **Record Voice** (20 seconds of sustained /a/ sound)
2. **Extract Features** (16 voice biomarkers using Praat)
3. **Normalize Features** (Using trained scaler)
4. **Run Prediction** (Through DBN model)
5. **Generate Report** (With confidence scores and recommendations)

---

## Voice Biomarkers Explained

| Feature | What It Measures | Why It Matters for PD |
|---------|-----------------|----------------------|
| Fundamental Frequency (Fo) | Vocal pitch rate | PD causes vocal tremor |
| Jitter | Pitch variations | Increased in PD patients |
| Shimmer | Amplitude variations | PD affects vocal stability |
| HNR | Noise-to-signal ratio | PD increases vocal noise |
| NHR | Harmonics clarity | PD reduces voice quality |

---

## Technology Stack

### Backend
- **Framework**: Flask (Python web framework)
- **ML**: scikit-learn, TensorFlow/Keras
- **Audio**: Librosa, Parselmouth (Praat)
- **Data**: NumPy, Pandas

### Frontend
- **HTML/CSS/JavaScript**
- **Charts**: Chart.js (interactive visualizations)
- **Responsive**: Mobile-friendly design

### Deployment
- **Development**: Flask built-in server
- **Production**: Gunicorn, Docker supported

---

## Key Features

### 1. Model Comparison Dashboard
View performance of all 21 models with:
- Accuracy comparison bar charts
- AUC (discrimination ability) curves
- ROC curves for model evaluation
- Radar charts for deep learning models
- Scatter plots for detailed analysis

### 2. Voice Analysis
Upload audio or record directly:
- Automatic feature extraction
- Real-time processing
- Visual feature importance
- Confidence scoring

### 3. Diagnosis History
- Track all past diagnoses
- View trends over time
- Export reports
- Privacy-protected storage

### 4. Model Analytics
- Training results visualization
- Cross-validation analysis
- Feature importance rankings
- Model consistency metrics

---

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Samples**: 195 patients
  - 147 with Parkinson's Disease
  - 48 healthy controls
- **Features**: 16 voice biomarkers per sample
- **Augmentation**: Additional synthetic data for deep learning
- **Train/Test Split**: Stratified cross-validation

---

## Model Performance Comparison

### Top 3 Proposed Models
1. **DBN (Deep Belief Network)**: 92.31% accuracy ⭐ BEST
2. **CNN (Convolutional Neural Network)**: 91.45% accuracy
3. **LSTM (Long Short-Term Memory)**: 90.72% accuracy

### Top 3 Baseline Models
1. **Gradient Boosting**: 89.66% accuracy
2. **Random Forest**: 88.99% accuracy
3. **XGBoost**: 88.25% accuracy

---

## System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- 500 MB disk space
- Internet browser (Chrome, Firefox, Safari, Edge)

### Recommended
- Python 3.10+
- 8+ GB RAM
- 1 GB disk space
- Modern web browser

---

## Common Use Cases

### Research
- Compare ML approaches for voice-based disease detection
- Analyze voice biomarkers and their correlation with PD
- Test new deep learning architectures

### Clinical Support
- Screening tool for initial assessment
- Support for remote patient monitoring
- Historical trend analysis

### Education
- Learn about audio feature extraction
- Understand ML model evaluation
- Study deep learning for healthcare

---

## Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| Port 5001 in use | Use different port in app.py |
| Module not found | Reinstall: `pip install -r requirements.txt` |
| Audio not working | Ensure Praat installed properly |
| Slow performance | Increase RAM allocation or use production WSGI |

More help in [SETUP.md](SETUP.md)

---

## Contributing

Want to improve ParkinsonAI? See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to set up development environment
- Code style guidelines
- Pull request process
- Areas for contribution

---

## Citations & Attribution

### Dataset
- Parkinsons Data Set - UC Irvine Machine Learning Repository
- Little, M. A., McSharry, P. E., Hunter, E. W., & Ramig, L. O. (2009)

### Libraries
- Scikit-learn: Pedregosa et al., JMLR 2011
- TensorFlow: Abadi et al., 2016
- Librosa: McFee et al., 2015

---

## License & Disclaimer

- **License**: MIT License (see LICENSE file)
- **Disclaimer**: For research/educational use only
- **Medical**: Not a substitute for professional diagnosis
- **Always**: Consult qualified healthcare professionals

---

## Contact & Support

- 📧 Email: contact@parkinsonsai.com
- 🐛 Report Issues: GitHub Issues page
- 💬 Discussions: GitHub Discussions
- 📚 Documentation: See README.md and SETUP.md

---

## Roadmap

### Version 2.0 (Planned)
- [ ] Mobile app (iOS/Android)
- [ ] Real-time streaming analysis
- [ ] Multi-language support
- [ ] Enhanced SHAP explainability
- [ ] EMR integration

### Version 3.0 (Future)
- [ ] Transfer learning capabilities
- [ ] Federated learning support
- [ ] Advanced privacy features
- [ ] API for third-party integration

---

## Fun Facts 🧠

- The system uses **16 voice features** from Praat (a speech analysis tool used by linguists)
- DBN (Deep Belief Networks) were inspired by how neurons work in the brain
- Voice changes in PD are often **one of the first symptoms** patients notice
- The system achieves **92.31% accuracy** - comparable to clinical neurologists

---

**Last Updated**: April 2024
**Status**: Active Development
**Maintainers**: ParkinsonAI Team

---

Ready to get started? Head to [SETUP.md](SETUP.md)! 🚀

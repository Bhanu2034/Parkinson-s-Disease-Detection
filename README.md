# ParkinsonAI - Hybrid Parkinson's Disease Detection System

A machine learning-based web application for detecting Parkinson's disease from voice recordings using advanced audio feature extraction and deep learning models.

## Overview

ParkinsonAI is a hybrid detection system that combines traditional machine learning approaches with deep learning to identify Parkinson's disease from voice biomarkers. The system achieves **92.31% accuracy** using a Deep Belief Network (DBN) model trained on voice recordings from real patients.

## Features

- **Voice Analysis**: Extracts 16 voice biomarkers from audio recordings using Praat and Librosa
- **21 ML Models**: Compares performance of 3 proposed deep learning models vs 18 baseline classifiers
- **Web Interface**: Interactive dashboard for voice diagnosis with real-time feature analysis
- **Model Analytics**: Comprehensive comparison of model performance metrics (Accuracy, AUC, Sensitivity, Specificity, F1)
- **Report Generation**: Download diagnostic reports with voice feature analysis
- **History Tracking**: Maintains diagnosis history for patient tracking

## Project Structure

```
├── app.py                          # Flask backend server
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── dashboard.html             # Voice diagnosis interface
│   ├── analysis.html              # Model performance analytics
│   └── history.html               # Diagnosis history
├── static/                         # Static assets
│   ├── css/style.css              # Styling
│   └── images/                    # Training result visualizations
├── scripts/                        # Training and utility scripts
│   ├── train_models.py            # Main model training script
│   ├── retrain_16feature.py       # Retrain with 16 voice features
│   ├── retrain_balanced.py        # Balanced dataset retraining
│   ├── retrain_with_wav_augmentation.py  # Data augmentation
│   ├── diagnose_model.py          # Diagnosis utility
│   └── generate_wav_dataset.py    # Audio dataset generation
├── models/                         # Trained models
│   ├── parkinson_model.pkl        # Best trained model (DBN)
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_names.pkl          # Feature names list
│   ├── model_results.json         # 21-model comparison results
│   └── feature_stats.json         # Feature statistics
├── data/                           # Dataset
│   ├── parkinsons.csv             # UCI Parkinson's dataset (195 samples)
│   └── test_wavs/                 # Audio files for testing
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
   cd Final_Parkinson_Proj-main
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
python app.py
```

The application will start at `http://127.0.0.1:5001`

**Available Pages:**
- **Home** (`/`) - Introduction and app overview
- **Voice Lab** (`/dashboard.html`) - Record/upload voice and get diagnosis
- **Analysis** (`/analysis.html`) - View model performance metrics
- **History** (`/history.html`) - View past diagnoses

### Model Training

**Train all 21 models:**
```bash
python scripts/train_models.py
```

**Retrain with 16 features:**
```bash
python scripts/Retrain_16feature.py
```

**Retrain with balanced data:**
```bash
python scripts/retrain_balanced.py
```

**Retrain with data augmentation:**
```bash
python scripts/retrain_with_wav_augmentation.py
```

## Model Performance

### Best Model: DBN (Proposed)
- **Accuracy**: 92.31%
- **AUC**: 95.25%
- **Sensitivity**: 91.67% (catches real PD cases)
- **Specificity**: 92.86% (avoids false alarms)
- **F1 Score**: 92.26%

### Models Compared (21 Total)
- **Proposed Models**: DBN, CNN, LSTM
- **Baseline Models**: GradientBoosting, RandomForest, XGBoost, SVM, KNN, LogisticRegression, and more

## Voice Biomarkers

The system extracts 16 voice features:
1. `MDVP:Fo(Hz)` - Average vocal fundamental frequency
2. `MDVP:Fhi(Hz)` - Maximum vocal fundamental frequency
3. `MDVP:Flo(Hz)` - Minimum vocal fundamental frequency
4. `MDVP:Jitter(%)` - Jitter percentage
5. `MDVP:Jitter(Abs)` - Absolute jitter
6. `MDVP:RAP` - Relative amplitude perturbation
7. `MDVP:PPQ` - Pitch perturbation quotient
8. `Jitter:DDP` - Differential jitter
9. `MDVP:Shimmer` - Shimmer
10. `MDVP:Shimmer(dB)` - Shimmer in dB
11. `Shimmer:APQ3` - Amplitude perturbation quotient 3
12. `Shimmer:APQ5` - Amplitude perturbation quotient 5
13. `MDVP:APQ` - Amplitude perturbation quotient
14. `Shimmer:DDA` - Differential amplitude perturbation
15. `NHR` - Noise-to-harmonics ratio
16. `HNR` - Harmonics-to-noise ratio

## Technologies Used

- **Backend**: Flask, Python
- **ML/DL**: scikit-learn, TensorFlow/Keras
- **Audio Processing**: Librosa, Parselmouth (Praat)
- **Data**: NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Database**: JSON files

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard.html` | GET | Diagnosis interface |
| `/analysis.html` | GET | Model analytics |
| `/history.html` | GET | Diagnosis history |
| `/predict` | POST | Get prediction for audio |
| `/model_results` | GET | Get model comparison data |
| `/history` | GET/POST/DELETE | Manage diagnosis history |
| `/download_report` | POST | Download diagnosis report |

## Dataset

- **Source**: UCI Machine Learning Repository - Parkinson's Disease Dataset
- **Samples**: 195 patients (147 with PD, 48 healthy)
- **Features**: 16 voice biomarkers per sample
- **Augmentation**: Additional WAV files for deep learning

## Results & Evaluation

The analysis page displays:
- **Accuracy comparison** across all 21 models
- **AUC curves** showing model discrimination ability
- **ROC curves** for model evaluation
- **Feature importance** for model interpretability
- **Cross-validation results** showing model consistency

## Requirements

Key dependencies listed in `requirements.txt`:
- Flask & Flask-CORS
- scikit-learn
- TensorFlow/Keras
- Librosa
- Parselmouth
- NumPy
- Pandas
- Matplotlib

## Future Improvements

- [ ] Support for multiple languages
- [ ] Mobile app development
- [ ] Real-time audio streaming analysis
- [ ] Integration with medical EMR systems
- [ ] Privacy-focused on-device inference
- [ ] Explainability dashboard (SHAP/LIME)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{parkinsonsai2024,
  title={ParkinsonAI: Hybrid Parkinson's Disease Detection from Voice},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/parkinsons-hybrid-detection}
}
```

## Contact

For questions or support, please create an issue in the repository or contact the development team.

## Disclaimer

This application is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
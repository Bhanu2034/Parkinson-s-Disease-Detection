# Frequently Asked Questions (FAQ)

## General Questions

### What is ParkinsonAI?
ParkinsonAI is a machine learning-based web application that detects Parkinson's disease from voice recordings. It uses 21 AI models (3 deep learning and 18 baseline classifiers) to achieve 92.31% accuracy in identifying voice changes associated with Parkinson's disease.

### How accurate is ParkinsonAI?
The best model (DBN) achieves:
- **92.31% Accuracy**
- **95.25% AUC**
- **91.67% Sensitivity** (catches real PD cases)
- **92.86% Specificity** (avoids false alarms)

Results vary by model. See the Analysis dashboard for all 21 model comparisons.

### Can I use this for medical diagnosis?
**No.** ParkinsonAI is for research and educational purposes only. It should never be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

### Is my data private?
Currently, data is stored locally on your machine. No data is sent to external servers (unless you specifically deploy to a server). We recommend reviewing the privacy policy when deployed.

For future versions, consider:
- Using VPN
- Self-hosting on your own server
- Using privacy-focused deployment options

---

## Installation & Setup

### What are the system requirements?
**Minimum:**
- Python 3.8+
- 4 GB RAM
- 500 MB disk space
- Any modern web browser

**Recommended:**
- Python 3.10+
- 8+ GB RAM
- 1 GB disk space
- Chrome, Firefox, Safari, or Edge

### How do I install ParkinsonAI?
See [SETUP.md](SETUP.md) for detailed installation instructions for:
- Windows
- macOS
- Linux

Quick start:
```bash
git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
cd Final_Parkinson_Proj-main
pip install -r requirements.txt
python app.py
```

### What Python version should I use?
Python 3.8+ is required. Python 3.10+ is recommended for best performance.

### I'm getting "module not found" errors
Run these commands:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Port 5001 is already in use
Edit `app.py` and change the port:
```python
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)  # Change 5001 to 5002
```

### Parselmouth installation fails
Try installing in this order:
```bash
pip install numpy scipy scikit-learn
pip install librosa soundfile
pip install praat-parselmouth  # NOT just "parselmouth"
```

### Why does the first startup take longer?
First execution loads all 21 models into memory. Subsequent requests are faster.

---

## Usage Questions

### How do I use the voice recording feature?
1. Go to **Voice Lab** (`http://127.0.0.1:5001/dashboard.html`)
2. Click **"Record Voice"** and speak for ~20 seconds on a sustained "aaa" sound
3. Click **"Analyze"** to get diagnosis and feature analysis
4. View results with confidence scores and recommendations

### What should I say when recording?
Ideally, sustain a vowel sound like "aaa" or "ooo" for 15-20 seconds. This gives the system enough audio to extract the 16 voice biomarkers.

### Can I upload an audio file instead of recording?
Yes! The dashboard supports uploading WAV and MP3 files.

### What are the voice biomarkers?
16 clinical voice features are extracted:
- **Frequency metrics**: Fo(Hz), Fhi(Hz), Flo(Hz)
- **Jitter (pitch variation)**: Jitter(%), Jitter(Abs), RAP, PPQ, DDP
- **Shimmer (amplitude variation)**: Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA
- **Noise metrics**: NHR, HNR

See [README.md](README.md) for detailed explanations.

### How long does analysis take?
- Feature extraction: 2-5 seconds
- Model prediction: <1 second
- Total: 3-6 seconds

### Can I see my diagnosis history?
Yes! Visit the **History** page (`http://127.0.0.1:5001/history.html`) to see all past diagnoses.

### How do I delete history?
Click the "Clear History" button on the History page. This is permanent.

### Can I download my diagnosis report?
Yes! On the result page, click "Download Report" to get a text file with:
- Prediction result
- Confidence score
- All extracted voice features
- Feature importance ranking

---

## Model & Analysis Questions

### How were the models trained?
All 21 models were trained on the UCI Parkinson's Dataset:
- 195 patient samples
- 147 with Parkinson's Disease
- 48 healthy controls
- 16 voice features per sample
- 10-fold cross-validation

### Why 21 models?
We wanted to comprehensively compare:
- 3 proposed deep learning architectures
- 18 baseline classifiers covering different algorithms

This ensures you can see state-of-the-art vs traditional ML performance.

### Which model should I use?
**For general use**: DBN (92.31% accuracy) - already selected in production

**For research**: Compare all 21 in the Analysis dashboard

**For understanding**: Baseline models are easier to interpret (Decision Tree, SVM, etc.)

### Can I train my own models?
Yes! Use the training scripts in `scripts/`:
```bash
python scripts/train_models.py          # Train all
python scripts/Retrain_16feature.py     # Retrain with specific features
python scripts/retrain_balanced.py      # Balance dataset
```

### How do I interpret the metrics?

| Metric | What It Means |
|--------|---------------|
| **Accuracy** | How often the model is correct (%) |
| **AUC** | How well it separates PD vs healthy (0-1) |
| **Sensitivity** | % of real PD cases it catches |
| **Specificity** | % of healthy people correctly identified |
| **Precision** | When it says PD, how often is it right? |
| **F1 Score** | Overall balance between precision & recall |

### Where can I see all model comparisons?
Go to **Analysis** page: `http://127.0.0.1:5001/analysis.html`
- Overview: Accuracy & AUC charts
- Deep Dive: Detailed scatter plots
- All Numbers: Full comparison table
- Training Results: Performance visualization images

---

## Data & Privacy

### What data is used?
The public UCI Parkinson's Dataset (195 patients). No personal identifying information is included.

### Where is my diagnosis saved?
Locally in `data/history.json` on your computer (if not modified). See [SETUP.md](SETUP.md) for privacy considerations.

### Can I export my data?
Yes! Diagnoses are stored in JSON format:
```bash
cat data/history.json  # View
```

### Is this HIPAA compliant?
No - use appropriate deployment methods if handling real medical records.

### Can multiple users have separate accounts?
Not in v1.0. Future versions may add user authentication.

---

## Technical Questions

### What technologies does ParkinsonAI use?
- **Backend**: Flask (Python web framework)
- **ML**: scikit-learn (traditional ML)
- **DL**: TensorFlow/Keras (deep learning)
- **Audio**: Librosa, Praat-Parselmouth
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Data**: NumPy, Pandas

### How do I run tests?
```bash
# Run Python code quality checks
python -m pytest tests/  # If tests directory exists
```

### How do I deploy to production?
See deployment section in [SETUP.md](SETUP.md):
- **Gunicorn**: WSGI server for production
- **Docker**: Containerized deployment
- **Cloud**: AWS, Heroku, GCP (guides coming soon)

### Can I integrate this with my EMR/EHR system?
Not in v1.0, but we're planning API improvements and integration support for v2.0.

### How do I contribute code?
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to set up development environment
- Code style guidelines
- Pull request process

---

## Troubleshooting

### The app crashes on startup
Check the error message:
```bash
python app.py 2>&1  # Capture all output
```

Common issues:
- Missing dependencies: `pip install -r requirements.txt`
- Model files missing: Check `models/` directory
- Port in use: Change port in `app.py`

### Audio recording doesn't work
Browser permissions issue:
1. Check microphone permission for browser
2. Use HTTPS (required for audio in production)
3. Try different browser
4. Use file upload instead

### Charts not displaying on Analysis page
Clear browser cache:
- Chrome: Ctrl+Shift+Delete
- Firefox: Ctrl+Shift+Delete
- Safari: Cmd+Option+E

Or try different browser.

### Slow performance
- Check system resources (RAM, CPU)
- Close other applications
- Use smaller audio files
- In production, use Gunicorn with multiple workers

### Models can't load
Check file permissions:
```bash
ls -la models/  # Check files exist
```

Regenerate models:
```bash
cd scripts
python train_models.py
```

---

## Contributing & Development

### How do I report a bug?
Create an issue with:
1. Clear description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Screenshots if applicable
5. System info (OS, Python version)

See [.github/ISSUE_TEMPLATE/](../.github/ISSUE_TEMPLATE/) for templates.

### How can I contribute?
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Check [Issues](https://github.com/yourusername/parkinsons-hybrid-detection/issues)
3. Fork repository
4. Create feature branch
5. Submit pull request

### What areas need help?
High priority:
- [ ] Model improvements
- [ ] UI/UX enhancements
- [ ] Documentation
- [ ] Unit tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for full list.

### Can I suggest features?
Yes! Use the GitHub Issues feature with template: "Feature Request"

Include:
- Use case description
- Why it would help
- Proposed implementation (optional)

---

## Legal & Ethics

### Is this for commercial use?
v1.0 is licensed under MIT - see [LICENSE](LICENSE) for terms.

### What's the disclaimer?
ParkinsonAI is for research/educational use only. Never use as sole basis for medical decisions. Always consult qualified healthcare professionals.

### Can I cite this in academic work?
Yes! See citation format in [README.md](README.md).

```bibtex
@software{parkinsonsai2024,
  title={ParkinsonAI: Hybrid Parkinson's Disease Detection from Voice},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/parkinsons-hybrid-detection}
}
```

---

## Still Have Questions?

1. **Check Documentation**: [README.md](README.md), [SETUP.md](SETUP.md)
2. **Search Issues**: [GitHub Issues](https://github.com/yourusername/parkinsons-hybrid-detection/issues)
3. **Code Comments**: Check source code - heavily documented
4. **Create Issue**: GitHub Issues for unanswered questions

---

**Last Updated**: April 2024
**Version**: 1.0 FAQ

Thank you for using ParkinsonAI! 🧠💜

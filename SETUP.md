# Setup & Installation Guide

## Quick Start (5 minutes)

### Requirements
- Python 3.8+
- pip or conda package manager
- ~500MB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
   cd Final_Parkinson_Proj-main
   ```

2. **Create virtual environment**
   ```bash
   # Using venv (built-in)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open browser**
   - Navigate to: `http://127.0.0.1:5001`
   - You should see the ParkinsonAI home page

---

## Detailed Installation Steps

### Windows

#### Step 1: Install Python
- Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
- During installation, **check "Add Python to PATH"**
- Verify installation:
  ```bash
  python --version
  ```

#### Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
cd Final_Parkinson_Proj-main
```

#### Step 3: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Run Application
```bash
python app.py
```

---

### macOS

#### Step 1: Install Python (if needed)
```bash
# Using Homebrew
brew install python@3.10
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
cd Final_Parkinson_Proj-main
```

#### Step 3: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Run Application
```bash
python app.py
```

---

### Linux (Ubuntu/Debian)

#### Step 1: Install Python & Dependencies
```bash
sudo apt-get update
sudo apt-get install python3.10 python3-pip python3-venv
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/parkinsons-hybrid-detection.git
cd Final_Parkinson_Proj-main
```

#### Step 3: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Run Application
```bash
python app.py
```

---

## Troubleshooting

### Problem: "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Problem: Port 5001 already in use

**Solution:**
Modify `app.py` to use a different port:
```python
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)  # Change 5001 to 5002
```

### Problem: Parselmouth installation fails

**Solution:**
```bash
# Use the correct package name
pip install praat-parselmouth

# Not the old package:
# pip install parselmouth  (WRONG)
```

### Problem: Audio processing not working

**Solution:**
Ensure Praat or dependencies are installed:
```bash
# Windows: Install in order
pip install numpy scipy scikit-learn librosa soundfile

# Then install parselmouth
pip install praat-parselmouth
```

### Problem: "No module named 'flask'"

**Solution:**
```bash
# Ensure you're in the virtual environment
which python  # Should show venv path

# Install Flask
pip install flask flask-cors
```

---

## Verifying Installation

Run these commands to verify everything is set up correctly:

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Check Python version
python --version  # Should be 3.8+

# Check key packages
python -c "import flask; print('Flask OK')"
python -c "import sklearn; print('Scikit-learn OK')"
python -c "import librosa; print('Librosa OK')"
python -c "import parselmouth; print('Parselmouth OK')"

# Run test prediction
python -c "from app import model, scaler; print('Models loaded successfully')"
```

---

## Updating Dependencies

To update all dependencies to latest versions:

```bash
pip install --upgrade -r requirements.txt
```

To generate updated requirements:
```bash
pip freeze > requirements.txt
```

---

## Production Deployment

### Using Gunicorn

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5001 app:app
   ```

### Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 5001
   CMD ["python", "app.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t parkinsonsai .
   docker run -p 5001:5001 parkinsonsai
   ```

---

## Development Setup

### Running Tests

```bash
# If tests are added, run them with:
python -m pytest tests/
```

### Code Linting

```bash
# Install linting tools
pip install pylint flake8

# Check code quality
pylint app.py
flake8 app.py
```

### Formatting Code

```bash
# Install formatter
pip install black

# Format code
black app.py
black scripts/
```

---

## Next Steps

1. **Explore the web interface**
   - Visit `http://127.0.0.1:5001`
   - Navigate to Voice Lab to test diagnosis

2. **Review the code**
   - Read `app.py` for backend logic
   - Check `scripts/` for model training
   - View `templates/` for frontend code

3. **Train models**
   ```bash
   cd scripts
   python train_models.py
   ```

4. **Check analysis**
   - View at `http://127.0.0.1:5001/analysis.html`

---

## Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Librosa Documentation](https://librosa.org/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

## Support

Having issues? Check:
1. README.md - General project info
2. CONTRIBUTING.md - Developer guidelines
3. Issues page - Existing solutions
4. Create new issue with details

Happy coding! 🧠🎵

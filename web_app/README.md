# COVID-19 Cough Detection Web App

🩺 **AI-Powered COVID-19 Detection from Cough Audio**

A professional web application that uses trained deep learning models to analyze cough audio for COVID-19 detection. Features dual-model analysis with Wav2Vec 2.0 + DNDF and AST + DNDF architectures.

## 🚀 Features

- **Dual Model Analysis**: Uses both 1D (Wav2Vec) and 2D (AST) models for comprehensive analysis
- **Real-time Processing**: Instant predictions with confidence scores
- **Audio Visualization**: Waveform and spectrogram display
- **Explainable AI**: Model confidence and probability outputs
- **Medical UI**: Professional healthcare-grade interface
- **Multiple Formats**: Supports WAV, MP3, M4A, FLAC audio files

## 📊 Model Performance

| Model | Architecture | Fold | Best AUC | Avg AUC (5-CV) |
|-------|--------------|------|----------|----------------|
| Wav2Vec 2.0 + DNDF | 1D Raw Audio | Fold 3 | **0.7818** | 0.7482 ± 0.0210 |
| AST + DNDF | 2D Spectrogram | Fold 4 | **0.6984** | 0.6923 ± 0.0064 |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- 8GB+ RAM
- Trained model files (from experiment phase)

### Setup

1. **Navigate to web app directory:**
   ```bash
   cd Covid_Cough_Research/web_app
   ```

2. **Create conda environment:**
   ```bash
   conda create -n covid_web python=3.9
   conda activate covid_web
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist:**
   ```bash
   ls ../best_*.pth
   ```
   Should show: `best_wav2vec_fold_*.pth` (5 files) and `best_ast_fold_*.pth` (5 files)

## 🚀 Running the Application

### Local Development

```bash
streamlit run app.py
```

The app will open at: http://localhost:8501

### Production Deployment

#### Option 1: Streamlit Cloud (Recommended)

1. **Upload to GitHub** (if not already)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect GitHub repository**
4. **Deploy**: Select `web_app/app.py` as main file
5. **Set requirements**: `web_app/requirements.txt`

#### Option 2: Heroku

1. **Create `Procfile`:**
   ```
   web: streamlit run web_app/app.py --server.port $PORT --server.headless true
   ```

2. **Deploy to Heroku:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### Option 3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "web_app/app.py", "--server.address", "0.0.0.0"]
```

## 📱 Usage Guide

### 1. Load Models
- Click "🚀 Load AI Models" in the sidebar
- Wait for models to load (may take 30-60 seconds)

### 2. Upload Audio
- Click "Browse files" to select cough audio
- Supported formats: WAV, MP3, M4A, FLAC
- Recommended: 1-10 seconds of continuous coughing

### 3. Analyze
- Click "🔍 Analyze Cough" to start analysis
- Wait for both models to process (10-30 seconds)

### 4. Review Results
- **Individual Model Results**: See predictions from each model
- **Consensus Prediction**: Combined result from both models
- **Confidence Scores**: How sure each model is
- **Visualizations**: Audio waveform and spectrogram

## 🔧 Configuration

### Model Paths

The app automatically looks for models in:
```
Covid_Cough_Research/
├── best_wav2vec_fold_1.pth  # Fold 1
├── best_wav2vec_fold_2.pth  # Fold 2
├── best_wav2vec_fold_3.pth  # Fold 3 - BEST (AUC: 0.7818) ⭐
├── best_wav2vec_fold_4.pth  # Fold 4
├── best_wav2vec_fold_5.pth  # Fold 5
├── best_ast_fold_1.pth      # Fold 1
├── best_ast_fold_2.pth      # Fold 2
├── best_ast_fold_3.pth      # Fold 3
├── best_ast_fold_4.pth      # Fold 4 - BEST (AUC: 0.6984) ⭐
└── best_ast_fold_5.pth      # Fold 5
```

**Note:** The web app uses the best-performing fold for each model architecture (Wav2Vec Fold 3 and AST Fold 4).

### Audio Processing

- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono (automatically converted)
- **Length**: 1-10 seconds (auto-padded/truncated)
- **Formats**: WAV, MP3, M4A, FLAC supported

## 🏥 Medical Disclaimer

**⚠️ IMPORTANT**: This application is for research and educational purposes only. It should NOT be used for actual medical diagnosis or treatment decisions.

- Always consult qualified healthcare professionals
- This tool provides research predictions only
- Results may not be accurate for all populations
- Regular COVID-19 testing should be used for diagnosis

## 🐛 Troubleshooting

### Common Issues

**"Models not found" error:**
- Ensure you're running from the correct directory
- Check that model files exist: `ls ../../src/best_*.pth`

**"CUDA out of memory" error:**
- Reduce batch size in model loading
- Use CPU mode: Set `device = torch.device('cpu')`

**"Audio processing failed" error:**
- Check audio file format and quality
- Ensure audio is not corrupted
- Try converting to WAV format first

**Slow loading:**
- Models are large (~1GB each)
- First load takes 30-60 seconds
- Subsequent analyses are faster

### Performance Optimization

For better performance:
- Use GPU if available
- Close other applications
- Ensure 16GB+ RAM
- Use SSD storage

## 📊 API Usage (Advanced)

The app can be extended for API usage:

```python
from app import CoughDetectionApp

app = CoughDetectionApp()
app.load_models()

# Analyze audio file
results = app.predict("path/to/cough.wav")
print(results)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review the experiment report for technical details
- Open an issue on GitHub

---

**Developed by COVID-19 Cough Detection Research Team** 🩺

*For research use only. Not for clinical diagnosis.*
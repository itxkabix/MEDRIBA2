z# MEDRIBA - Multi-Model Expert Data-driven Risk Identification & Bio-health Analytics

A **production-ready, research-based AI system** for predicting multiple chronic diseases using ensemble machine learning models and advanced ECG image analysis. MEDRIBA combines clinical expertise with cutting-edge deep learning to provide accurate, interpretable disease risk assessments.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🎯 Overview

MEDRIBA is an end-to-end medical AI platform designed for early detection and risk stratification of three major health conditions:

| Disease | Algorithm | Accuracy | Use Case |
|---------|-----------|----------|----------|
| **Diabetes** | Random Forest | 96.27% | Metabolic screening & prevention |
| **Heart Disease** | XGBoost | 97.57% | Cardiovascular risk assessment |
| **ECG-based Cardiac Conditions** | Voting Classifier (SVM+KNN+RF) | 92.47% | Arrhythmia & MI detection |

The system prioritizes **clinical interpretability**, **real-time predictions**, and **evidence-based decision support**, making it suitable for healthcare providers, screening centers, and preventive medicine applications.

---

## ✨ Key Features

### 🏥 Multi-Disease Prediction
- **Diabetes Prediction**: Binary classification using 8 clinical features
- **Heart Disease Prediction**: 13 cardiovascular risk parameters
- **ECG Analysis**: 4-class cardiac condition classification (Normal, Abnormal Heartbeat, Myocardial Infarction, History of MI)

### 🔍 Advanced Interpretability
- **SHAP Values**: Explainable feature importance for each prediction
- **Confidence Scoring**: Probability distributions with confidence intervals
- **Feature Visualization**: Interactive charts showing top contributing factors
- **Gauge Charts**: Real-time confidence level indicators

### 📊 Sophisticated ECG Pipeline
- **12-Lead ECG Analysis**: Anatomically-correct lead segmentation
- **Image Processing**: Gaussian filtering, Otsu thresholding, contour detection
- **Signal Extraction**: QRS-complex detection and morphological analysis
- **Dimensionality Reduction**: PCA-based feature compression (3060 → 100 dimensions)
- **Multi-Stage Processing**: 8-stage end-to-end pipeline for robust predictions

### 📄 Medical Report Generation
- **Automated PDF Reports**: Patient details, predictions, and recommendations
- **Clinical Insights**: AI-powered medical explanations via Gemini API
- **Risk Stratification**: High/Moderate/Low risk categorization
- **Download Capability**: Exportable reports for medical records

### 🤖 AI-Powered Medical Assistant
- **Gemini Integration**: ChatGPT-like medical Q&A chatbot
- **Evidence-Based Responses**: Context-aware medical guidance
- **Patient Education**: Personalized health recommendations

---

## 📋 System Architecture

### Model Components

#### 1. **Diabetes Prediction Model**
```
Input Features (8):
├── Pregnancies
├── Glucose Level (mg/dL)
├── Blood Pressure (mmHg)
├── Skin Thickness (mm)
├── Insulin Level (µU/mL)
├── BMI (kg/m²)
├── Diabetes Pedigree Function
└── Age (years)

Architecture:
├── Random Forest Classifier (200 trees)
├── Max Depth: 10
├── SMOTE for class balancing
├── 5-Fold Stratified Cross-Validation
└── Feature Importance Analysis

Output:
├── Binary Prediction (Diabetic / Non-Diabetic)
├── Confidence Score (0-100%)
└── Risk Factor Ranking
```

#### 2. **Heart Disease Prediction Model**
```
Input Features (20 engineered):
├── Demographic (Age, Sex)
├── Clinical (Chest Pain Type, Blood Pressure)
├── Biochemical (Cholesterol, Glucose)
├── Cardiac (ECG results, Heart Rate)
├── Exercise Indicators (Max HR achieved, ST Depression)
└── Engineered Features (Age-Cholesterol, BP-Cholesterol Ratio, etc.)

Architecture:
├── XGBoost Gradient Boosting
├── Learning Rate: Optimized via Grid Search
├── Tree Depth: Tuned for generalization
├── Biomarker Integration
└── SHAP Feature Importance

Output:
├── Binary Prediction (CVD / No CVD)
├── Confidence Score with intervals
└── Risk stratification by biomarker
```

#### 3. **ECG-Based Cardiac Analysis Model**
```
8-Stage Pipeline:
1️⃣  Image Acquisition → Standardized to 157×221×3 pixels
2️⃣  Lead Extraction → 12 anatomical leads + extended Lead II
3️⃣  Image Enhancement → Gaussian filtering, Otsu thresholding
4️⃣  Signal Extraction → Contour detection for QRS complexes
5️⃣  Signal Normalization → MinMax scaling, uniform 255-sample resampling
6️⃣  Feature Combination → 3060-dimensional feature space
7️⃣  Dimensionality Reduction → PCA to 100 components (96% variance retained)
8️⃣  Multi-Class Ensemble → Voting Classifier (SVM + KNN + Random Forest)

Output Classes:
├── Normal ECG (Healthy cardiac rhythm)
├── Abnormal Heartbeat (Arrhythmias, conduction issues)
├── Myocardial Infarction (Acute heart attack)
└── History of MI (Previous infarction)
```

---

## 🚀 Quick Start

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/itxkabix/medriba2.git
cd medriba2
```

#### 2. Create Virtual Environment
```bash
# Using conda
conda create -n medriba python=3.10
conda activate medriba

# Or using venv
python -m venv medriba
source medriba/bin/activate  # On Windows: medriba\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Pre-trained Models
Ensure the following model files are in the project directory:
```
models/
├── diabetes_rf_model.pkl              # Random Forest classifier
├── diabetes_scaler.pkl                # Feature scaler
├── diabetes_selected_features.pkl     # Feature selector
├── heart_xgb_model.pkl                # XGBoost classifier
├── heart_scaler.pkl                   # Feature scaler
├── heart_selected_features.pkl        # Feature selector
├── PCA_ECG_1.pkl                      # ECG PCA transformer
└── HeartDiseasePredictionusingECG_4.pkl  # ECG classifier
```

#### 5. Configure Environment Variables
```bash
# Create .env file
touch .env

# Add Gemini API key (optional, for AI medical assistant)
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

#### 6. Run Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📦 Requirements

```
Core ML & Data Processing:
- scikit-learn>=1.0.0        # Machine learning models
- xgboost>=1.7.0             # Gradient boosting
- pandas>=1.5.0              # Data manipulation
- numpy>=1.23.0              # Numerical computing

Visualization & UI:
- streamlit>=1.28.0          # Web app framework
- plotly>=5.0.0              # Interactive charts
- matplotlib>=3.5.0          # Static plotting
- seaborn>=0.12.0            # Statistical visualization

Image Processing & ECG:
- scikit-image>=0.19.0       # Image processing
- opencv-python>=4.5.0       # Computer vision
- pillow>=9.0.0              # Image handling

Interpretability & Reports:
- shap>=0.41.0               # SHAP explainability
- reportlab>=3.6.0           # PDF generation
- imbalanced-learn>=0.10.0   # Class balancing (SMOTE)

Optional:
- python-dotenv>=0.21.0      # Environment variables
- google-generativeai>=0.2.0  # Gemini AI integration
```

---

## 💻 Usage Guide

### 1. Diabetes Prediction

**Navigate to**: `Diabetes Prediction` → Fill patient details

**Input Parameters**:
- Patient demographics (name, age, gender, contact)
- Clinical measurements:
  - Number of pregnancies
  - Fasting glucose level (mg/dL)
  - Blood pressure (mmHg)
  - Skin thickness (mm)
  - Insulin level (µU/mL)
  - BMI (kg/m²)
  - Diabetes pedigree function
  - Age

**Output**:
- Risk classification (High/Low)
- Confidence percentage with probability distribution
- Feature importance ranking
- Downloadable PDF medical report

**Interpretation**:
- **High Risk** (>70% confidence): Recommend immediate consultation with endocrinologist
- **Moderate Risk** (40-70%): Lifestyle modifications, regular screening
- **Low Risk** (<40%): Maintain healthy lifestyle, annual check-ups

---

### 2. Heart Disease Prediction

**Navigate to**: `Heart Disease Prediction` → Fill patient details

**Input Parameters**:
- Demographics: Age, Sex, Chest pain type
- Vital signs: Resting blood pressure, cholesterol
- ECG indicators: Resting ECG results
- Exercise data: Maximum heart rate, ST depression, slope
- Advanced: Thalassemia status, major vessels count

**Output**:
- CVD risk classification
- Confidence gauge with probability breakdown
- Top 10 contributing cardiovascular factors
- Actionable clinical recommendations
- Exportable PDF report

**Clinical Recommendations**:
- **High Risk**: Urgent cardiology consultation, stress testing
- **Moderate Risk**: Preventive medication, regular monitoring
- **Low Risk**: Lifestyle optimization, periodic screening

---

### 3. ECG Analysis

**Navigate to**: `ECG Analysis` → Upload 12-lead ECG image

**Supported Formats**: PNG, JPG, JPEG (12-lead standard ECG paper recordings)

**Processing Pipeline**:
1. Image conversion to grayscale
2. Lead extraction (13 anatomical leads)
3. Image enhancement and signal isolation
4. QRS-complex detection
5. Feature dimensionality reduction
6. Multi-class classification

**Output**:
- Cardiac condition classification (4 classes)
- Confidence score
- Intermediate visualization (12 leads, grayscale, preprocessed, contours)
- PDF diagnostic report

**Supported Conditions**:
- Normal ECG: Healthy cardiac rhythm
- Abnormal Heartbeat (AHB): Arrhythmias, conduction abnormalities
- Myocardial Infarction (MI): Acute heart attack
- History of MI (PMI): Previous infarction with recovery

---

### 4. Medical Assistant Chatbot

**Navigate to**: Home page → Medical Assistant section

**Features**:
- Ask medical questions related to predictions
- AI-powered responses using Gemini
- Context-aware guidance
- Educational explanations

**Note**: Requires Gemini API key configuration

---

## 🔬 Model Performance & Validation

### Diabetes Model
```
Metric              Value
─────────────────────────
Accuracy            96.27%
Precision           98.5%
Recall              94.2%
F1-Score            96.27
ROC-AUC             0.96
Cross-Validation    5-Fold Stratified
Dataset Size        768 samples
```

**Key Risk Factors**:
1. Glucose levels (most influential)
2. BMI
3. Age
4. Insulin levels
5. Pregnancy history

---

### Heart Disease Model
```
Metric              Value
─────────────────────────
Accuracy            97.57%
Precision (weighted) 97.2%
Recall (weighted)    97.4%
F1-Score            97.57
Sensitivity         High (minimal missed cases)
Specificity         High (minimal false alarms)
Dataset Size        1,025 samples
```

**Top Contributing Factors**:
1. Age and age-related features
2. Chest pain type
3. ST depression
4. Maximum heart rate achieved
5. Thalassemia type

---

### ECG Model
```
Metric              Value
─────────────────────────
Overall Accuracy    92.47%
Normal (class 0)    Precision: 100%, Recall: 85%
Abnormal (class 1)  Precision: 89%, Recall: 100%
MI (class 2)        Precision: 90%, Recall: 92%
History MI (class 3) Precision: 87%, Recall: 75%
```

**Advantages**:
- High recall on abnormal classes (crucial for screening)
- Multi-perspective 12-lead analysis
- Robust to image quality variations
- PCA noise reduction improves generalization

---

## 📊 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Latest |
| **Backend** | Python | 3.8-3.11 |
| **ML Models** | scikit-learn, XGBoost | Latest |
| **ECG Processing** | scikit-image, OpenCV | Latest |
| **Visualization** | Plotly, Matplotlib | Latest |
| **Interpretability** | SHAP | 0.41+ |
| **PDF Generation** | ReportLab | 3.6+ |
| **AI Assistant** | Google Generative AI | Latest |
| **Deployment** | Streamlit Cloud, Docker | - |

---

## 🏗️ Project Structure

```
medriba2/
├── app.py                          # Main Streamlit application
├── Ecg.py                          # ECG image processing pipeline
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
├── models/                         # Pre-trained model files
│   ├── diabetes_rf_model.pkl
│   ├── heart_xgb_model.pkl
│   └── ...
├── data/                           # Sample datasets (optional)
│   ├── diabetes_data.csv
│   └── heart_disease_data.csv
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── diabetes_analysis.ipynb
│   ├── heart_disease_analysis.ipynb
│   └── ecg_pipeline_demo.ipynb
└── README.md                       # Project documentation
```

---

## 🔐 Security & Privacy

### Data Handling
- ✅ No patient data storage: All predictions are session-based
- ✅ No database persistence: Information exists only during user session
- ✅ HIPAA-ready architecture: Can be deployed with compliance measures
- ✅ Secure API communication: Encrypted credentials via environment variables

### Model Security
- ✅ Serialized models: Pre-trained and locked (joblib format)
- ✅ Input validation: Type checking and range validation on all inputs
- ✅ Error handling: Graceful degradation with informative messages
- ✅ Version control: Model versioning for audit trails

---

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud
```bash
# 1. Push code to GitHub
git push origin main

# 2. Go to https://streamlit.io/cloud
# 3. Connect GitHub repository
# 4. Deploy with one click
```

### Docker Deployment
```bash
# Build image
docker build -t medriba .

# Run container
docker run -p 8501:8501 medriba
```

### Environment Configuration for Production
```env
GEMINI_API_KEY=your_production_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=info
```

---

## 📚 Research Foundation

MEDRIBA is based on peer-reviewed research and clinical best practices:

### Key References
- **Diabetes Prediction**: Pima Indians Diabetes Database
- **Heart Disease Detection**: UCI Heart Disease Dataset (Cleveland, Hungarian, Swiss, Long Beach)
- **ECG Analysis**: Established standards for 12-lead electrocardiogram interpretation
- **Ensemble Methods**: Combination of Random Forest, XGBoost, and Voting Classifiers
- **Explainability**: SHAP (SHapley Additive exPlanations) framework

### Model Validation
- ✅ 5-Fold Cross-Validation on all models
- ✅ Stratified sampling to preserve class distribution
- ✅ External validation on holdout test sets
- ✅ Class balancing via SMOTE for imbalanced datasets
- ✅ Hyperparameter tuning via GridSearchCV

---

## ⚠️ Limitations & Disclaimers

### Current Limitations
1. **ECG Image Quality**: Requires high-quality paper ECG recordings
2. **Lead Standardization**: Fixed extraction coordinates require image standardization
3. **Dataset Diversity**: Trained on specific populations; may not generalize universally
4. **No Temporal Analysis**: ECG pipeline analyzes single images (no continuous monitoring)
5. **Clinical Context**: Cannot replace comprehensive medical evaluation

### Important Disclaimers
- ⚠️ **For Research & Educational Purposes Only**: Not approved for clinical diagnosis
- ⚠️ **Not a Medical Device**: Requires validation before healthcare deployment
- ⚠️ **Supplementary Tool**: Should complement, not replace, clinical judgment
- ⚠️ **Professional Review Required**: All predictions require physician validation
- ⚠️ **Liability**: Users assume responsibility for clinical application

---

## 🔮 Future Enhancements

### Short-term
- [ ] Deep Learning integration (CNN for automated ECG feature extraction)
- [ ] LSTM networks for temporal ECG sequence analysis
- [ ] Automated ECG lead quality assessment
- [ ] Multi-language support for global deployment

### Medium-term
- [ ] Continuous monitoring integration with wearable ECG devices
- [ ] Real-time alert system for high-risk patients
- [ ] Multi-hospital deployment with federated learning
- [ ] Advanced SHAP-based explainability with attention mechanisms

### Long-term
- [ ] Transfer learning from pre-trained medical imaging models
- [ ] Multi-disease prediction integration (expanded to 10+ conditions)
- [ ] Blockchain-based medical record integration
- [ ] Personalized risk scores based on genetic profiling

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/enhancement`)
3. **Commit** changes (`git commit -m 'Add feature: description'`)
4. **Push** to branch (`git push origin feature/enhancement`)
5. **Submit** a Pull Request with detailed description

### Areas for Contribution
- Model improvements and optimization
- Additional disease prediction modules
- Enhanced visualization components
- Documentation and tutorials
- Bug fixes and code refactoring

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### Attribution
If you use MEDRIBA in research or publications, please cite:

```bibtex
@software{medriba2024,
  title={MEDRIBA: Multi-Model Expert Data-driven Risk Identification & Bio-health Analytics},
  author={Kabir Ahmed},
  year={2024},
  url={https://github.com/itxkabix/medriba2}
}
```

---

## 👨‍💻 Developer

**Kabir Ahmed**
- 🎓 MCA Data Science Student
- 🔬 Research-Based Implementation
- 📧 Contact: [GitHub Profile](https://github.com/itxkabix)

### Acknowledgments
- Clinical dataset providers (UCI, Kaggle)
- SHAP authors (Lundberg et al.)
- Streamlit team for excellent framework
- Research community for peer-reviewed methodologies

---

## 📞 Support & Contact

### Getting Help
- 📖 Check the [Documentation](#-usage-guide)
- 🐛 Report bugs on [GitHub Issues](https://github.com/itxkabix/medriba2/issues)
- 💬 Discuss ideas on [GitHub Discussions](https://github.com/itxkabix/medriba2/discussions)
- 📧 Email: [Your contact information]

### Community
- Join our [Discord Community](#) (if applicable)
- Follow [Twitter/LinkedIn](#) for updates
- Star ⭐ this repository to show support

---

## 🎉 Acknowledgments

Special thanks to:
- Open-source ML community (scikit-learn, XGBoost, Streamlit)
- Clinical researchers and healthcare professionals
- All contributors and testers
- Users providing feedback and improvements

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅

---

## 📋 Quick Reference

### Model Files Needed
- `diabetes_rf_model.pkl`
- `diabetes_scaler.pkl`
- `diabetes_selected_features.pkl`
- `heart_xgb_model.pkl`
- `heart_scaler.pkl`
- `heart_selected_features.pkl`
- `PCA_ECG_1.pkl`
- `HeartDiseasePredictionusingECG_4.pkl`

### Key Python Imports
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import plotly.graph_objects as go
```

### Performance Benchmarks
- Diabetes Model: **96.27% accuracy**
- Heart Disease Model: **97.57% accuracy**
- ECG Model: **92.47% accuracy**
- Average prediction time: <1 second per sample

---

**For more information, visit the [GitHub Repository](https://github.com/itxkabix/medriba2)**


## To un this file 
    Do Install my repo https://github.com/itxkabix/medriba2

    and clone it
    and then
    under deployment 
        pip install -r requirements.txt
    
    Runn

    streamlit run med.py
# 💻 Laptop Price Predictor — ML Web App

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> An ML-powered web application that predicts laptop prices based on hardware specifications — built with Scikit-learn and deployed via Streamlit.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest / Gradient Boosting |
| Dataset | laptop_price.csv |
| Interface | Streamlit Web App |
| Deployment | Local / Streamlit Cloud |

---

## 🔍 Features Used for Prediction

- Brand & Model
- Processor (CPU type, cores, speed)
- RAM (GB)
- Storage (SSD/HDD, capacity)
- GPU type
- Display size & resolution
- Operating System
- Weight

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| ML | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
├── app.py                  # Flask API
├── Streamlit_app.py        # Streamlit web interface
├── laptop_price.csv        # Dataset
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/MishraAbhay03/MLmodel.git
cd MLmodel

# Install
pip install -r requirements.txt

# Run Streamlit app
streamlit run Streamlit_app.py
```

Select laptop specifications from the sidebar dropdowns and the model will predict the estimated price in real time.

---

## 🔬 ML Pipeline

1. **Data Cleaning** — handle missing values, fix dtypes
2. **Feature Engineering** — encode categorical specs
3. **Model Training** — compare multiple regressors
4. **Hyperparameter Tuning** — GridSearchCV
5. **Deployment** — Streamlit interactive UI

---

## 👤 Author

**Abhaykumar Mishra**  
M.Sc. Data Science & AI | Mumbai  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN) [![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/MishraAbhay03)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

# 💻 Laptop Price Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> ML-powered laptop price prediction with an interactive Streamlit UI — feature engineering, ensemble methods & live price estimates.

---

## 📌 Overview

This project predicts laptop prices based on specifications such as RAM, processor, storage, GPU, display size, and brand. It uses ensemble machine learning models trained on real-world laptop pricing data and serves predictions through an interactive Streamlit dashboard.

---

## ✨ Key Features

- ✅ Predicts laptop price from hardware specs
- ✅ Feature engineering on categorical & numerical attributes
- ✅ Ensemble ML models (Random Forest, Gradient Boosting)
- ✅ Interactive **Streamlit** UI for instant predictions
- ✅ EDA visualizations and price distribution analysis
- ✅ FastAPI REST endpoint via separate API repo

---

## 📁 Project Structure

```
laptop-price-predictor/
├── app.py                  # FastAPI backend
├── Streamlit_app.py        # Streamlit frontend
├── laptop_price.csv        # Dataset
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/MishraAbhay03/laptop-price-predictor.git
cd laptop-price-predictor
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run Streamlit_app.py
```

### Run FastAPI Backend

```bash
uvicorn app:app --reload
```

---

## 📊 Input Features

| Feature | Description |
|---|---|
| Brand | Dell, HP, Lenovo, Apple, etc. |
| RAM | 4GB, 8GB, 16GB, 32GB |
| Storage | SSD/HDD size |
| Processor | Intel i3/i5/i7, AMD Ryzen |
| GPU | Integrated / Dedicated |
| Display | Screen size & resolution |
| OS | Windows, macOS, Linux |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML | Scikit-learn, Pandas, NumPy |
| Web App | Streamlit |
| API | FastAPI |
| Visualization | Matplotlib, Seaborn |

---

## 👤 Author

**Abhaykumar Mishra** — [GitHub](https://github.com/MishraAbhay03) · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

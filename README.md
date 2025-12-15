# Laptop Price Prediction Model

A machine learning project to predict laptop prices based on various specifications and features. This model leverages advanced machine learning techniques to provide accurate and reliable price estimations.

---

## 🚀 Features
- Predicts laptop prices based on input features like brand, processor type, RAM, storage, and more.
- Utilizes multiple machine learning algorithms, including:
  - Ensemble methods
  - K-Nearest Neighbors (KNN)
  - Neural Networks (CNN)
- Implements data preprocessing techniques such as feature scaling and encoding for optimal performance.
- Provides insights into feature importance and model evaluation metrics.

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - NumPy, Pandas (Data manipulation)
  - Scikit-learn (Model building and evaluation)
  - Matplotlib, Seaborn (Data visualization)
  - TensorFlow/Keras (Deep learning for CNN)
  
---

## 📂 Project Structure
laptop-price-prediction/ 
├── data/ │ 
├── raw_data.csv # Raw dataset │ 
├── cleaned_data.csv # Preprocessed dataset 
├── models/ │ 
├── final_model.pkl # Trained machine learning model │ 
├── cnn_model.h5 # Trained CNN model ├── notebooks/ │ 
├── data_analysis.ipynb # Exploratory Data Analysis (EDA) │ 
├── model_training.ipynb # Model training and evaluation ├── app/ │ 
├── app.py # Flask or Streamlit application │ 
├── templates/ # HTML templates (if applicable) 
├── requirements.txt # Python dependencies 
├── README.md # Project documentation └── LICENSE # License file


## 📊 Dataset
- **Source**: [Dataset source or description].
- **Features**:
  - `Brand`: Brand of the laptop.
  - `Processor`: Type and generation of the processor.
  - `RAM`: Size of the RAM in GB.
  - `Storage`: Type and capacity of storage (e.g., SSD or HDD).
  - `Screen Size`: Screen dimensions.
  - `Operating System`: OS type (e.g., Windows, macOS, Linux).
  - `Price`: Target variable representing the laptop price.

---

## ⚙️ How It Works
1. **Data Preprocessing**:
   - Handles missing values, outliers, and inconsistent data.
   - Encodes categorical features using One-Hot Encoding or Label Encoding.
   - Scales numerical features for better model performance.
   
2. **Model Training**:
   - Trained using algorithms such as Linear Regression, Random Forest, and CNN.
   - Fine-tuned hyperparameters using GridSearchCV for optimal results.
   
3. **Model Evaluation**:
   - Assessed using metrics like Mean Absolute Error (MAE) and R² score.
   - Visualized feature importance and residuals for better interpretability.

4. **Prediction**:
   - Accepts user input for features and returns the predicted price.

---

## 🚀 Getting Started
### Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt

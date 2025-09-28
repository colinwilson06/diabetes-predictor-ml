## 🩺 Diabetes Prediction App

An interactive web application to predict whether a patient has diabetes based on medical diagnostic data. Built with **Python**, **Streamlit**, and **Logistic Regression**, this app allows users to input patient data and get instant predictions.

🔗 **Try the live app:**  
[https://diabetes-predictor-ml-6nggut8qx6jnekzb3p73ns.streamlit.app/](https://diabetes-predictor-ml-6nggut8qx6jnekzb3p73ns.streamlit.app/)

📺 **Watch the demo video:**  
[https://youtu.be/QVGM2wDyfQs](https://youtu.be/QVGM2wDyfQs)

---

## 📂 Project Structure

```bash
diabetes-predictor-ml/
│── demo/                       # Folder containing demo video of app usage
│── screenshots/                # Folder containing screenshots of app usage and results
│── Diabetes_Logistic2.ipynb    # Jupyter Notebook with model training & evaluation
│── app_diabetes.py             # Streamlit application for interactive prediction
│── diabetes_model.pkl          # Trained Logistic Regression model (saved)
│── diabetes_prediction_dataset.csv  # Dataset used for training the model
│── requirements.txt            # Python dependencies
│── scaler.pkl                  # Saved StandardScaler for input normalization
└── README.md                   # Project documentation
```

---

## 💻 Features

- **Interactive Web App** built with Streamlit  
- **User Input:** Age, BMI, Blood Glucose, HbA1c, Hypertension, Heart Disease, Gender, Smoking History  
- **Prediction Output:** Likely or Unlikely Diabetes (user-friendly display)  
- **Health Recommendation:** Advice based on prediction  
- **Machine Learning Model:** Logistic Regression  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC  

---

## 📊 Dataset

- **Source:** Kaggle – Diabetes Prediction Dataset  
- **Features include:**  
  - Age  
  - BMI  
  - Blood Glucose Level  
  - HbA1c Level  
  - Gender, Smoking History, Hypertension, Heart Disease  
- **Target:** `diabetes` (0 = No Diabetes, 1 = Diabetes)  

---

## 🔎 Web App Demo

### **Screenshots**
The `screenshots/` folder contains images documenting the **model training, evaluation, and exploratory data analysis (EDA)**.  
Click [here](screenshots/) to view all screenshots.

### **Video Demo**
- [Watch Demo Video](https://youtu.be/QVGM2wDyfQs)


---

## 🤖 Model Evaluation

- **Accuracy:** 94%  
- **Precision:** 95% (Diabetes), 93% (No Diabetes)  
- **Recall:** 92% (Diabetes), 96% (No Diabetes)  
- **F1-score:** 94%  
- **AUC (ROC):** 0.988  

**Confusion Matrix:**  
```bash
[[16664   775]   → True Negatives / False Positives
 [ 1323 16304]]   → False Negatives / True Positives
```

> The model shows strong performance in distinguishing diabetic and non-diabetic patients. High recall for diabetes means it effectively detects actual diabetic patients.

---

## 🚀 How to Run

### **Run Streamlit App (Recommended)**
1. Clone this repository:
```bash
git clone https://github.com/colinwilson06/diabetes-predictor-ml.git
cd diabetes-predictor-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app_diabetes.py
```

4. Open your browser → go to `http://localhost:8501`  


**Or access the live app directly from this link:**  
[https://diabetes-predictor-ml-6nggut8qx6jnekzb3p73ns.streamlit.app/](https://diabetes-predictor-ml-6nggut8qx6jnekzb3p73ns.streamlit.app/)

---

### **Run Jupyter Notebook (Optional)**
If you want to see the model training & evaluation:

```bash
jupyter notebook src/Diabetes_Logistic2.ipynb
```

---

## 📦 Requirements

- Python 3.10+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- imbalanced-learn  
- streamlit  

Install all dependencies with:  
```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Dataset: `data/diabetes_prediction_dataset.csv`  
- Screenshots: `screenshots/`  
- Video demo: `demo/`  
- Prediction output in app shows **Likely / Unlikely Diabetes** with simple recommendation.  
- Streamlit app allows users to **interactively test the model** without coding.  

---


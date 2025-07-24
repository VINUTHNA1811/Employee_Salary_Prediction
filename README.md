## 💼 Employee Salary Prediction

A machine learning web application that predicts whether an employee earns more than \$50K/year based on demographic and professional information. Built as part of an internship project with full-cycle development: from data cleaning and model training to UI deployment.

🔗 [Live Demo](https://employeesalarypredictiongit-vin.streamlit.app/)  

---

### 📌 Project Objective

The goal of this project is to build a predictive system that classifies employees into two income brackets:

* `>50K/year`
* `<=50K/year`

---

### 🔍 Dataset Overview

* **Source**: UCI Adult Income Dataset
* **Records**: \~45,000
* **Target Feature**: `income`
* **Features**: Age, Education, Workclass, Hours/week, Occupation, Capital gain/loss, etc.

---

### 📊 Exploratory Data Analysis

Key EDA Insights:

* Visualized distributions (age, education, hours/week)
* Identified missing/placeholder values ('?')
* Correlation heatmaps
* Income distribution comparisons across features

---

### 🛠️ Machine Learning Models Compared

| Model               | Accuracy   | F1 Score |
| ------------------- | ---------- | -------- |
| Logistic Regression | 85.1%      | 0.66     |
| Random Forest       | 85.0%      | 0.67     |
| KNN                 | 82.2%      | 0.61     |
| SVM                 | 85.2%      | 0.65     |
| Decision Tree       | 81.3%      | 0.61     |
| **CatBoost**        | **87.35%** | **0.72** |

✅ **CatBoost Classifier** was selected as the final model based on highest accuracy and F1 score.

---

### 🧠 Final Model Details

* **Algorithm**: CatBoost Classifier
* **Accuracy**: 87.35%
* **F1 Score**: 0.72
* **Saved Components**:

  * `catboost_salary_model.pkl`
  * `cat_features.pkl`
  * `column_order.pkl`

---

### 🚀 Web App Features

#### 🏠 Home

* Project overview
* Features summary
* Instructions

#### 📥 Predict Salary

* Interactive form to enter employee details
* Real-time predictions with confidence
* Downloadable prediction reports (TXT/JSON)

#### 📈 Batch Processing

* Upload CSV files for bulk predictions
* Visual summary of predictions
* Download processed results

---

### 🧾 Report Generation

After each prediction, users can download:

* 📄 TXT Report
* 📊 JSON Summary

Reports include prediction confidence and employee profile breakdown.

---

### 📦 Installation

```bash
git clone https://github.com/VINUTHNA1811/Employee_Salary_Prediction.git
cd Employee_Salary_Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

### 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **ML Models**: scikit-learn, CatBoost
* **Visualization**: seaborn, matplotlib
* **Deployment**: Streamlit Cloud

---

### 📁 Project Structure

```
├── app/
│   └── app.py
├── models/
│   ├── catboost_salary_model.pkl
│   ├── cat_features.pkl
│   └── column_order.pkl
├── eda_and_modeling.ipynb
├── requirements.txt
└── README.md
```

---
### 🙋‍♀️ Author
📬 [LinkedIn – Budde Vinuthna](https://www.linkedin.com/in/budde-vinuthna-231642345)  
💻 [GitHub – VINUTHNA1811](https://github.com/VINUTHNA1811)


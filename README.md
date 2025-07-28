# Advanced_AI_ML_Internship_Tasks

🧠 AI/ML Engineering Internship – Task Submission  
🏢 Organization: DevelopersHub Corporation  
📅 Submission Date: 28th June 2025  
👨‍🎓 Submitted by: Shariff Mehmood (f2021065011)  
📂 Repository: AI_ML_Internship_Tasks  

---

✅ Completed Tasks

| Task No. | Title                             | ML Type        | Model Used                |
|----------|-----------------------------------|----------------|---------------------------|
| Task 1   | AG News Text Classification       | NLP / Text     | BERT (Transformer)        |
| Task 2   | Customer Churn Prediction         | Classification | Logistic Regression / RF  |
| Task 3   | Titanic Survival Prediction       | Classification | Logistic Regression       |

---

🔍 Task Summaries

🟩 **Task 1: AG News Text Classification using BERT**  
**Objective:** Classify news articles into one of four categories using a pretrained transformer.  
**Dataset:** AG News (via Hugging Face `datasets`)  
**Model:** `BertForSequenceClassification`  
**Libraries:** `transformers`, `datasets`, `torch`, `sklearn`  
**Techniques:**
- Tokenization via `BertTokenizerFast`
- Fine-tuning on AG News dataset
- Evaluation using accuracy and F1-score  
**Results:**  
- Accuracy: ~94% (varies slightly)  
- F1 Score: ~0.94 (weighted)  
📁 Files:
- `task1_bert_news_classifier.py`
- `bert_news_model/` (model directory)
- `app.py` (optional Streamlit UI)

---

🟨 **Task 2: Customer Churn Prediction Pipeline**  
**Objective:** Predict whether a customer will churn based on service usage and demographics.  
**Dataset:** Telco Customer Churn (CSV required)  
**Models:** Logistic Regression, Random Forest (grid searched)  
**Techniques:**
- Data cleaning and imputation
- Categorical encoding and scaling
- Pipeline with `ColumnTransformer` + `GridSearchCV`  
**Metrics:**  
- Accuracy, Precision, Recall, F1-score  
**Result:**  
- Best model depends on grid search (Logistic or RF)  
📁 File: `task2_churn_pipeline.py`  
📁 Model: `churn_pipeline.pkl`  

---

🟦 **Task 3: Titanic Survival Prediction**  
**Objective:** Predict if a passenger survived the Titanic disaster based on features.  
**Dataset:** Titanic Dataset (CSV)  
**Model:** Logistic Regression  
**Techniques:**
- Handling missing values
- Encoding categorical features
- Model training and evaluation  
**Metrics:** Accuracy, Confusion Matrix  
**Result:**
- Accuracy: ~80–85% (varies with preprocessing)  
📁 File: `task3_titanic_survival.py`  
📁 Model: `titanic_model.pkl`

---

🧪 How to Run

Install required packages:
'''bash
pip install pandas matplotlib seaborn scikit-learn transformers datasets torch streamlit joblib
▶️ Running Python Scripts
Run each script in the terminal:

'''bash
python task1_bert_news_classifier.py
python task2_churn_pipeline.py
python task3_titanic_survival.py
▶️ Running Streamlit App (Optional for Task 1)
'''bash
streamlit run app.py
📥 Dataset Notes
Ensure the following datasets are placed in the root directory:

Telco-Customer-Churn.csv (for Task 2)

titanic.csv (for Task 3)

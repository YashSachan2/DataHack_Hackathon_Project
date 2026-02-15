# 💉 Vaccine Usage Prediction System

This project is a machine learning solution designed to predict the likelihood of individuals receiving two specific vaccines: the **H1N1 vaccine** (labeled as `xyz_vaccine`) and the **Seasonal Flu vaccine**. By analyzing behavioral, demographic, and opinion-based data, the system identifies key factors driving vaccination rates.

---

## 📊 Dataset Metrics

The project utilizes a dataset containing survey responses regarding health behaviors and demographics.

* **Training Set Size**: 26,707 records
* **Test Set Size**: 26,708 records
* **Raw Features**: 36 columns (including behavioral flags, health opinions, and demographics)
* **Processed Features**: 105 columns (after One-Hot Encoding and Scaling)

### Key Features
* **Behavioral**: `behavioral_antiviral_meds`, `behavioral_wash_hands`, `behavioral_face_mask`
* **Demographic**: `age_group`, `education`, `race`, `sex`, `income_poverty`
* **Opinions**: `opinion_xyz_vacc_effective`, `opinion_seas_risk`

---

## 🛠️ Data Preprocessing Pipeline

To prepare the data for modeling, a robust preprocessing pipeline was implemented:

1.  **Imputation**: Missing values were handled using a `SimpleImputer` with the "most_frequent" strategy.
2.  **Encoding**: Categorical variables (e.g., `hhs_geo_region`, `employment_industry`) were transformed using `OneHotEncoder`.
3.  **Scaling**: Numerical features were standardized using `StandardScaler`.


---

## 🤖 Model Evaluation & Selection

The project evaluated multiple classification algorithms using **Accuracy** as the primary metric via 10-fold cross-validation.

### Target 1: H1N1 Vaccine (`xyz_vaccine`)
| Model | Cross-Val Accuracy |
| :--- | :--- |
| **Gradient Boosting** | **84.00%** |
| SVC | 83.81% |
| Logistic Regression | 83.61% |
| Random Forest | 83.58% |
| XGBoost | 83.54% |
| Naive Bayes | 64.99% |

### Target 2: Seasonal Vaccine (`seasonal_vaccine`)
| Model | Cross-Val Accuracy |
| :--- | :--- |
| **LightGBM** | **78.53%** |
| Gradient Boosting | 78.50% |
| Logistic Regression | 78.04% |
| Random Forest | 77.73% |
| XGBoost | 77.63% |

---

## 🏆 Final Model Architecture

Based on the evaluation metrics, a heterogeneous ensemble approach was selected for the final submission:

* **H1N1 Prediction Model**: `XGBClassifier` (XGBoost)
    * Selected for its high accuracy and efficiency in handling sparse data.
* **Seasonal Flu Prediction Model**: `LGBMClassifier` (LightGBM)
    * Selected for achieving the highest accuracy (78.53%) on the seasonal target.

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm

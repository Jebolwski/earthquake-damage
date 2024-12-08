# Earthquake Damage Prediction Project

## Overview
This project focuses on predicting building damage levels caused by earthquakes using machine learning techniques. The dataset includes various features about buildings and their surroundings, such as structural attributes, building age, and ground conditions. The objective is to classify the severity of damage (`damage_grade`) into three levels:

- **Damage Grade 1**: Minor damage  
- **Damage Grade 2**: Moderate damage  
- **Damage Grade 3**: Severe damage  

Machine learning models like **Random Forest** and **XGBoost** were trained. To handle class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) was used.

---

## Features
The dataset includes the following key features:

- **Structural Attributes**:  
  - `has_superstructure_adobe_mud`, `has_superstructure_cement_mortar_stone`, etc.  
- **Environmental Attributes**:  
  - `land_surface_condition`, `foundation_type`, `ground_floor_type`  
- **Building Information**:  
  - `plinth_area_sq_ft`, `age_building`, `height_ft_pre_eq`, `count_floors_pre_eq`  
- **Earthquake Impact**:  
  - `magnitude_felt`, `magnitude`  

Additional engineered features:
- `damage_potential` = `magnitude_felt` × `age_building`  
- `structure_risk` = `count_floors_pre_eq` ÷ (`height_ft_pre_eq` + 1)  
- `foundation_age_interaction` = `foundation_type` × `age_building`

---

## Methodology

### 1. **Data Preprocessing**
- Label encoding was applied to categorical variables.
- Missing values were imputed as required.
- Features were normalized and scaled for some models.

### 2. **Class Imbalance Handling**
- SMOTE was applied to balance the target variable `damage_grade`.

### 3. **Model Training**
The following machine learning models were trained:
- **Random Forest**
- **XGBoost**


### 4. **Evaluation**
- Models were evaluated using accuracy scores and cross-validation.
- **Random Forest** achieved the best accuracy of **93.5%**.

---

## Results


structure.ipynb
| Model                 | Accuracy   |
|-----------------------|------------|
| Random Forest         | **93.5%**  |
| XGBoost               | 84.5%      |

structure.ipynb
| Model                 | Accuracy   |
|-----------------------|------------|
| Random Forest         | **41.5%**  |
| XGBoost               | 40.2%      |


### Improvements
- **Feature Engineering**: Added interaction terms for better representation.
- **Hyperparameter Tuning**: Used RandomizedSearchCV for optimal parameters.
- **Cross-validation**: Ensured robustness with StratifiedKFold.

---

## Requirements

To run the project, ensure the following libraries are installed:
```bash
pip install pandas numpy scikit-learn matplotlib xgboost imbalanced-learn tensorflow
```
In the first.ipynb file, the damage_grade is estimated by only taking certain columns without processing the csv file. In the structure.ipynb file, we increase the accuracy value with our own developments.

# Machine Learning Applications in Engineering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Active-green)

## 📌 Overview
This repository bridges the gap between **Mechanical Engineering** and **Data Science**. It contains a progression of projects ranging from raw data engineering for robotics to advanced materials informatics, focusing on **Fatigue Analysis** and **Inverse Kinematics**.

## 📂 Project Modules

### 1. Robotics Data Engineering (ABB IRB 2400)
**Objective:** Build a machine learning-ready dataset for the Inverse Kinematics (IK) problem.
* **System:** 6-DOF Anthropomorphic Manipulator (ABB IRB 2400).
* **Process:** * Collected simulation data mapping End-Effector Poses ($x, y, z, yaw, pitch, roll$) to Joint Angles ($q_1...q_6$).
    * Performed data cleaning (handling kinematic singularities and redundant solutions).
    * Established version control (DVC/Git) for large engineering datasets.

### 2. Fatigue Strength Prediction (Regression Analysis)
**Objective:** Minimize RMSE in predicting the fatigue strength of steel using high-dimensional feature spaces.
* **Techniques:**
    * **Feature Engineering:** Polynomial feature expansion (Degree 2-4) to capture non-linear material behaviors.
    * **Regularization:** Compared **LASSO (L1)**, **Ridge (L2)**, and **ElasticNet** to handle multicollinearity in 350+ features.
    * **Non-Linear Models:** Implemented **Kernel Ridge Regression (KRR)** with RBF kernels and **KNN Regressor**.
* **Analysis:**
    * **Training Time vs. Complexity:** Benchmarked computational costs of hyperparameter tuning for KRR vs. Linear models.
    * **Learning Curves:** Diagnosed overfitting in high-variance models (KRR).

### 3. Materials Informatics (NOMAD API)
**Objective:** Create a custom dataset of >1,000 bulk materials to predict electronic properties.
* **Source:** **NOMAD Laboratory API** (v1 Archive & Query).
* **Engineering Logic:** Implemented custom Regex parsers to calculate atomic counts from chemical formulas (e.g., parsing `Cu2O` $\to$ `N_Atoms=3`).
* **Outcome:** A clean dataset of Bulk Copper-based materials ready for supervised learning.

### 4. Classification & Uncertainty Quantification
**Objective:** Classify materials as "High" or "Low" fatigue life and quantify prediction confidence.
* **Algorithms:**
    * **SVM:** Optimized Polynomial vs. RBF kernels using GridSearch.
    * **Decision Trees:** Visualized physical logic flows for interpretability.
    * **Gaussian Process Regression (GPR):** Utilized `WhiteKernel` + `RBF` to provide **uncertainty intervals ($\sigma$)**, critical for safety margins in R&D.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn (Pipelines, GridSearchCV, KFold)
* **Data Processing:** Pandas, NumPy, Regex
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ayberkkutlu/Machine-Learning-Course-and-Project.git](https://github.com/ayberkkutlu/Machine-Learning-Course-and-Project.git)

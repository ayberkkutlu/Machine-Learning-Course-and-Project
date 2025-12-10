# Machine Learning Course & Project

This repository serves as a comprehensive archive of the coursework, assignments, and the semester-long project for the Machine Learning course. It demonstrates the application of classical machine learning algorithms, deep learning models, and hybrid control strategies to solve engineering problems.

## 📂 Repository Structure & Detailed Contents

### 1. Course Project: Hybrid Inverse Kinematics for ABB IRB 2400
**Objective:** Develop a robust, high-speed Inverse Kinematics (IK) solver for the ABB IRB 2400 industrial robot, overcoming the rigidity of analytical methods and the inaccuracy of pure machine learning approaches.

* **Phase 1 - Geometric Forensic Analysis:**
    * **Challenge:** Standard Denavit-Hartenberg (D-H) parameters yielded massive geometric errors (~42 cm).
    * **Methodology:** Performed numerical optimization to reverse-engineer the dataset's kinematic parameters.
    * **Key Discovery:** Identified a non-standard **154.7 mm Tool Center Point (TCP) offset** and specific joint zero-position offsets ($q_2=0^\circ, q_3=180^\circ$).
    * **Result:** Established a valid ground truth by reducing geometric error to **< 10 mm**.

* **Phase 2 - Machine Learning (The "Coarse Estimator"):**
    * **Model:** Multi-Layer Perceptron (MLP) with a "Funnel" architecture (512-256-128 neurons).
    * **Feature Engineering:** Implemented Physics-Informed features including pre-calculated Cartesian error vectors ($dx, dy, dz$) and trigonometric encoding ($\sin/\cos$) to avoid angular discontinuities.
    * **Outcome:** Achieved Mean Absolute Error (MAE) of **1.9°**. Insufficient for precision tasks alone, but ideal for initializing a numerical solver.

* **Phase 3 - The Hybrid AI-Numeric Solver:**
    * **Algorithm:** **Predict-Verify-Correct**. The MLP provides an initial guess, which is refined by a custom **Damped Least Squares (DLS)** loop with Adaptive Damping.
    * **Performance:** 84.5% Success Rate (<1mm) vs 0% for pure ML; Average time 1.20 ms (**~833 Hz**).

---

### 2. Homework 1: Foundations & Classical ML
This section covers fundamental data preprocessing and regression concepts applied to the "Fatigue" dataset.

* **[`HW1_Items1-10.pdf`](HW1_Items1-10.pdf) - Linear Regression Pipeline**
    * **Data Preprocessing:** Implemented manual data shuffling and splitting (Training/Validation/Testing).
    * **Standardization:** Calculated mean and standard deviation from the training set to manually standardize (Z-score normalization) all subsets.
    * **Modeling:** Trained a **Linear Regression** model using Scikit-Learn.
    * **Evaluation:** Computed Root Mean Squared Error (RMSE) for training and validation sets and generated "Actual vs. Predicted" scatter plots to visualize model performance.

* **[`HW1-Items11-16.pdf`](HW1-Items11-16.pdf) - Random Forests & Sensitivity Analysis**
    * **Ensemble Modeling:** Implemented a **Random Forest Regressor** to improve predictive performance over linear models.
    * **Hyperparameter Tuning:** Conducted a sensitivity analysis by varying the number of trees (`n_estimators`) from 1 to 200.
    * **Overfitting Analysis:** Plotted "RMSE vs. Number of Trees" for both Training and Validation sets to identify the point of diminishing returns and detect overfitting.
    * **Interpretability:** Extracted and visualized **Feature Importance** to determine which physical parameters most strongly influenced fatigue life.

---

### 3. Homework 2: Advanced ML & Deep Learning
This homework delves into advanced classification techniques and introduces Deep Learning for computer vision tasks.

* **[`HW2_Items1-9.pdf`](HW2_Items1-9.pdf) - Gradient Boosting & Classification**
    * **Objective:** Binary classification of fatigue life ("Good" vs. "Bad") using ensemble methods.
    * **Methodology:**
        * **Gradient Boosting:** Implementation of Scikit-Learn's `GradientBoostingClassifier` and `XGBoost`.
        * **Parameter Tuning:** Analysis of how Learning Rate and Tree Depth affect model convergence and accuracy.
    * **Evaluation:**
        * **AUC-ROC:** Computed Area Under the Curve (AUC) scores to evaluate the trade-off between True Positive Rate and False Positive Rate.
        * **Confusion Matrix:** Visualized classification performance to identify false alarms vs. missed detections.

* **[`HW2_Items10-15.pdf`](HW2_Items10-15.pdf) - CNNs & Transfer Learning**
    * **Objective:** Image Classification on the **Atomagined** dataset (Simulated atomic-resolution HAADF STEM images).
    * **Task (Item 12):** Classify images based on their symmetry group (`symmetry_Int_Tables_number`).
    * **Methodology:**
        * **Data Pipeline:** Implemented a custom data loader to match `ref_id` from `key.csv` to image filenames on disk.
        * **Transfer Learning:** Utilized the **Xception** architecture (pre-trained on ImageNet) with a custom dense classification head.
        * **Regularization:** Applied Data Augmentation (Rotation, Width/Height Shift, Horizontal Flip) and Dropout (0.5) to prevent overfitting on the small dataset.
    * **Result:** Achieved **100% Validation Accuracy** on the target subset and successfully classified holdout images with high confidence.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/ayberkkutlu/Machine-Learning-Course-and-Project.git](https://github.com/ayberkkutlu/Machine-Learning-Course-and-Project.git)
cd Machine-Learning-Course-and-Project
````

### 2\. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install the required libraries (Pandas, Scikit-Learn, TensorFlow, XGBoost, etc.):

```bash
pip install -r requirements.txt
```

-----

## 📊 Dataset Information

### ABB IRB 2400 Kinematics (Project)

  * **Description:** Dataset containing joint configurations ($q_1 \dots q_6$) and corresponding End-Effector positions ($x, y, z, q_x, q_y, q_z, q_w$).
  * **Features:** Includes non-standard TCP offsets and specific joint zero-positions derived via forensic analysis.

### Atomagined Dataset (HW2)

  * **Source:** [MaterialEyes/atomagined](https://github.com/MaterialEyes/atomagined)
  * **Description:** Simulated atomic-resolution HAADF STEM imaging dataset.
  * **Usage:** Used the "Small Subset" (Retrieval/Targets) for symmetry class prediction.

### Fatigue Dataset (HW1/HW2)

  * **Description:** Engineering dataset containing stress/strain parameters and cycle life data.
  * **Source:** Data can be found on Kaggle as Fatigue Dataset.

-----

## 👤 Author

**Ayberk Kutlu**

  * **Institution:** Bilkent University, Mechanical Engineering
  * **Focus:** Mechatronics, Control Systems, Data-Driven Engineering

<!-- end list -->
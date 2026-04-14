# Bank Customer Churn — Statistical Learning

A full supervised learning pipeline for binary churn classification, developed as coursework for the *Statistical Learning* module (MSc in Statistics for Data Science, Universidad Carlos III de Madrid). The project spans probabilistic classifiers, margin-based and tree-based methods, and a PyTorch neural network, with a consistent cost-sensitive decision framework across all parts.

**Authors:** María Colado Montañés · Luis Calderón Yunda

---

## Dataset

[Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn) — 10,000 records, 14 predictors (demographic, financial, and bank-relationship variables), binary target `Exited`.

Key preprocessing decisions:
- `RowNumber`, `CustomerId`, `Surname` dropped (identifiers).
- `Complain` excluded in all parts due to near-perfect data leakage with the target.
- Stratified 80/20 train/test split, fixed seed for reproducibility across parts.

---

## Decision Framework

A common cost-sensitive evaluation framework is maintained throughout the project. False negatives (missed churners) are assigned cost $C_{FN} = 100$ and false positives (false alarms) cost $C_{FP} = 10$, reflecting the asymmetry between losing a customer and the cost of a retention campaign.

The Bayes-optimal operating threshold under this loss structure is:

$$\pi^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{10}{110} \approx 0.09$$

**AUC** is the primary model selection criterion (threshold-independent discrimination); **expected cost** is the practical evaluation metric at the final operating point.

---

## Project Structure

### Part 1 · EDA and Probabilistic Classifiers (`01_probabilistic`)

Exploratory analysis of the class imbalance (~20% churn), feature distributions, and multicollinearity. Probabilistic classifiers fitted and compared:

- **Linear Discriminant Analysis (LDA)** — assumes equal, shared covariance; linear decision boundary.
- **Quadratic Discriminant Analysis (QDA)** — relaxes the common-covariance constraint; quadratic boundary.
- **Naïve Bayes** — conditional feature independence; closed-form posterior via Bayes' theorem.
- **Imbalanced learning** — ROSE (random over-sampling with smoothing) and SMOTE (synthetic minority over-sampling) evaluated against baseline. QDA on original data achieved the best AUC and lowest expected cost, motivating training on the original distribution in Part 2.

*Tools: R (`tidymodels`, `caret`, `MASS`, `ROSE`)*

---

### Part 2 · Discriminative Methods (`02_svm_trees_knn`)

Margin-based and tree-based classifiers, building on the Part 1 evidence of class overlap and non-linearity.

**k-Nearest Neighbours (k-NN)**
- Distance-based, non-parametric; decision boundary determined implicitly by local geometry.
- Hyperparameter *k* tuned by cross-validated AUC.

**Support Vector Machines**
- *Support Vector Classifier (soft-margin SVM):* linear boundary with slack variables; $C$ trades margin width against misclassification.
- *Kernel SVM (RBF):* implicit feature map via the Gaussian kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$; hyperparameters $C$ and $\gamma$ tuned on a grid.
- Hard-margin SVM excluded: evidence from Part 1 of substantial class overlap makes perfect linear separability implausible.
- Class weights (`class.weights`) used inside the SVM objective to further account for imbalance.

**Tree-Based Methods**
- *Decision tree:* recursive binary splitting on information-gain / Gini impurity; pruned by cross-validated cost-complexity ($\alpha$).
- *Random Forest:* bagged ensembles with random feature subsets; variance reduction via averaging over decorrelated trees.
- *Gradient Boosting (XGBoost):* additive model where each tree fits the negative gradient of the loss; tuned on `eta`, `max_depth`, and `nrounds`.

*Tools: R (`e1071`, `kernlab`, `class`, `rpart`, `randomForest`, `xgboost`, `caret`, `pROC`)*

---

### Part 3 · Neural Networks (`03_neural_networks`)

Multi-Layer Perceptron (MLP) implemented in PyTorch, extending the analysis to learned non-linear representations.

Architecture and training:
- Fully connected network with ReLU activations and batch normalisation.
- Binary cross-entropy loss with class weights proportional to inverse class frequency, to address imbalance.
- Adam optimiser; learning rate and weight decay tuned manually. Early stopping monitored on validation AUC.

Evaluation:
- The cost-sensitive threshold $\pi^*$ from the decision framework serves as a theoretical reference; the final operating threshold is determined empirically on the validation set.
- **SHAP values** (DeepExplainer) computed for feature attribution, connecting model predictions to input features via Shapley values from cooperative game theory.

*Tools: Python (`torch`, `scikit-learn`, `shap`, `pandas`, `matplotlib`, `seaborn`)*

---

## Repository Layout

```
bank-churn-statistical-learning/
├── data/
│   └── Customer-Churn-Records.csv
├── reports/                        # rendered HTML outputs (view directly in browser)
│   ├── 01_eda_probabilistic.html
│   ├── 02_svm_trees_knn.html
│   └── 03_neural_networks.html
└── code/                           # source: R Markdown and Jupyter
    ├── 01_probabilistic.Rmd
    ├── 02_svm_trees_knn.Rmd
    └── 03_neural_network.ipynb

```

---

## Reproducing the Analysis

**Parts 1 and 2 (R)**

```r
# Required packages
install.packages(c(
  "tidyverse", "tidymodels", "caret", "MASS", "pROC",
  "e1071", "kernlab", "class", "rpart", "randomForest",
  "xgboost", "ROSE", "janitor", "ggplot2", "patchwork"
))
# Knit the .Rmd files in order; update the data path on line ~40.
```

**Part 3 (Python)**

```bash
pip install torch scikit-learn shap pandas matplotlib seaborn
# Run the Jupyter notebook.
# Update the CSV path in the data-loading cell.
```

---

## Notes

- Absolute file paths in the source files point to local machines and must be updated before re-running.
- Cached `.rds` files used to speed up compilation are excluded from the repository.
- The three parts are designed to be read in order: Part 2 references modelling decisions from Part 1, and Part 3 preserves the same preprocessing pipeline for comparability.

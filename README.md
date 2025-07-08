# Lung Cancer Risk Prediction: A Data-Driven Approach

## Overview
This project develops an advanced machine learning system to classify lung cancer risk levels, providing accessible healthcare screening for underprivileged regions, particularly in countries like South Africa. The model achieves 96% accuracy while prioritizing the minimization of critical false negatives to ensure patient safety.

**Key Metrics**:
- **Model Accuracy**: 96%
- **Patient Records**: 1,000
- **Critical False Negatives**: 0
- **Clinical Features**: 26

## Healthcare Impact & Motivation
Lung cancer is a leading cause of cancer deaths globally, with survival rates improving significantly with early detection (from 4-28% in late stages to 55% in early stages). This project addresses healthcare challenges in underprivileged regions with limited medical resources by providing a cost-effective, questionnaire-based screening tool. It enables healthcare workers in remote areas to identify high-risk patients for immediate medical attention, bridging the gap between limited infrastructure and life-saving early detection.

**Critical Mission**: Inხ

In regions with high pollution and smoking rates, this model can be the difference between life and death, offering guidance where traditional screening is inaccessible or unaffordable.

## Technical Methodology & Innovation

### Data Analysis & Preprocessing
- **Comprehensive EDA**: Analyzed 1,000 patient records with 26 clinical features.
- **Feature Engineering**: Identified optimal predictive features through correlation analysis.
- **Data Quality**: Handled categorical variables and ensured balanced class distribution.
- **Cross-Validation**: Implemented K-fold validation (K=5,10,20,50) for robust evaluation.

### Advanced Model Development
- **Regularization Techniques**: Lasso, Ridge, and Elastic Net optimization.
- **Hyperparameter Tuning**: GridSearchCV for optimal regularization parameters.
- **Ensemble Methods**: Random Forest with Gini and Entropy criteria.
- **Performance Optimization**: Focused on minimizing life-critical false negatives.

### Model Performance & Improvements
| Model                          | Accuracy | Precision | Recall | F1-Score |
|--------------------------------|----------|-----------|--------|----------|
| Baseline Logistic Regression   | 90.3%    | 0.923     | 0.922  | 0.922    |
| Ridge Regression (Final Model) | 96.0%    | 0.962     | 0.961  | 0.961    |
| Random Forest (Entropy)        | 97.5%    | 0.974     | 0.974  | 0.974    |

**Model Selection Rationale**: Ridge Regression was chosen as the final model due to zero critical false negatives (predicting low risk when actually high risk), prioritizing safety in healthcare applications where missing a high-risk patient could be fatal.

### Critical Healthcare Predictions
- **Most Critical: False Negatives** (Predicting LOW risk when actually HIGH risk): **0 cases** – No missed high-risk patients, ensuring no lives are at risk from delayed treatment.
- **Secondary Concern: False Positives** (Predicting HIGH risk when actually LOW risk): **0 cases** – Minimizes unnecessary stress and medical costs for patients.

## Technologies & Methodologies
- **Programming**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Techniques**: Logistic Regression, Ridge Regularization, Lasso Regularization, Elastic Net, Random Forest, GridSearchCV, K-Fold Cross-Validation, Feature Engineering, EDA, Statistical Analysis, Hyperparameter Tuning
- **Domains**: Healthcare ML, Risk Classification, Medical Diagnostics

## Real-World Applications & Impact
- **Accessible Screening**: Questionnaire-based screening deployable in remote areas without advanced medical equipment.
- **Early Detection**: Identifies high-risk patients early when treatment is most effective, improving survival rates.
- **Cost-Effective**: Reduces healthcare costs by focusing resources on high-risk patients.
- **Implementation in South Africa**: Particularly valuable in regions with high pollution, dense smoking populations, and limited healthcare access. The model runs on basic computing devices, making it suitable for rural clinics and community health centers.

## Technical Challenges & Innovative Solutions
1. **Challenge**: Baseline Model Limitations  
   **Solution**: Implemented comprehensive EDA, systematic feature selection, and advanced regularization with hyperparameter optimization.

2. **Challenge**: Critical False Negative Minimization  
   **Solution**: Developed custom evaluation metrics prioritizing false negative reduction while maintaining accuracy and precision.

3. **Challenge**: Model Generalization & Overfitting  
   **Solution**: Applied multiple cross-validation strategies, regularization techniques, and bias-variance trade-off optimization.

## Key Learning Outcomes & Insights
### Technical Skills Developed
- Advanced regularization and hyperparameter optimization
- Healthcare-specific model evaluation and risk assessment
- Comprehensive data preprocessing and feature engineering
- Cross-validation strategies for robust model validation
- Ensemble method comparison and selection criteria

### Domain Expertise Gained
- Healthcare ML ethics and critical decision-making
- Understanding lung cancer risk factors and early detection
- Healthcare accessibility challenges in developing regions
- Cost-benefit analysis in medical screening applications
- Real-world deployment considerations for healthcare AI

## Project Details
- **Completed**: April 2024
- **Institution**: Stellenbosch University
- **Course**: Data Science 316
- **Collaboration**: Developed with Andre van der Merwe
- **Type**: Team Project

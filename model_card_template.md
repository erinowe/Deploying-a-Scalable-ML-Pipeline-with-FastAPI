# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    This model is a Logistic Rregression classifier trained to predict whether an individual's oincome exceeds $50k per year based on demographic and employment features from the U.S Census dataset. The model was trained using scikit-learn's LogisticRegression implementation. 
## Intended Use
    The model is intended for educational purposes to demonstrate building, training, evaluating and deploying a machine learning pipeline using FastAPI and CI/CD pratices. it predicts category based on structured census features. 
## Training Data
    The model was trained on the Census Income dataset. The dataset includes both catgeorical and continuous featues such as: age, workclass, education, marital status, occupation, relationshp, race, sec and native country. The target variable is slary, which indicates wether income is above or below $50k. An 80/20 trained-test split was used with stratification on the salary label. 
## Evaluation Data
    The evaluation was performed on the 20% held-out test set. Performance metrics were computed on this test dataset as well as across slices of categorical features. 
## Metrics
_Please include the metrics used and your model's performance on those metrics._
    The model was evaluated using: Percision, recall and F1 score. 
    Test set Performance: Percision - 0.7159, Recall - 0.5963, F1 score - 0.6507
   Performance was also evaluated across slices of each categorical features and written to slice_output.txt 
## Ethical Considerations
    The dataset contains sensitive demographic attributes such as race, sex and nationality. Predictions from this model may reflect historical biases present in the dataset. This model should not be used in real-world decision-making related to employment, compensation or financial eligibility.
## Caveats and Recommendations
    - Logistic Regression produced a convergence warning, indicating that additional scaling or increased iteration limits could improve optimization 
    - Feature scaling may improve model performance. 
    - further fairness analysis should be conducted before any deployment beyond educational use 
# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model trained to predict whether an individual earns more than $50K per year based on demographic and employment-related features.

The model is implemented using Logistic Regression from scikit-learn. Categorical features are one-hot encoded, and the target variable is binarized prior to training.

## Intended Use
The intended use of this model is for educational and demonstration purposes only. It is designed to showcase how to build, evaluate, and deploy a machine learning pipeline using Python and FastAPI.

This model should not be used for real-world income prediction, employment decisions, credit scoring, or any high-stakes decision-making.

## Training Data
The model was trained using the UCI Census Income dataset. The dataset includes demographic and employment-related attributes such as age, education, occupation, marital status, race, sex, and hours worked per week.

The target variable indicates whether an individual's income exceeds $50K per year.

## Evaluation Data
The evaluation data consists of a held-out test split (20%) of the original census dataset. Stratified sampling was used to preserve the distribution of the target variable during the train-test split.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7343
Recall:    0.5746
F1:        0.6447
The model was evaluated using precision, recall, and F1 score.

Overall performance on the test dataset:
- Precision: 0.73
- Recall: 0.57
- F1 Score: 0.64
Additionally, performance was evaluated across categorical data slices (e.g., workclass, education) to assess model behavior across subgroups.

## Ethical Considerations
This model is trained on historical census data, which may reflect existing societal biases related to income, race, sex, and occupation. As a result, the model may reproduce or amplify these biases in its predictions.Care should be taken when interpreting model outputs, and fairness-aware evaluation should be considered if adapting this model for other purposes.

## Caveats and Recommendations
The model uses a simple Logistic Regression algorithm and does not include feature scaling or advanced hyperparameter tuning. Performance could potentially be improved by experimenting with alternative models, feature engineering, or optimization techniques.Future work could also include bias mitigation strategies and more robust evaluation across demographic subgroups

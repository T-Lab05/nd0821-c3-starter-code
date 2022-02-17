# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
TLab241 created the model as 2022/2/16. It is RandomForest classifier using the default parameters in scikit-learn.

## Intended Use
This model should be used to predict if one's salary is above $50,000 or not, based off several attributes. The users can be a sales person who want to focus on high salary comsumers.

## Training Data
The data was obtained from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 32,562 rows, and a 80-20 split was used to break this into a train and test set stratified with the target label. To use the data for training One Hot Encoder was used on the categorical features and a label binarizer was used on the labels.


## Evaluation Data
As stated above, the evaluation data was prepared by splitting the original data.

## Metrics
- Metrics on the whole test dataset
    - Precision: 0.734
    - Recall: 0.628
    - Fbeta-Score: 0.677
- Metrics on the data slice
    - Examined performances among groups based on categorical features. Result plots are preserved in 'model performance' directory. Here are some remarks on that.
        - Sex: There is little disparity between male and female.
        - Race: The model performance is lower among American Indian-Eskimo group than other groups such as White, Black and Asian.

## Ethical Considerations
The overall model performance is not so high and the performance is also low  among specific groups. Therefore, this model should not be used for critical case such as credit risk assessment. 

## Caveats and Recommendations
- Model bias may arise because sample of certain groups are relatively small and failed to catch patterns. Increasing samples for such groups could be a remedy for bias.
- The model predicts mainly on one's static attribution such as sex, race and workclass. Adding more finacial or behaviral features may contributes to improve overall model performance.
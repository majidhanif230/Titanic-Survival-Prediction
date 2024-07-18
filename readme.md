# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using a machine learning model. The dataset used for this project is the Titanic Dataset, and the model used is a Random Forest Classifier.

## Project Overview

This project was provided by Machine Learning1 Pvt Limited and completed by Majid Hanif.

## Dataset

The dataset used in this project is the Titanic Dataset, which contains information about the passengers who were on board the Titanic. The dataset includes the following columns:

- `Survived`: Survival (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Sex
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Fare`: Passenger fare
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Steps

1. **Data Loading**: Load the dataset using pandas.
2. **Data Exploration**: Display the first few rows and summary of the dataset, and check for missing values.
3. **Data Preprocessing**:
   - Fill missing values for 'Age' and 'Fare' with their median values.
   - Fill missing values for 'Embarked' with the mode.
   - Convert 'Sex' and 'Embarked' columns to numerical values.
   - Create a new feature 'FamilySize'.
   - Drop irrelevant columns.
4. **Data Visualization**:
   - Visualize survival rate by gender.
   - Visualize survival rate by Pclass.
   - Display a correlation matrix heatmap.
5. **Model Building**:
   - Split the data into training and testing sets.
   - Train a RandomForestClassifier.
   - Make predictions on the test set.
6. **Model Evaluation**:
   - Evaluate the model using accuracy score and classification report.

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas seaborn matplotlib scikit-learn
## Usage
1. Clone the repository:
    it clone https://github.com/your-username/titanic-survival-prediction.git
2. Navigate to the project directory:
    cd titanic-survival-prediction
3. Run the script:
    python titanic_survival_prediction.py
## Results:
Accuracy: 0.8547486033519553
Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.91      0.88       110
           1       0.84      0.77      0.80        69

    accuracy                           0.85       179
   macro avg       0.85      0.84      0.84       179
weighted avg       0.85      0.85      0.85       179

## Contributing:
If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License.

## Acknowledgements
Machine Learning1 Pvt Limited for providing the project.
Kaggle for the Titanic Dataset.

# Music_Genre_classification-
## Libraries Used:
### Pandas
#### Purpose: Data manipulation and analysis.
* Load the dataset from a CSV file.
* Display the first and last few rows of the dataset.
* Show the shape and column names of the dataset.
* Check for and handle missing values.
* Drop non-feature columns for model training.
### Numpy
#### Purpose: Numerical operations.
* Handle infinite values and replace them with NaN.
* Perform array operations for data preprocessing.
### Sklearn.model_selection
#### Purpose: Splitting the dataset and hyperparameter tuning.
* train_test_split: Split the data into training and testing sets.
* GridSearchCV: Perform hyperparameter tuning to find the best model parameters using cross-validation.
### Sklearn.ensemble
#### Purpose: Model training using ensemble methods.
* RandomForestClassifier: Train a Random Forest model for classification.
### Sklearn.metrics
#### Purpose: Model evaluation.
* Calculate accuracy, precision, recall, and F1-score.
* Generate a classification report.
* Create and display a confusion matrix.
### Matplotlib.pyplot
#### Purpose: Plotting and visualization.
* Visualize the distribution of genres in the dataset.
* Plot the correlation matrix.
* Visualize PCA results and confusion matrices.
### Seaborn
#### Purpose: Enhanced data visualization.
* Create heatmaps for the correlation matrix and confusion matrices.
* Enhance scatter plots for PCA visualization.
### Sklearn.decomposition
#### Purpose: Dimensionality reduction.
* PCA: Perform Principal Component Analysis to reduce the dimensionality of the feature set.
### Sklearn.preprocessing
#### Purpose: Feature scaling.
* StandardScaler: Standardize features by removing the mean and scaling to unit variance.

## Steps Performed
* Import Libraries: Imported necessary libraries for data processing, model training, evaluation, and visualization.
* Load Dataset: Loaded the dataset from a CSV file.
* Display First Few Rows: Displayed the first few rows of the dataset.
* Display Last Few Rows: Displayed the last few rows of the dataset.
* Display Dataset Shape: Displayed the shape of the dataset (number of rows and columns).
* Display Column Names: Displayed the column names of the dataset.
* Check Missing Values: Checked for missing values in each column.
* Display Dataset Information: Displayed information about the dataset including columns, non-null counts, and data types.
* Replace and Drop NaN Values: Replaced infinite values with NaN and dropped rows with NaN values.
* Drop Non-feature Columns: Dropped non-feature columns ('filename', 'length', 'label').
* Separate Features and Target: Separated features (X) and target (y).
* Visualize Genre Distribution: Visualized the distribution of genres in the dataset using a bar plot.
* Compute and Visualize Correlations: Computed and visualized correlations between features using a heatmap.
* Standard Scaling: Performed standard scaling on features.
* PCA: Performed PCA with 2 principal components.
* Visualize PCA: Visualized PCA results with colors based on labels.
* Train-Test Split: Split the data into training and testing sets.
* Initialize and Train Model: Initialized and trained the Random Forest Classifier.
* Make Predictions: Made predictions on the test set.
* Evaluate Model: Evaluated the model using accuracy, precision, recall, and F1-score.
* Classification Report: Displayed the classification report.
* Confusion Matrix: Visualized the confusion matrix.
* Hyperparameter Tuning: Used GridSearchCV to find the best hyperparameters for the Random Forest Classifier.
* Print Best Parameters: Printed the best parameters found by GridSearchCV.
* Evaluate Best Model: Evaluated the best model using accuracy, precision, recall, and F1-score.
* Classification Report for Best Model: Displayed the classification report for the best model.
* Confusion Matrix for Best Model: Visualized the confusion matrix for the best model.

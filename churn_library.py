"""
# churn_library.py
# This module contains functions for data import, EDA, feature engineering,
# model training, and evaluation for customer churn prediction.
"""


# Import libraries
import logging
import os

import sklearn.base
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'



# Set up logging
def setup_logging(
        logfile,
        formatpattern='%(asctime)s - %(levelname)s - %(message)s'):
    """
    Set up logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(formatpattern)

    # Create and configure file handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create and configure stream handler (prints to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Attach the handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging(logfile='./logs/pipeline_results.log')


def import_data(pth: str) -> pd.DataFrame:
    '''
    Returns dataframe for the csv found at pth

    Input:
            pth: a path to the csv
    Output:
            df: pandas dataframe
    '''
    # Read csv file
    logger.info("\nImporting data from %s", pth)
    if not os.path.exists(pth):
        logger.error("File not found: %s", pth)
        raise FileNotFoundError("The file at %s was not found.", pth)
    df = pd.read_csv(pth)
    logger.info("Data imported successfully with shape: %s", df.shape)

    # Display
    logger.info("Displaying first few rows of the dataframe.")
    logger.info("\n%s", df.head())

    logger.info("Data import complete.")
    return df


def create_target(df: pd.DataFrame, target: str = 'Churn') -> pd.DataFrame:
    '''
    Create target column 'Churn' in the dataframe based on 'Attrition_Flag'.

    Input:
            df: pandas dataframe with 'Attrition_Flag' column
            target: name of the target column to be created

    Output:
            df: pandas dataframe with target column added
    '''
    logger.info(
        "Creating target column [%s] based on 'Attrition_Flag'.", target)

    # Create target column based on 'Attrition_Flag'
    df[target] = df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)

    logger.info(
        f"Target column [{target}] created successfully with unique values: {df[target].unique()}")

    return df


def create_target_dist_plot(df: pd.DataFrame,
                            target: str = 'Churn',
                            output_dir: str = f'./images/') -> None:
    '''
    Create a bar plot for the target distribution.

    Input:
            df: pandas dataframe with target column
            target: name of the target column

    Output:
            None
    '''
    logger.info("Plotting target distribution.")
    filename = f'{target}_histogram.png'
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 5))
    df[target].hist()
    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()

    logger.info(
        "Target distribution plot for [%s] saved in %s.", target, output_path)


def create_bar_plot(df: pd.DataFrame,
                    column: str,
                    output_dir: str = f'./images/') -> None:
    '''
    Helper function to create bar plots for categorical features.
    Input:
        df: pandas dataframe
        column: str name of the categorical column to plot
    Output:
        None
    '''
    logger.info("Creating bar plot for feature: %s", column)
    filename = f'cat_{column}_bar_plot.png'
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(20, 10))
    df[column]\
        .value_counts(normalize=True)\
        .plot(kind='bar', title=column)
    plt.savefig(output_path)
    plt.close()
    logger.info("Bar plot for [%s] saved in [%s].", column, output_path)


def create_histogram(df: pd.DataFrame,
                     column: str,
                     output_dir: str = f'./images/') -> None:
    '''
    Helper function to create histograms for quantitative features.
    Input:
            df: pandas dataframe
            column: name of the quantitative column to plot
    Output:
            None
    '''
    logger.info("Creating histogram for feature: %s", column)
    filename = f'quant_{column}_histogram.png'
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(20, 10))
    df[column].hist()
    plt.title(column)
    plt.savefig(output_path)
    plt.close()
    logger.info("Histogram for [%s] saved in [%s].", column, output_path)


def create_density_plot(df: pd.DataFrame,
                        column: str,
                        output_dir: str = f'./images/') -> None:
    '''
    Helper function to create density plots for quantitative features.
    Input:
            df: pandas dataframe
            column: name of the quantitative column to plot
    Output:
            None
    '''
    logger.info("Plotting density for: %s", column)
    filename = f'quant_{column}_density_plot.png'
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(20, 10))
    sns.histplot(df[column], stat='density', kde=True)
    plt.title(f'Density Plot for {column}')
    plt.savefig(output_path)
    plt.close()
    logger.info("Density plot for [%s] saved in [%s].", column, output_path)


def create_heatmap(df: pd.DataFrame,
                   output_dir: str = f'./images/') -> None:
    '''
    Helper function to create a heatmap for the correlation matrix of the dataframe.
    Input:
            df: pandas dataframe
    Output:
            None
    '''
    # Define paths
    logger.info("Plotting correlation heatmap.")
    filename = 'correlation_heatmap.png'
    output_path = os.path.join(output_dir, filename)

    # Select data: numeric columns only
    numeric_df = df.select_dtypes(include=['number'])

    # Plot the heatmap
    plt.figure(figsize=(20, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    logger.info("Correlation heatmap saved in [%s].", output_path)


def perform_eda(df: pd.DataFrame,
                cat_columns: list,
                quant_columns: list,
                target: str = 'Churn',
                output_dir: str = f'./images/') -> None:
    '''
    Perform EDA on df and save figures to images folder
    Input:
            df: pandas dataframe
    Output:
            None
    '''
    logger.info("\nPerforming EDA on the dataframe.")
    logger.info("DataFrame shape: %s", df.shape)

    # Missing values
    logger.info("Checking for missing values in the dataframe:")
    missing_values = df.isnull().sum()
    logger.info("\n%s", missing_values)

    # Descriptive statistics
    logger.info("Generating descriptive statistics for the dataframe:")
    logger.info("\n%s", df.describe())

    # Plotting target distribution
    logger.info("Plotting target distribution.")
    create_target_dist_plot(df, target, output_dir)

    # Bar plot categorical features
    logger.info("Plotting categorical features.")
    for column in cat_columns:
        create_bar_plot(df, column, output_dir)

    # Histogram for quantitative features
    logger.info("Plotting quantitative features.")
    for column in quant_columns:
        create_histogram(df, column, output_dir)

    # Density plot for quantitative features
    logger.info("Plotting density plots for quantitative features.")
    for column in quant_columns:
        create_density_plot(df, column, output_dir)

    # Correlation heatmap
    logger.info("Creating correlation heatmap.")
    create_heatmap(df, output_dir)

    logger.info("EDA performed successfully.")


def encoder_helper(df: pd.DataFrame,
                   category_lst: list,
                   target: str = 'Churn',
                   response: str = 'Churn') -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    Input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: name of the target column (default is 'Churn')
            response: string of response name\
                [optional argument that could be used for naming variables or index y column]

    Output:
            df: pandas dataframe with new columns for
    '''
    logger.info("Encoding categorical features with churn proportions.")

    # Use response name if provided, else fallback to target
    col_suffix = response if response else target.lower()
    logger.info("Using '%s' as column suffix for encoding.", col_suffix)

    for category in category_lst:
        logger.info("Encoding category: %s", category)

        # Create a new column with the churn proportions
        new_col_name = f"{category}_{col_suffix}"
        logger.info("Creating new column: %s", new_col_name)
        df[new_col_name] = df\
            .groupby(category)[target]\
            .transform('mean')
        logger.info("Encoded %s with churn proportions in column: %s", category, new_col_name)

    return df


def perform_feature_engineering(df: pd.DataFrame,
                                keep_cols: list,
                                target: str = 'Churn') -> None:
    '''
    Input:
              df: pandas dataframe
              keep_cols: list of columns to keep in the feature set


    Output:
              X_train: numpy array X training data
              X_test: numpy array X testing data
              y_train: numpy array y training data
              y_test: numpy array y testing data
    '''
    logger.info("Performing feature engineering.")

    # Separate features and target
    logger.info("Separating features and target: %s", target)
    X = df[keep_cols].values
    y = df[target].values

    logger.info("Features shape: %s, Target shape: %s", X.shape, y.shape)
    logger.info("Features preview:\n%s", X[0:5])

    # Split the data into training and testing sets
    X_train, \
        X_test, \
        y_train, \
        y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    logger.info(f"Data split into training and testing sets:\n"
                f"X_train shape: {X_train.shape},\
                        X_test shape: {X_test.shape}, "
                f"y_train shape: {y_train.shape},\
                        y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: np.ndarray,
                                y_test: np.ndarray,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray,
                                output_dir: str = './images/') -> None:
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder
    Input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Output:
             None
    '''
    logger.info(
        "Generating classification reports for training and testing data.")

    # Random Forest Classifier
    # ----------------------------------------------
    logger.info("Random Forest Classifier")

    filename_rf = 'rfc_classification_report.png'
    save_path_rf = os.path.join(output_dir, filename_rf)

    # Generate classification report
    logger.info(
        "Generating and saving classification report for Random Forest Classifier.")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1.3)

    plt.savefig(save_path_rf)
    plt.close()

    # Logistic Regression Classifier
    # ----------------------------------------------
    logger.info("Logistic Regression Classifier")

    filename_lr = 'lc_classification_report.png'
    save_path_lr = os.path.join(output_dir, filename_lr)

    # Generate and save classification report as an image
    logger.info(
        "Saving classification report for Logistic Regression Classifier.")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1.3)

    plt.savefig(save_path_lr)
    plt.close()


def feature_importance_plot(model: sklearn.base.BaseEstimator,
                            feature_names: list,
                            output_dir: str = './images/') -> None:
    '''
    Creates and stores the feature importances in pth
    Input:
            model: model object containing feature_importances_
            feature_names: list of feature names
            output_pth: path to store the figure

    Output:
             None
    '''
    filename = 'feature_importance_rf.png'
    output_pth = os.path.join(output_dir, filename)

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    # names = [X_df.columns[i] for i in indices]
    names = [feature_names[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    # plt.bar(range(X_df.shape[1]), importances[indices])
    plt.bar(range(len(feature_names)), importances[indices])

    # Add feature names as x-axis labels
    # plt.xticks(range(X_df.shape[1]), names, rotation=90)
    plt.xticks(range(len(feature_names)), names, rotation=90)

    # save the plot
    logger.info("Saving feature importance plot to %s", output_pth)
    plt.savefig(output_pth)
    plt.close()


def plot_roc_curve(model1,
                   X,
                   y,
                   model2=None,
                   alpha=0.8,
                   ax=None,
                   output_dir: str = './images/') -> None:
    '''
    Plots the ROC curve for a given model and dataset.
    Input:
            model1: trained model
            model2: second trained model (optional)
            X: feature data
            y: target data
            ax: matplotlib axis object (optional)
    Output:
            None
    '''

    model1_name = model1.__class__.__name__.lower()
    if model2 is None:
        logger.info(
            f"Plotting ROC curve of model: {model1.__class__.__name__}")
        filename = f'{model1_name}_roc_curve.png'
        output_path = os.path.join(output_dir, filename)

        plt.figure(figsize=(10, 5))
        RocCurveDisplay.from_estimator(model1, X, y, alpha=alpha).plot()
        plt.title(f"ROC Curve - {model1_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"ROC curve for {model1_name} saved in {output_path}.")
    else:
        model1_name = model1.__class__.__name__.lower()
        model2_name = model2.__class__.__name__.lower()

        logger.info(
            f"Plotting ROC curves of models: {model1_name} and {model2_name}")
        filename = f'{model1_name}_{model2_name}_roc_curve.png'
        output_path = os.path.join(output_dir, filename)

        m1_plot = RocCurveDisplay\
            .from_estimator(
                model1, X, y,
                ax=ax, alpha=alpha
            ).plot()

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        m2_plot = RocCurveDisplay\
            .from_estimator(
                model2, X, y,
                ax=ax, alpha=alpha,
                label=model2_name
            )\
            .plot()
        m1_plot.plot(ax=ax, alpha=alpha, label=model1_name)
        plt.title(f"ROC Curve - {model1_name} vs {model2_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(output_path)
        plt.close()
        logger.info(
            f"ROC curve for {model1_name} and {model2_name} saved in {output_path}.")


def shap_plot(model: sklearn.base.BaseEstimator,
              X: np.ndarray,
              feature_names: list,
              output_dir: str = './images/') -> None:
    '''
    Creates a SHAP summary plot for the given model and data.
    Input:
        model: trained model
        X: feature data
        feature_names: list of feature names
        output_dir: directory to save the SHAP summary plot
    Output:
        None
    '''
    filename = 'shap_summary_plot_rf.png'
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 5))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(
        shap_values[:, :, 1],
        X,
        feature_names,
        plot_type="bar"
    )
    plt.savefig(output_path)
    plt.close('all')
    logger.info(
        "SHAP summary plot for Random Forest Classifier saved in %s.", output_path)


def train_models(X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 feature_names: list = None,
                 output_dir_images: str = './images/',
                 output_dir_models: str = './models/') -> None:
    '''
    Train, store model results: images + scores, and store models
    Input:
              X_train: numpy arrary X training data
              X_test: numpy array X testing data
              y_train: numpy array y training data
              y_test: numpy array y testing data
    Output:
              None
    '''
    logger.info("Training models on the training data.")

    # Grid search parameters
    logger.info("Defining parameter grid for Grid Search.")
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    logger.info("Parameter grid: %s", param_grid)

    # Fit the models
    # ---------------------------------------------
    logger.info("Initializing and fitting models.")

    # Random Forest Classifier
    logger.info("Initializing Random Forest Classifier.")
    rfc = RandomForestClassifier(random_state=42)

    # Fit the Random Forest model
    logger.info("Fitting Random Forest Classifier.")
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5
    )
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression
    logger.info("Initializing Logistic Regression.")
    lrc = LogisticRegression(solver='lbfgs', max_iter=5000)

    # Fit the Logistic Regression model
    logger.info("Fitting Logistic Regression.")
    lrc.fit(X_train, y_train)

    # Store the models
    # ----------------------------------------------
    logger.info("Storing trained models in the models directory.")

    filename_rc = 'rfc_model.pkl'
    path_rc = os.path.join(output_dir_models, filename_rc)

    joblib.dump(cv_rfc.best_estimator_, path_rc)
    logger.info("Random Forest Classifier model stored as '%s'", path_rc)

    filename_lr = 'logistic_model.pkl'
    path_lr = os.path.join(output_dir_models, filename_lr)

    joblib.dump(lrc, path_lr)
    logger.info("Logistic Regression model stored as '%s'", path_lr)

    # Predict on training and testing data
    # ----------------------------------------------
    logger.info("Making predictions with trained models.")

    # Predict on training and testing data with Random Forest Classifier and
    # Logistic Regression
    logger.info("Making predictions with Random Forest Classifier.")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Predict on training and testing data with Logistic Regression
    logger.info("Making predictions with Logistic Regression.")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Classification report
    # ----------------------------------------------
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_dir_images
                                )

    # ROC curve
    # ----------------------------------------------
    # Plot ROC curve for Random Forest Classifer
    logger.info("Plotting ROC curve for Random Forest Classifier.")
    plot_roc_curve(
        model1=cv_rfc.best_estimator_,
        X=X_test,
        y=y_test,
        output_dir=output_dir_images
    )

    # Plot ROC curve for Logistic Regression
    logger.info("Plotting ROC curve for Logistic Regression.")
    plot_roc_curve(
        model1=lrc,
        X=X_test,
        y=y_test,
        output_dir=output_dir_images
    )

    # Combined ROC curve
    logger.info("Plotting combined ROC curve for both models.")
    plot_roc_curve(
        model1=cv_rfc.best_estimator_,
        X=X_test,
        y=y_test,
        model2=lrc,
        output_dir=output_dir_images
    )

    # Detailed Analyses for Random Forest Classifier
    # ----------------------------------------------
    logger.info("Detailed analyses for Random Forest Classifier.")

    # Tree Explainer
    logger.info("Using SHAP Tree Explainer for Random Forest Classifier.")
    shap_plot(
        cv_rfc.best_estimator_,
        X_test,
        feature_names,
        output_dir_images
    )

    # Feature importance plot
    logger.info("Creating feature importance plot for Random Forest Classifier.")
    feature_importance_plot(
        cv_rfc.best_estimator_,
        feature_names,
        output_dir_images
    )


if __name__ == "__main__":
    # Define variables
    target = 'Churn'
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    features = ['Customer_Age',
                'Dependent_count',
                'Months_on_book',
                'Total_Relationship_Count',
                'Months_Inactive_12_mon',
                'Contacts_Count_12_mon',
                'Credit_Limit',
                'Total_Revolving_Bal',
                'Avg_Open_To_Buy',
                'Total_Amt_Chng_Q4_Q1',
                'Total_Trans_Amt',
                'Total_Trans_Ct',
                'Total_Ct_Chng_Q4_Q1',
                'Avg_Utilization_Ratio',
                'Gender_Churn',
                'Education_Level_Churn',
                'Marital_Status_Churn',
                'Income_Category_Churn',
                'Card_Category_Churn']

    # Import data
    data_path = 'data/bank_data.csv'
    df = import_data(data_path)

    # Create target column
    df = create_target(df=df, target='Churn')

    # Perform EDA
    perform_eda(
        df=df,
        cat_columns=cat_columns,
        quant_columns=quant_columns
    )

    # Encode categorical features
    df = encoder_helper(
        df=df,
        category_lst=cat_columns,
        target=target,
        response=target
    )

    # Perform feature engineering
    X_train, \
        X_test, \
        y_train, \
        y_test = perform_feature_engineering(
            df,
            keep_cols=features,
            target=target
        )

    # Train models
    train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=features
    )

    logger.info("Churn prediction pipeline completed successfully.")

"""
# churn_library.py
# This module contains functions for data import, EDA, feature engineering,
# model training, and evaluation for customer churn prediction.
"""


# Import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import logging
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report



# Set up logging
logging.basicConfig(
    filename='./logs/test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth: str) -> pd.DataFrame:
    '''
    Returns dataframe for the csv found at pth

    Input:
            pth: a path to the csv
    Output:
            df: pandas dataframe
    '''	    
    # Read csv file
    logging.info(f"\nImporting data from {pth}")
    if not os.path.exists(pth):
        logging.error(f"File not found: {pth}")
        raise FileNotFoundError(f"The file at {pth} was not found.")
    df = pd.read_csv(pth)
    logging.info(f"Data imported successfully with shape: {df.shape}")

    # Display
    logging.info("Displaying first few rows of the dataframe.")
    logging.info(f"\n{df.head()}")

    logging.info("Data import complete.")
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
    logging.info(f"Creating target column [{target}] based on 'Attrition_Flag'.")
        
    # Create target column based on 'Attrition_Flag'
    df[target] = df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)

    logging.info(f"Target column [{target}] created successfully with unique values: {df[target].unique()}")
    
    return df

def create_target_dist_plot(df: pd.DataFrame, target: str = 'Churn') -> None:
    '''
    Create a bar plot for the target distribution.

    Input:
            df: pandas dataframe with target column
            target: name of the target column

    Output:
            None
    '''
    logging.info("Plotting target distribution.")
    output_path = f'./images/{target}_histogram.png'
    
    
    plt.figure(figsize=(10, 5))
    df[target].hist()
    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"Target distribution plot for [{target}] saved in {output_path}.")


def create_bar_plot(df: pd.DataFrame, column: str) -> None:
    '''
    Helper function to create bar plots for categorical features.
    Input:
        df: pandas dataframe
        column: name of the categorical column to plot
    Output:
        None
    '''
    logging.info(f"Creating bar plot for feature: {column}")
    output_path = f'./images/cat_{column}_bar_plot.png'

    plt.figure(figsize=(20, 10))
    df[column]\
        .value_counts('normalize')\
        .plot(kind='bar', title=column)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Bar plot for {column} saved in {output_path}.")


def create_histogram(df: pd.DataFrame, column: str) -> None:
        '''
        Helper function to create histograms for quantitative features.
        Input:
                df: pandas dataframe
                column: name of the quantitative column to plot
        Output:
                None
        '''
        logging.info(f"Creating histogram for feature: {column}")
        output_path = f'./images/quant_{column}_histogram.png'

        plt.figure(figsize=(20, 10))
        df[column].hist()        
        plt.title(column)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Histogram for {column} saved in {output_path}.")


def create_density_plot(df: pd.DataFrame, column: str) -> None:
        '''
        Helper function to create density plots for quantitative features.
        Input:
                df: pandas dataframe
                column: name of the quantitative column to plot
        Output:
                None
        '''
        logging.info(f"Plotting density for: {column}")
        output_path = f'./images/quant_{column}_density_plot.png'

        plt.figure(figsize=(20, 10))
        sns.histplot(df[column], stat='density', kde=True)
        plt.title(f'Density Plot for {column}')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Density plot for {column} saved in {output_path}.")


def create_heatmap(df: pd.DataFrame) -> None:
        '''
        Helper function to create a heatmap for the correlation matrix of the dataframe.
        Input:
                df: pandas dataframe
        Output:
                None
        '''
        logging.info("Plotting correlation heatmap.")
        output_path = './images/correlation_heatmap.png'

        plt.figure(figsize=(20, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
        plt.title("Correlation Heatmap")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Correlation heatmap saved in {output_path}.")


def perform_eda(df: pd.DataFrame,                
                cat_columns: list,
                quant_columns: list,
                target: str = 'Churn') -> None:
    '''
    Perform EDA on df and save figures to images folder
    Input:
            df: pandas dataframe
    Output:
            None
    '''
    logging.info("\nPerforming EDA on the dataframe.")
    logging.info(f"DataFrame shape: {df.shape}")

    # Missing values
    logging.info("Checking for missing values in the dataframe:")
    missing_values = df.isnull().sum()
    logging.info(f"\n{missing_values}")

    # Descriptive statistics
    logging.info("Generating descriptive statistics for the dataframe:")
    logging.info(f"\n{df.describe()}")

    # Plotting target distribution
    logging.info("Plotting target distribution.")
    create_target_dist_plot(df, target)

    # Bar plot categorical features
    logging.info("Plotting categorical features.")
    for column in cat_columns:        
        create_bar_plot(df, column)      

    # Histogram for quantitative features
    logging.info("Plotting quantitative features.")
    for column in quant_columns:
        create_histogram(df, column)        

    # Density plot for quantitative features
    logging.info("Plotting density plots for quantitative features.")
    for column in quant_columns:
        create_density_plot(df, column)

    # Correlation heatmap
    logging.info("Creating correlation heatmap.")
    create_heatmap(df)

    logging.info("EDA performed successfully.")
    
    


def encoder_helper(df: pd.DataFrame,\
                   category_lst: list,\
                   target: str='Churn',\
                   response: str='Churn') -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    Input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: name of the target column (default is 'Churn')
            response: string of response name [optional argument that could be used for naming variables or index y column]

    Output:
            df: pandas dataframe with new columns for
    '''
    logging.info("Encoding categorical features with churn proportions.")

    # Use response name if provided, else fallback to target
    col_suffix = response if response else target.lower()
    logging.info(f"Using '{col_suffix}' as column suffix for encoding.")


    for category in category_lst:
            logging.info(f"Encoding category: {category}")             
            
            # Calculate the proportion of churn for each category
            churn_proportions = df.groupby(category).mean()[target]
            
            # Create a new column with the churn proportions
            new_col_name = f"{category}_{col_suffix}"
            logging.info(f"Creating new column: {new_col_name}")
            df[new_col_name] = df[category].map(churn_proportions)
            logging.info(f"Encoded {category} with churn proportions\
                         in column: {new_col_name}")


def perform_feature_engineering(df: pd.DataFrame,\
                                keep_cols: list,\
                                target: str='Churn') -> None:
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
    logging.info("Performing feature engineering.")

    # Separate features and target
    logging.info(f"Separating features and target: {target}")
    X = df[keep_cols].values
    y = df[target].values

    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logging.info(f"Features preview:\n{X.head()}")

    # Split the data into training and testing sets        
    X_train,\
    X_test,\
    y_train,\
    y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
    )

    logging.info(f"Data split into training and testing sets:\n"
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
                                y_test_preds_rf: np.ndarray) -> None:
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
    logging.info("Generating classification reports for training and testing data.")
    
    # Random Forest Classifier
    # ----------------------------------------------
    logging.info("Random Forest Classifier")

    # Generate classification report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')    
    plt.axis('off')

    # Save the classification report as an image
    logging.info("Saving classification report for Random Forest Classifier.")
    save_path_rf = './images/rfc_classification_report.png'
    plt.savefig(save_path_rf)
    plt.close()

    # Logistic Regression Classifier
    # ----------------------------------------------    
    logging.info("Logistic Regression Classifier")
    
    # Generate classification report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')

    # Save the classification report as an image
    logging.info("Saving classification report for Logistic Regression Classifier.")
    save_path_lr = './images/lrc_classification_report.png'
    plt.savefig(save_path_lr)
    plt.close()
    


def feature_importance_plot(model: sklearn.base.BaseEstimator,\
                            X_data: pd.DataFrame,\
                            output_pth: str='./images/feature_importance.png') -> None:
    '''
    Creates and stores the feature importances in pth
    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    Output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # save the plot
    logging.info(f"Saving feature importance plot to {output_pth}")
    plt.savefig(output_pth)
    plt.close()


def plot_roc_curve(model1, X, y, model2=None,alpha=0.8):
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
                logging.info(f"Plotting ROC curve of model: {model1.__class__.__name__}")
                output_path = f'./images/{model1_name}_roc_curve.png'
        
                plt.figure(figsize=(10, 5))        
                plot_roc_curve(model1, X, y, alpha=alpha)
                plt.title(f"ROC Curve - {model1_name}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.savefig(output_path)
                plt.close()
                logging.info(f"ROC curve for {model1_name} saved in {output_path}.")
        else:                
                model2_name = model2.__class__.__name__.lower()
                logging.info(f"Plotting ROC curves of models: {model1} and {model2}")
                output_path = f'./images/{model1_name}_{model2_name}_roc_curve.png'

                plt.figure(figsize=(15, 8))
                ax = plt.gca()
                model1_plot = plot_roc_curve(model1, X, y, ax=ax, alpha=alpha)
                model2_plot = plot_roc_curve(model2, X, y, ax=ax, alpha=alpha)
                plt.title(f"ROC Curve - {model1_name} vs {model2_name}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.savefig(output_path)
                plt.close()
                logging.info(f"ROC curve for {model1_name} and {model2_name} saved in {output_path}.")  

def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
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
    logging.info("Training models on the training data.")
    
    # Grid search parameters
    logging.info("Defining parameter grid for Grid Search.")
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    logging.info(f"Parameter grid: {param_grid}")

    # Fit the models
    #---------------------------------------------
    logging.info("Initializing and fitting models.")

    # Random Forest Classifier
    logging.info("Initializing Random Forest Classifier.")
    rfc = RandomForestClassifier(random_state=42)

    # Fit the Random Forest model
    logging.info("Fitting Random Forest Classifier.")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression
    logging.info("Initializing Logistic Regression.")
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Fit the Logistic Regression model
    logging.info("Fitting Logistic Regression.")
    lrc.fit(X_train, y_train)


    # Store the models
    # ----------------------------------------------
    logging.info("Storing trained models in the models directory.")
    
    path_rc = './models/rfc_model.pkl'
    joblib.dump(cv_rfc.best_estimator_, path_rc)
    logging.info(f"Random Forest Classifier model stored as '{path_rc}'")
    
    path_lr = './models/logistic_model.pkl'
    joblib.dump(lrc, path_lr)
    logging.info(f"Logistic Regression model stored as '{path_lr}'")


    # Predict on training and testing data
    # ----------------------------------------------
    logging.info("Making predictions with trained models.")
    
    # Predict on training and testing data with Random Forest Classifier and Logistic Regression
    logging.info("Making predictions with Random Forest Classifier.")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Predict on training and testing data with Logistic Regression
    logging.info("Making predictions with Logistic Regression.")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    


    # Classification report
    # ----------------------------------------------    
    classification_report_image(y_train,\
        y_test,\
        y_train_preds_lr,\
        y_train_preds_rf,\
        y_test_preds_lr,\
        y_test_preds_rf)
    
    # ROC curve
    # ----------------------------------------------   

    # Plot ROC curve for Random Forest Classifer
    logging.info("Plotting ROC curve for Random Forest Classifier.")
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test)    

    # Plot ROC curve for Logistic Regression
    logging.info("Plotting ROC curve for Logistic Regression.")
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test)    

    # Combined ROC curve
    logging.info("Plotting combined ROC curve for both models.")
    plot_roc_curve(model1=cv_rfc.best_estimator_,\
                   X=X_test,\
                   y=y_test,\
                   model2=lrc)
    

    # Detailed Analyses for Random Forest Classifier
    # ----------------------------------------------
    logging.info("Detailed analyses for Random Forest Classifier.")
    
    # Tree Explainer
    logging.info("Using SHAP Tree Explainer for Random Forest Classifier.")
    plt.rc('figure', figsize=(10, 5))
    
    # Create SHAP explainer
    def shap_plot(model, X, output_path='./images/shap_summary_plot_rf.png'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.savefig(output_path)        
        plt.close()
        logging.info(f"SHAP summary plot for Random Forest Classifier saved in {shap_output_path}.")

        return shap_values


    logging.info("Calculating SHAP values for Random Forest Classifier.")    
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    

    # Feature importance plot    
    logging.info("Creating feature importance plot for Random Forest Classifier.")
    output_rf = './images/feature_importance_rf.png'
    feature_importance_plot(cv_rfc.best_estimator_, X_train, output_rf)
    

if __name__ == "__main__":
    # Define variables
    target = 'Churn'
    cat_columns=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
    ],
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
    data_path = 'data/churn_data.csv'
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
    df=encoder_helper(df=df,\
                      category_lst=cat_columns,\
                      target=target,\
                      response=target)

    # Perform feature engineering
    X_train,\
    X_test,\
    y_train,\
    y_test = perform_feature_engineering(df,
        keep_cols=features,
        response=target,
        target=target
)
    
    # Train models
    train_models(X_train, X_test, y_train, y_test)
    logging.info("Churn prediction pipeline completed successfully.")
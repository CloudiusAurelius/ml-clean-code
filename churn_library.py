"""
# churn_library.py
# This module contains functions for data import, EDA, feature engineering,
# model training, and evaluation for customer churn prediction.
"""


# Import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import logging

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
    pass


def feature_importance_plot(model: sk.base.BaseEstimator, X_data: pd.DataFrame, output_pth: str) -> None:
    '''
    Creates and stores the feature importances in pth
    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    Output:
             None
    '''
    pass






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
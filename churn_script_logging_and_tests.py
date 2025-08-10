"""
# churn_script_logging_and_tests.py
# Churn Prediction Model Testing
# This script contains unit tests for the churn prediction model library.
# It tests various functions including data import, feature engineering,
# EDA, model training, and visualization functions.
"""

import os
import logging
import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from churn_library import import_data,\
						  create_target,\
						  create_target_dist_plot,\
						  create_bar_plot,\
						  create_histogram,\
						  create_density_plot,\
						  create_heatmap,\
						  perform_eda,\
						  encoder_helper,\
                          perform_feature_engineering,\
                          classification_report_image,\
                          feature_importance_plot,\
                          plot_roc_curve,\
                          shap_plot,\
                          train_models


# Set up logging
def setup_logging(
        filename,
        format='%(asctime)s - %(levelname)s - %(message)s'):
    """
    Set up logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Adding a file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format))

    # Attach the handler to the logger
    logger.addHandler(file_handler)

    return logger

logger = setup_logging(filename='./logs/churn_libary_tests.log')


# Define variables
global output_dir
output_dir = './tests/images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

global target
target = 'Churn'

global cat_columns, quant_columns, features
cat_columns=[
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


def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	logger.info("Testing import_data function")
	try:
		df = import_data("./data/bank_data.csv")
		logger.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logger.error("Testing import_eda: The file wasn't found")
		raise err
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logger.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err
	



@pytest.fixture(scope='module')
def import_dataframe_fixture():
	'''
	Provides a pandas DataFrame for testing.

	Returns
	-------
	pandas.DataFrame
		The imported bank data as a DataFrame.
	'''
	df = import_data("./data/bank_data.csv")
	return df


def test_create_target(import_dataframe_fixture):
	'''
	test create_target function
	'''
	logger.info("Testing create_target function")
	df = import_dataframe_fixture

	if df is None:
		logger.error("Testing create_target: DataFrame is None")
		raise ValueError("DataFrame is None")	
	try:
		df = create_target(df)
		logger.info("Testing create_target: SUCCESS")
	except FileNotFoundError as err:
		logger.error("Testing create_target: The file wasn't found")
		raise err
	try:
		assert 'Churn' in df.columns
	except AssertionError as err:
		logger.error("Testing create_target: The target column 'churn' was not created")
		raise err
	

def test_create_target_dist_plot(import_dataframe_fixture):
	'''
	test create_target_dist_plot function
	'''
	logger.info("Testing create_target_dist_plot function")
	df = import_dataframe_fixture	
	filename = f'{target}_histogram.png'
	test_path = os.path.join(output_dir, filename)

	try:
		create_target_dist_plot(df, target, output_dir)
		logger.info("Testing create_target_dist_plot: SUCCESS")
	except Exception as err:
		logger.error(f"Testing create_target_dist_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logger.error(f"Testing create_target_dist_plot: The plot was not saved to the specified path")
		raise err



def test_create_bar_plot(import_dataframe_fixture):
	'''
	test create_bar_plot function
	'''
	logger.info("Testing create_bar_plot function")
	df = import_dataframe_fixture
	test_column = cat_columns[0]  # Use the first categorical column for testing
	
	filename = f'cat_{test_column}_bar_plot.png'
	test_path = os.path.join(output_dir, filename)
	try:
		assert isinstance(test_column, str)
	except AssertionError as err:
		logger.error(f"Testing create_bar_plot: The column name is not a string")
		raise err
	try:
		create_bar_plot(df, test_column, output_dir)
		logger.info("Testing create_bar_plot: SUCCESS")
	except Exception as err:
		logger.error(f"Testing create_bar_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logger.error(f"Testing create_bar_plot: The plot was not saved to the specified path")
		raise err


def test_create_histogram(import_dataframe_fixture):
	'''
	test create_histogram function
	'''
	logger.info("Testing create_histogram function")
	df = import_dataframe_fixture
	test_column = quant_columns[0]  # Use the first quantitative column for testing
    
	filename = f'quant_{test_column}_histogram.png'
	test_path = os.path.join(output_dir, filename)
	
	try:
		create_histogram(df, test_column, output_dir)
		logger.info("Testing create_histogram: SUCCESS")
	except Exception as err:
		logger.error(f"Testing create_histogram: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logger.error(f"Testing create_histogram: The plot was not saved to the specified path")
		raise err
	

def test_create_density_plot(import_dataframe_fixture):
	'''
	test create_density_plot function
	'''
	logger.info("Testing create_density_plot function")
	df = import_dataframe_fixture
	test_column = quant_columns[0]  # Use the first quantitative column for testing
	
	filename = f'quant_{test_column}_density_plot.png'
	test_path = os.path.join(output_dir, filename)
	
	try:
		create_density_plot(df, test_column, output_dir)
		logger.info("Testing create_density_plot: SUCCESS")
	except Exception as err:
		logger.error(f"Testing create_density_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logger.error(f"Testing create_density_plot: The plot was not saved to the specified path")
		raise err

def test_create_heatmap(import_dataframe_fixture):
	'''
	test create_heatmap function
	'''
	logger.info("Testing create_heatmap function")
	df = import_dataframe_fixture
	filename = 'correlation_heatmap.png'
	test_path = os.path.join(output_dir, filename)

	try:
		create_heatmap(df, output_dir)
		logger.info("Testing create_heatmap: SUCCESS")
	except Exception as err:
		logger.error(f"Testing create_heatmap: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logger.error(f"Testing create_heatmap: The heatmap was not saved to the specified path")
		raise err


def test_perform_eda(import_dataframe_fixture):
	'''
	test perform eda function
	'''
	logger.info("Testing perform_eda function")
	df = import_dataframe_fixture
	output_dir_eda = './tests/images/eda_images/'
	if not os.path.exists(output_dir_eda):
		os.makedirs(output_dir_eda)

	try:
		perform_eda(df,
			        cat_columns,
					quant_columns,
					target,
					output_dir_eda)
		success = 1
		logger.info("Testing perform_eda: SUCCESS")
	except Exception as err:
		logger.error(f"Testing perform_eda: {err}")
		raise err
	try:
		assert success == 1
	except AssertionError as err:
		logger.error(f"Testing perform_eda: The EDA function did not complete successfully")
		raise err
	try:
		# check if *.png files exist in the output directory
		for column in cat_columns:
			filename = f'cat_{column}_bar_plot.png'
			test_path = os.path.join(output_dir_eda, filename)
			assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir_eda}"
		for column in quant_columns:
			filename = f'quant_{column}_histogram.png'
			test_path = os.path.join(output_dir_eda, filename)
			assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir_eda}"
			filename = f'quant_{column}_density_plot.png'
			test_path = os.path.join(output_dir_eda, filename)
			assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir_eda}"
		filename = 'correlation_heatmap.png'
		test_path = os.path.join(output_dir_eda, filename)
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir_eda}"
	except AssertionError as err:
		logger.error(f"Testing perform_eda: {err}")
		raise err
	

def test_encoder_helper(import_dataframe_fixture):
	'''
	test encoder helper
	'''
	logger.info("Testing encoder_helper function")
	df = import_dataframe_fixture
	try:
		encoded_df = encoder_helper(df, cat_columns, target)
		logger.info("Testing encoder_helper: SUCCESS")
	except Exception as err:
		logger.error(f"Testing encoder_helper: {err}")
		raise err
	try:
		assert all(f"{col}_Churn" in encoded_df.columns for col in cat_columns),\
			"Not all categorical columns were encoded correctly"
	except AssertionError as err:
		logger.error(f"Testing encoder_helper: {err}")
		raise err
	


def test_perform_feature_engineering(import_dataframe_fixture):
	'''
	test perform_feature_engineering
	'''
	logger.info("Testing perform_feature_engineering function")
	df = import_dataframe_fixture
	try:
		X_train,\
		X_test,\
		y_train,\
		y_test = perform_feature_engineering(df,
			features,
			target)
		logger.info("Testing perform_feature_engineering: SUCCESS")
	except Exception as err:
		logger.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert len(X_train) > 0 and len(X_test) > 0, "Training and test sets are empty"
	except AssertionError as err:
		logger.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert len(y_train) > 0 and len(y_test) > 0, "Training and test labels are empty"
	except AssertionError as err:
		logger.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert X_train.shape[1] == len(features), "X_train does not have the correct number of features"
		assert X_test.shape[1]==X_train.shape[1], "X_train and X_test have different number of features"
		assert y_train\
			   .reshape(-1,1)\
			   .shape[1] == y_test.reshape(-1,1).shape[1], "y_train and y_test have different number of features"
	except AssertionError as err:
		logger.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert isinstance(X_train, np.ndarray), "X_train is not a numpy array"	
		assert isinstance(X_test, np.ndarray), "X_test is not a numpy array"	
		assert isinstance(y_train, np.ndarray), "y_train is not a numpy array"	
		assert isinstance(y_test, np.ndarray), "y_test is not a numpy array"
	except AssertionError as err:
		logger.error(f"Testing perform_feature_engineering: {err}")
		raise err



@pytest.fixture(scope='module')
def feature_engineering_fixture(import_dataframe_fixture):
	'''
	provides feature engineering data for testing
	'''
	df = import_dataframe_fixture	
	X_train,\
	X_test,\
	y_train,\
	y_test = perform_feature_engineering(df,
				features,
				target)
	return X_train, X_test, y_train, y_test


def test_classification_report_image(feature_engineering_fixture):
	'''
	test classification_report_image
	'''
	logger.info("Testing classification_report_image function")
	filename_rf = 'rfc_classification_report.png'
	test_path_rf = os.path.join(output_dir, filename_rf)

	_, _, y_train, y_test = feature_engineering_fixture
	y_train_preds = [0] * len(y_train)  # Dummy predictions for testing
	y_test_preds = [0] * len(y_test)    # Dummy predictions for testing
	
	try:
		classification_report_image(y_train,
							  y_test,
							  y_train_preds,
							  y_train_preds,
							  y_test_preds,
							  y_test_preds,
							  output_dir
							  )
		logger.info("Testing classification_report_image: SUCCESS")
	except Exception as err:
		logger.error(f"Testing classification_report_image: {err}")
		raise err
	try:
		assert os.path.exists(test_path_rf), f"File {filename_rf} does not exist in {output_dir}"
	except AssertionError as err:
		logger.error(f"Testing classification_report_image: {err}")
		raise err
	


def test_feature_importance_plot(feature_engineering_fixture):
	'''
	test feature_importance_plot
	'''
	logger.info("Testing feature_importance_plot function")
	filename = 'feature_importance_rf.png'
	test_path = os.path.join(output_dir, filename)

	X_train, _, y_train, _ = feature_engineering_fixture	

	rfc = RandomForestClassifier(random_state=42)
	rfc.fit(X_train, y_train)
		
	try:
		feature_importance_plot(
			rfc,
			features,
			output_dir
		)
		logger.info("Testing feature_importance_plot: SUCCESS")
	except Exception as err:
		logger.error(f"Testing feature_importance_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logger.error(f"Testing feature_importance_plot: {err}")
		raise err
	


def test_plot_roc_curve(feature_engineering_fixture):
	'''
	test plot_roc_curve
	'''
	logger.info("Testing plot_roc_curve function")
	X_train,\
	X_test,\
	y_train,\
	y_test = feature_engineering_fixture
	quick_model_name = 'DummyClassifier'
	filename = f'{quick_model_name}_roc_curve.png'
	test_path = os.path.join(output_dir, filename)	
	
	quick_model = DummyClassifier(strategy="most_frequent")
	quick_model.fit(X_train, y_train)
	
	try:
		plot_roc_curve(
			model1=quick_model,
			X=X_test,
			y=y_test,
			output_dir=output_dir
		)
		logger.info("Testing plot_roc_curve: SUCCESS")
	except Exception as err:
		logger.error(f"Testing plot_roc_curve: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logger.error(f"Testing plot_roc_curve: {err}")
		raise err


def test_shap_plot(feature_engineering_fixture):
	'''
	test shap_plot
	'''
	logger.info("Testing shap_plot function")
	filename = 'shap_summary_plot_rf.png'
	test_path = os.path.join(output_dir, filename)

	X_train,\
	X_test,\
	y_train,\
	_ = feature_engineering_fixture
		
	rfc = RandomForestClassifier(random_state=42)
	rfc.fit(X_train, y_train)
	
	try:
		assert isinstance(features, list), "Features are not provided as a list"
	except AssertionError as err:
		logger.error(f"Testing shap_plot: {err}")
		raise err
	try:		
		assert len(features) == X_test.shape[1], "Number of features does not match the model input"
	except AssertionError as err:
		logger.error(f"Testing shap_plot: {err}")
		raise err
	try:
		shap_plot(
			rfc,
			X_test,
			features,
			output_dir
		)
		logger.info("Testing shap_plot: SUCCESS")
	except Exception as err:
		logger.error(f"Testing shap_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logger.error(f"Testing shap_plot: {err}")
		raise err
	

def test_train_models(feature_engineering_fixture):
	'''
	test train_models
	'''
	logger.info("Testing train_models function")

	X_train, X_test, y_train, y_test = feature_engineering_fixture

	output_dir_images_test = './tests/images/model_images/'
	if not os.path.exists(output_dir_images_test):
		os.makedirs(output_dir_images_test)
	output_dir_models_test = './tests/models/'
	if not os.path.exists(output_dir_models_test):
		os.makedirs(output_dir_models_test)

	try:
		train_models(
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			feature_names=features,
            output_dir_images=output_dir_images_test,
			output_dir_models=output_dir_models_test
		)
		success = 1
		logger.info("Testing train_models: SUCCESS")
	except Exception as err:
		logger.error(f"Testing train_models: {err}")
		raise err
	try:
		assert success == 1, "train_models did not complete successfully"
	except AssertionError as err:
		logger.error(f"Testing train_models: {err}")
		raise err
	try:
		filename_rf = 'rfc_classification_report.png'
		test_path_rf = os.path.join(output_dir_images_test, filename_rf)
		assert os.path.exists(test_path_rf), f"File {filename_rf} does not exist in {output_dir_images_test}"

		filename_lr = 'lc_classification_report.png'
		test_path_lr = os.path.join(output_dir_images_test, filename_lr)
		assert os.path.exists(test_path_lr), f"File {filename_lr} does not exist in {output_dir_images_test}"

		filename_roc_curve = 'randomforestclassifier_roc_curve.png'
		test_path_roc_curve = os.path.join(output_dir_images_test, filename_roc_curve)
		assert os.path.exists(test_path_roc_curve), f"File {filename_roc_curve} does not exist in {output_dir_images_test}"

		filename_roc_curve = 'logisticregression_roc_curve.png'
		test_path_roc_curve = os.path.join(output_dir_images_test, filename_roc_curve)
		assert os.path.exists(test_path_roc_curve), f"File {filename_roc_curve} does not exist in {output_dir_images_test}"

		filename_roc_curve = 'randomforestclassifier_logisticregression_roc_curve.png'
		test_path_roc_curve = os.path.join(output_dir_images_test, filename_roc_curve)
		assert os.path.exists(test_path_roc_curve), f"File {filename_roc_curve} does not exist in {output_dir_images_test}"

		filename_feature_importance = 'feature_importance_rf.png'
		test_path_feature_importance = os.path.join(output_dir_images_test, filename_feature_importance)
		assert os.path.exists(test_path_feature_importance), f"File {filename_feature_importance} does not exist in {output_dir_images_test}"

		filename_shap = 'shap_summary_plot_rf.png'
		test_path_shap = os.path.join(output_dir_images_test, filename_shap)
		assert os.path.exists(test_path_shap), f"File {filename_shap} does not exist in {output_dir_images_test}"

		# Check if models are saved
		filename_rc = 'rfc_model.pkl'
		path_rc = os.path.join(output_dir_models_test, filename_rc)
		assert os.path.exists(path_rc), f"File {filename_rc} does not exist in {output_dir_models_test}"

		filename_lr = 'logistic_model.pkl'
		path_lr = os.path.join(output_dir_models_test, filename_lr)
		assert os.path.exists(path_lr), f"File {filename_lr} does not exist in {output_dir_models_test}"
	except AssertionError as err:
		logger.error(f"Testing train_models: {err}")
		raise err
		


if __name__ == "__main__":
	logger.info("Starting tests for churn prediction library")
	# Run all tests
	test_import()
	test_create_target()
	test_create_target_dist_plot()
	test_create_bar_plot()
	test_create_histogram()
	test_create_density_plot()
	test_create_heatmap()
	test_perform_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_classification_report_image()
	test_feature_importance_plot()
	test_plot_roc_curve()
	test_shap_plot()
	test_train_models()




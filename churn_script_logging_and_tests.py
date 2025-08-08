import os
import logging
import pytest
from sklearn.dummy import DummyClassifier
#import churn_library_solution as cls
from churn_library import X_train, import_data,\
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

logging.basicConfig(
    filename='./logs/churn_library_tests.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Define variables
global output_dir
output_dir = './tests/images/'

global target
target = 'Churn'

global cat_columns, quant_columns, features
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


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	logging.info("Testing import_data function")
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


@pytest.fixture
def import_dataframe(scope='module'):
	'''
	provides a dataframe for testing
	'''
	df = import_data("./data/bank_data.csv")
	return df


def test_create_target(import_dataframe):
	'''
	test create_target function
	'''
	logging.info("Testing create_target function")
	df = import_dataframe

	if df is None:
		logging.error("Testing create_target: DataFrame is None")
		raise ValueError("DataFrame is None")	
	try:
		df = create_target("./data/bank_data.csv")
		logging.info("Testing create_target: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing create_target: The file wasn't found")
		raise err
	try:
		assert 'churn' in df.columns
	except AssertionError as err:
		logging.error("Testing create_target: The target column 'churn' was not created")
		raise err


def test_create_target_dist_plot(import_dataframe):
	'''
	test create_target_dist_plot function
	'''
	logging.info("Testing create_target_dist_plot function")
	df = import_dataframe	
	filename = f'{target}_histogram.png'
	test_path = os.path.join(output_dir, filename)

	try:
		create_target_dist_plot(df, target, output_dir)
		logging.info("Testing create_target_dist_plot: SUCCESS")
	except Exception as err:
		logging.error(f"Testing create_target_dist_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logging.error(f"Testing create_target_dist_plot: The plot was not saved to the specified path")
		raise err


def test_create_bar_plot(import_dataframe):
	'''
	test create_bar_plot function
	'''
	logging.info("Testing create_bar_plot function")
	df = import_dataframe
	test_column = cat_columns[0]  # Use the first categorical column for testing
	
	filename = f'cat_{test_column}_bar_plot.png'
	test_path = os.path.join(output_dir, filename)
		
	try:
		create_bar_plot(df, test_column)
		logging.info("Testing create_bar_plot: SUCCESS")
	except Exception as err:
		logging.error(f"Testing create_bar_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logging.error(f"Testing create_bar_plot: The plot was not saved to the specified path")
		raise err

def test_create_histogram(import_dataframe):
	'''
	test create_histogram function
	'''
	logging.info("Testing create_histogram function")
	df = import_dataframe
	test_column = quant_columns[0]  # Use the first quantitative column for testing
    
	filename = f'quant_{test_column}_histogram.png'
	test_path = os.path.join(output_dir, filename)
	
	try:
		create_histogram(df, test_column, output_dir)
		logging.info("Testing create_histogram: SUCCESS")
	except Exception as err:
		logging.error(f"Testing create_histogram: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logging.error(f"Testing create_histogram: The plot was not saved to the specified path")
		raise err
	

def test_create_density_plot(import_dataframe):
	'''
	test create_density_plot function
	'''
	logging.info("Testing create_density_plot function")
	df = import_dataframe
	test_column = quant_columns[0]  # Use the first quantitative column for testing
	
	filename = f'quant_{test_column}_density_plot.png'
	test_path = os.path.join(output_dir, filename)
	
	try:
		create_density_plot(df, test_column, output_dir)
		logging.info("Testing create_density_plot: SUCCESS")
	except Exception as err:
		logging.error(f"Testing create_density_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logging.error(f"Testing create_density_plot: The plot was not saved to the specified path")
		raise err

def test_create_heatmap(import_dataframe):
	'''
	test create_heatmap function
	'''
	logging.info("Testing create_heatmap function")
	df = import_dataframe
	filename = 'correlation_heatmap.png'
	test_path = os.path.join(output_dir, filename)

	try:
		create_heatmap(df, quant_columns, output_dir)
		logging.info("Testing create_heatmap: SUCCESS")
	except Exception as err:
		logging.error(f"Testing create_heatmap: {err}")
		raise err
	try:
		assert os.path.exists(test_path)
	except AssertionError as err:
		logging.error(f"Testing create_heatmap: The heatmap was not saved to the specified path")
		raise err


def test_perform_eda(import_dataframe, perform_eda):
	'''
	test perform eda function
	'''
	logging.info("Testing perform_eda function")
	df = import_dataframe
	output_dir_eda = './tests/images/eda_images/'

	try:
		perform_eda(df,
			        cat_columns,
					quant_columns,
					target,
					output_dir_eda)
		success = 1
		logging.info("Testing perform_eda: SUCCESS")
	except Exception as err:
		logging.error(f"Testing perform_eda: {err}")
		raise err
	try:
		assert success == 1
	except AssertionError as err:
		logging.error(f"Testing perform_eda: The EDA function did not complete successfully")
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
		logging.error(f"Testing perform_eda: {err}")
		raise err
		

def test_encoder_helper(import_dataframe):
	'''
	test encoder helper
	'''
	logging.info("Testing encoder_helper function")
	df = import_dataframe
	try:
		encoded_df = encoder_helper(df, cat_columns, target)
		logging.info("Testing encoder_helper: SUCCESS")
	except Exception as err:
		logging.error(f"Testing encoder_helper: {err}")
		raise err
	try:
		assert all(f"{col}_Churn" in encoded_df.columns for col in cat_columns),\
			"Not all categorical columns were encoded correctly"
	except AssertionError as err:
		logging.error(f"Testing encoder_helper: {err}")
		raise err


def test_perform_feature_engineering(import_dataframe):
	'''
	test perform_feature_engineering
	'''
	logging.info("Testing perform_feature_engineering function")
	df = import_dataframe
	try:
		X_train,\
		X_test,\
		y_train,\
		y_test = perform_feature_engineering(df,
			features,
			target)
		logging.info("Testing perform_feature_engineering: SUCCESS")
	except Exception as err:
		logging.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert len(X_train) > 0 and len(X_test) > 0, "Training and test sets are empty"
	except AssertionError as err:
		logging.error(f"Testing perform_feature_engineering: {err}")
		raise err
	try:
		assert len(y_train) > 0 and len(y_test) > 0, "Training and test labels are empty"
	except AssertionError as err:
		logging.error(f"Testing perform_feature_engineering: {err}")
		raise err



@pytest.fixture
def perform_feature_engineering(import_dataframe,\
								scope='module'):
	'''
	provides a perform_feature_engineering function for testing
	'''
	df = import_dataframe	
	X_train,\
	X_test,\
	y_train,\
	y_test = perform_feature_engineering(df,
				features,
				target)
	return X_train, X_test, y_train, y_test



def test_classification_report_image(perform_feature_engineering):
	'''
	test classification_report_image
	'''
	logging.info("Testing classification_report_image function")
	filename_rf = 'rfc_classification_report.png'
	test_path_rf = os.path.join(output_dir, filename_rf)

	#df = import_dataframe
	_, _, y_train, y_test = perform_feature_engineering
	y_train_preds = [0] * len(y_train)  # Dummy predictions for testing
	y_test_preds = [0] * len(y_test)    # Dummy predictions for testing
	
	try:
		classification_report_image(y_train,
							  y_test,
							  y_train_preds,
							  y_train_preds,
							  y_test_preds,
							  y_test_preds
							  )
		logging.info("Testing classification_report_image: SUCCESS")
	except Exception as err:
		logging.error(f"Testing classification_report_image: {err}")
		raise err
	try:
		assert os.path.exists(test_path_rf), f"File {filename_rf} does not exist in {output_dir}"
	except AssertionError as err:
		logging.error(f"Testing classification_report_image: {err}")
		raise err


def test_feature_importance_plot(perform_feature_engineering):
	'''
	test feature_importance_plot
	'''
	logging.info("Testing feature_importance_plot function")
	#df = import_dataframe
	filename = 'feature_importance_rf.png'
	test_path = os.path.join(output_dir, filename)

	X_train,\
	_,\
	y_train,\
	y_test = perform_feature_engineering
	y_train_preds_rf = [0] * len(y_train)  # Dummy predictions for testing
	y_test_preds_rf = [0] * len(y_test)    # Dummy predictions for testing
	
	try:
		feature_importance_plot(X_train,
						  y_train,
						  y_train_preds_rf,
						  y_test_preds_rf,
						  output_dir)
		logging.info("Testing feature_importance_plot: SUCCESS")
	except Exception as err:
		logging.error(f"Testing feature_importance_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logging.error(f"Testing feature_importance_plot: {err}")
		raise err

def test_plot_roc_curve(perform_feature_engineering):
	'''
	test plot_roc_curve
	'''
	logging.info("Testing plot_roc_curve function")
	#df = import_dataframe
	X_train,\
	X_test,\
	y_train,\
	y_test = perform_feature_engineering
	quick_model_name = 'DummyClassifier'
	filename = f'{quick_model_name}_roc_curve.png'
	test_path = os.path.join(output_dir, filename)	
	
	quick_model = DummyClassifier(strategy="most_frequent")
	quick_model.fit(X_train, y_train)
	
	try:
		plot_roc_curve(
			quick_model,
			X_test,
			y_test
		)
		logging.info("Testing plot_roc_curve: SUCCESS")
	except Exception as err:
		logging.error(f"Testing plot_roc_curve: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logging.error(f"Testing plot_roc_curve: {err}")
		raise err


def test_shap_plot(perform_feature_engineering):
	'''
	test shap_plot
	'''
	logging.info("Testing shap_plot function")
	#df = import_dataframe
	filename = 'shap_summary_plot_rf.png'
	test_path = os.path.join(output_dir, filename)

	X_train,\
	X_test,\
	y_train,\
	_ = perform_feature_engineering
		
	quick_model = DummyClassifier(strategy="most_frequent")
	quick_model.fit(X_train, y_train)

	try:
		shap_plot(quick_model, X_test, output_dir)
		logging.info("Testing shap_plot: SUCCESS")
	except Exception as err:
		logging.error(f"Testing shap_plot: {err}")
		raise err
	try:
		assert os.path.exists(test_path), f"File {filename} does not exist in {output_dir}"
	except AssertionError as err:
		logging.error(f"Testing shap_plot: {err}")
		raise err	



def test_train_models(perform_feature_engineering):
	'''
	test train_models
	'''
	logging.info("Testing train_models function")
	#df = import_dataframe

	X_train, X_test, y_train, y_test = perform_feature_engineering

	output_dir_images_test = './tests/images/model_images/'
	output_dir_models_test = './tests/models/'

	try:
		train_models(X_train,
			         X_test,
					 y_train,
					 y_test,
					 output_dir_images_test,
					 output_dir_models_test)
		success = 1
		logging.info("Testing train_models: SUCCESS")
	except Exception as err:
		logging.error(f"Testing train_models: {err}")
		raise err
	try:
		assert success == 1, "train_models did not complete successfully"
	except AssertionError as err:
		logging.error(f"Testing train_models: {err}")
		raise err
	try:		
		filename_rf = 'rfc_classification_report.png'
		test_path_rf = os.path.join(output_dir_images_test, filename_rf)
		assert os.path.exists(test_path_rf), f"File {filename_rf} does not exist in {output_dir_images_test}"
		
		filename_lr = 'logistic_classification_report.png'
		test_path_lr = os.path.join(output_dir_images_test, filename_lr)
		assert os.path.exists(test_path_lr), f"File {filename_lr} does not exist in {output_dir_images_test}"

		filename_roc_curve = 'DummyClassifier_roc_curve.png'
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
		logging.error(f"Testing train_models: {err}")
		raise err
		


if __name__ == "__main__":
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




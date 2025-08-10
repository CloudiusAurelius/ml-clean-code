import logging



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

logger = setup_logging(filename='./logs/test_log.log')


"""
# Initialize the logging modulelogging.basicConfig(
filename='./logs/test_log.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Adding a file handler
file_handler = logging.FileHandler(filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Attach the handler to the logger
logger.addHandler(file_handler)
"""

# Log an error message
logging.error("An error occurred with logging. Check app.log for details.")
#logger.error("An error occurred with logger. Check app.log for details.")

logging.info("this is a logging test message with logging")
#logger.info("this is a logging test message with logger")

cat_columns=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
]
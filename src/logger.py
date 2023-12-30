import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure the logger with a FileHandler for file output
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(LOG_FILEPATH)])

# Optionally, add a StreamHandler for console output
logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logging.info("This is a test log message")
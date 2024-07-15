import os
import sys
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entred the data ingestion method or component")
        try:
            dataset = pd.read_csv("/home/yassine/Desktop/Auto_MPG/notebooks/data/data.csv")
            logging.info("Read data as Dataframe")
            ## save raw data as csv 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            dataset.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            ## split data into test and train
            train_data, test_data = train_test_split(dataset, test_size=0.2,random_state=42)
            logging.info("Spliting dataset into train and test")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    ingestion_obj.initiate_data_ingestion()

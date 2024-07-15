import os
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.exception import CustomException
import pandas as pd
import numpy as np
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_file_path  = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            num_features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
            cat_features = ["car name"]
            
            num_pipeline = Pipeline(
                steps=[
                ("scaler",StandardScaler(),num_features)
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("oh-encoder",OneHotEncoder()),
                    ("scalar",StandardScaler())
                ]
            )

            logging.info("numerical columns standred scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num-pipeline",num_pipeline,num_features),
                    ("cat-pipeline",cat_pipeline,cat_features),
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("read train and test data are completed")

            logging.info("Obtaining preprocessor object")

            preprocessor = self.get_data_transformer()

            target_column_name = "mpg"
            X_train = train_data.drop(columns=[target_column_name],axis=1)
            y_train = train_data[target_column_name]

            X_test = test_data.drop(columns=[target_column_name],axis=1)
            y_test = test_data[target_column_name]

            logging.info("Applying preprocessor on training and testing data")

            input_train_feature_arr = preprocessor.fit_transform(X_train)
            input_test_feature_arr = preprocessor.fit_transform(X_test)

            train_arr = np.c_[ input_train_feature_arr, np.array(y_train) ] 
            test_arr = np.c_[input_test_feature_arr, np.array(y_test)] 

            logging.info("Saved Preproccessing object")

            save_obj(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
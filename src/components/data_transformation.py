import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):    
        '''
        This function si responsible for data transformation
        
        '''
        try:
            numerical_cols =['age', 'capital-gain', 'capital-loss', 'hours-per-week']
            categorical_cols =  ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'country']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                    steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())                
                    ]
                )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='most_frequent')),
                            ('OneHotEncoder',OneHotEncoder()),
                            ("scaler",StandardScaler(with_mean=False))
                            ]
                        )
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor = ColumnTransformer(
                            [
                            ('num_pipeline',num_pipeline,numerical_cols),
                            ('cat_pipeline',cat_pipeline,categorical_cols)
                            ]
                        )
            return preprocessor

            

        except Exception as e:
            logging.info("Exception occured in Data Transformation Phase")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:    
            # Reading the train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            
           
            target_column_name = 'salary'
           
            
            
            drop_columns = [target_column_name,'education-num','fnlwgt']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
           


           
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("applying label encoder to target data")
            
            le = LabelEncoder()

            target_train_arr = le.fit_transform(target_feature_train_df)
            target_test_arr = le.transform(target_feature_test_df)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                input_feature_train_arr,
                target_train_arr,
                input_feature_test_arr,
                target_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)


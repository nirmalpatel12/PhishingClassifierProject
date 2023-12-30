import sys
sys.path.append('D:\\ML_projects\\PhishingClassifier\\src')
import os
from exception import CustomException
from logger import logging
from utils import load_object
import pandas as pd
import joblib


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            
            #preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            
            #preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            logging.info('preprocessor and model is loaded')
            
            #data_scaled=preprocessor.transform(features)

            pred=model.predict(features)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 
                 qty_slash_url:float,
                 length_url:float,
                 qty_dot_domain:float, 
                 qty_dot_directory:float,
                 qty_hyphen_directory	:float,
                 qty_underline_directory	:float,
                 
                 asn_ip:float,
                 time_domain_activation:float,
                 time_domain_expiration:float,
                 ttl_hostname:float,
                
                 
                ):
        
        self.qty_slash_url=qty_slash_url
        self.length_url=length_url
        self.qty_dot_domain=qty_dot_domain
        self.qty_dot_directory=qty_dot_directory
        self.qty_hyphen_directory=qty_hyphen_directory
        self.qty_underline_directory=qty_underline_directory
        
        self.asn_ip=asn_ip
        self.time_domain_activation=time_domain_activation
        self.time_domain_expiration=time_domain_expiration
        self.ttl_hostname=ttl_hostname
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                
                'qty_slash_url':[self.qty_slash_url],
                'length_url':[self.length_url],
                'qty_dot_domain':[self.qty_dot_domain],
                'qty_dot_directory':[self.qty_dot_directory],
                'qty_hyphen_directory':[self.qty_hyphen_directory],
                'qty_underline_directory':[self.qty_underline_directory],
                
                'asn_ip':[self.asn_ip],
                'time_domain_activation':[self.time_domain_activation],
                'time_domain_expiration':[self.time_domain_expiration],
                'ttl_hostname':[self.ttl_hostname]
                
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
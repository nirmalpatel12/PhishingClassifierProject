import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import FunctionTransformer
import sys
import os
from joblib import dump, load
sys.path.append('D:\\ML_projects\\PhishingClassifier\\src')
from dataclasses import dataclass
import pandas as pd
import numpy as np

from exception import CustomException
from logger import logging

from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.abspath(os.path.join('artifacts', 'preprocessor.pkl'))

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def load_data(self, file_path):
        try:
            logging.info(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise CustomException(f"Error loading data: {str(e)}")

    def resample_data(self, X, Y):
        try:
            logging.info("Resampling data using SMOTE")
            
        
            
            sm = SMOTE(sampling_strategy='minority', random_state=42)
            
            X_resampled, Y_resampled = sm.fit_resample(X, Y)
            return pd.concat([pd.DataFrame(Y_resampled), pd.DataFrame(X_resampled)], axis=1)
        except Exception as e:
            logging.error(f"Error during data resampling: {str(e)}")
            raise CustomException(f"Error during data resampling: {str(e)}")

    def handle_outliers_continious(self, data, continuous_features):
        try:
            logging.info("Handling outliers")
            for feature in continuous_features:
                IQR = data[feature].quantile(0.75) - data[feature].quantile(0.25)
                lower_bridge = data[feature].quantile(0.25) - (IQR * 1.5)
                upper_bridge = data[feature].quantile(0.75) + (IQR * 1.5)
                data.loc[data[feature] < lower_bridge, feature] = lower_bridge
                data.loc[data[feature] >= upper_bridge, feature] = upper_bridge
            return data
        except Exception as e:
            logging.error(f"Error handling outliers: {str(e)}")
            raise CustomException(f"Error handling outliers: {str(e)}")
        
    def handle_outliers_discrete(self, data, discrete_features):
        try:
            logging.info("Handling outliers")
            for feature in discrete_features:
                IQR = data[feature].quantile(0.75) - data[feature].quantile(0.25)
                lower_bridge = data[feature].quantile(0.25) - (IQR * 1.5)
                upper_bridge = data[feature].quantile(0.75) + (IQR * 1.5)
                data.loc[data[feature] < lower_bridge, feature] = lower_bridge
                data.loc[data[feature] >= upper_bridge, feature] = upper_bridge
            return data
        except Exception as e:
            logging.error(f"Error handling outliers: {str(e)}")
            raise CustomException(f"Error handling outliers: {str(e)}")    

    def drop_highly_correlated_features(self, X, threshold=0.85):
        try:
            logging.info("Dropping highly correlated features")
            corr_matrix = X.corr()
            corr_features = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        colname = corr_matrix.columns[i]
                        corr_features.add(colname)
            X.drop(corr_features, axis=1, inplace=True)
            return X
        except Exception as e:
            logging.error(f"Error dropping highly correlated features: {str(e)}")
            raise CustomException(f"Error dropping highly correlated features: {str(e)}")

    def drop_constant_features(self, X):
        try:
            logging.info("Dropping constant features")
            var_thres = VarianceThreshold(threshold=0)
            var_thres.fit(X)
            constant_columns = [column for column in X.columns if column not in X.columns[var_thres.get_support()]]
            X.drop(constant_columns, axis=1, inplace=True)
            return X
        except Exception as e:
            logging.error(f"Error dropping constant features: {str(e)}")
            raise CustomException(f"Error dropping constant features: {str(e)}")
    
    def scale_features(self, df, dependent_feature):
        try:
            logging.info("Scaling features")
            scaler = StandardScaler()
            if dependent_feature in df.columns:
               # If the target column is present, scale all columns except the target column
                scaled_features = scaler.fit_transform(df.drop(columns=[dependent_feature]))
                scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns[df.columns != dependent_feature])
                scaled_features_df[dependent_feature] = df[dependent_feature]
            else:
                # If the target column is not present, scale all columns
                scaled_features = scaler.fit_transform(df)
                scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

            return scaled_features_df
        except Exception as e:
            logging.error(f"Error scaling features: {str(e)}")
            raise CustomException(f"Error scaling features: {str(e)}")

    def select_features_correlation(self, df, dependent_feature, threshold=0.5):
        try:
           logging.info("Selecting features based on correlation")

           # Calculate correlations with the dependent feature
           correlation_matrix = df.corr()
           correlation_with_target = correlation_matrix[dependent_feature]


           # Select columns correlated threshold and above with the dependent feature
           selected_columns = correlation_with_target[abs(correlation_with_target) >= threshold].index.tolist()

           # Exclude the dependent feature itself from the selected columns
           if dependent_feature in selected_columns:
               selected_columns.remove(dependent_feature)

           # Create a DataFrame with the selected columns
           selected_features_df = df[selected_columns]

           # Add the dependent feature column to the selected features
           selected_features_df[dependent_feature] = df[dependent_feature]

           return selected_features_df

        except Exception as e:
            logging.error(f"Error selecting features based on correlation: {str(e)}")
            raise CustomException(f"Error selecting features based on correlation: {str(e)}")


    
    def preprocess_data(self, config):
      try:
          
          logging.info("Starting data preprocessing")
          # Load data
          df = self.load_data(config.input_file_path)

          # Convert to int32
          df = df.astype('int32')
          target_column_name='phishing'
          # Extract features and target
          X = df.drop(labels=config.target_column_name, axis=1)
          Y = df[[config.target_column_name]]

          # Define continuous and discrete features
          numerical_feature = [feature for feature in X.columns if X[feature].dtypes != 'O']
          discrete_feature = [feature for feature in numerical_feature if len(X[feature].unique()) < 25]
          continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]

          # Create the pipeline
          preprocessing_pipeline = Pipeline([
             # ('resample', FunctionTransformer(self.resample_data, kw_args={'Y': Y})),
              ('handle_outliers_continuous', FunctionTransformer(self.handle_outliers_continious, kw_args={'continuous_features': continuous_feature})),
              ('handle_outliers_discrete', FunctionTransformer(self.handle_outliers_discrete, kw_args={'discrete_features': discrete_feature})),
              ('drop_corr_features', FunctionTransformer(self.drop_highly_correlated_features, kw_args={'threshold': 0.85})),
              ('drop_constant_features', FunctionTransformer(self.drop_constant_features)),
              ('scale_features', FunctionTransformer(self.scale_features, kw_args={'dependent_feature': target_column_name})),
              #('select_features_correlation', FunctionTransformer(self.select_features_correlation, kw_args={'dependent_feature': config.target_column_name, 'threshold': 0.5}))
          ])

          # Fit and transform the data
          df_processed = preprocessing_pipeline.fit_transform(X, Y)

          # Save the preprocessor object (optional)
          dump(preprocessing_pipeline, config.preprocessor_obj_file_path)

          # Save the preprocessed data
          df_processed.to_csv(config.output_file_path, index=False)

          logging.info(f"Preprocessed data saved to {config.output_file_path}")
          logging.info(f"Preprocessor object saved to {config.preprocessor_obj_file_path}")
          return preprocessing_pipeline
      except Exception as e:
               logging.error(f"Error during data preprocessing: {str(e)}")
               raise CustomException(f"Error during data preprocessing: {str(e)}")
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
           
           # Reading train and test data
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           logging.info('Read train and test data completed')
           logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
           logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

           logging.info('Obtaining preprocessing object')
           config = DataTransformationConfig()

           preprocessing_obj = self.preprocess_data(config)

           target_column_name = 'phishing'
           drop_columns = [target_column_name]

           # features into independent and dependent features
           input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
           target_feature_train_df = train_df[target_column_name]

           input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
           target_feature_test_df = test_df[target_column_name]

           # Apply the transformation
           input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
           input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

           logging.info("Applying preprocessing object on training and testing datasets.")

           train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
           test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                # Print train_arr and test_arr
           logging.info(f"Train Array: \n{train_arr}")
           logging.info(f"Test Array: \n{test_arr}")
           # Print train_arr and test_arr shapes
           logging.info(f"Train Array Shape: {train_arr.shape}")
           logging.info(f"Test Array Shape: {test_arr.shape}")

           save_object(
              file_path=self.data_transformation_config.preprocessor_obj_file_path,
              obj=preprocessing_obj
            )

           logging.info('Processor pickle is created and saved')

           return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error during data transformation initialization: {str(e)}")
            raise CustomException(f"Error during data transformation initialization: {str(e)}")

    

class DataTransformationConfig:
    input_file_path = 'artifacts/raw.csv'
    target_column_name = 'phishing'
    output_file_path = 'artifacts/preprocessed.csv'
    preprocessor_obj_file_path: str = 'artifacts/preprocessor.pkl'
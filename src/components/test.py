from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

ingest_obj = DataIngestion()
train_data_path, test_data_path = ingest_obj.initiate_data_ingestion()

transform_obj = DataTransformation()
train_arr, test_arr = transform_obj.initiate_data_transformation(train_data_path, test_data_path)

trainer_obj = ModelTrainer()
trainer_obj.initiate_model_trainer(train_arr, test_arr)


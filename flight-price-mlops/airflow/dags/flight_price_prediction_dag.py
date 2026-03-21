from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from utils.data_ingestion import DataLoader
from utils.data_transformation import DataTransformer
from utils.model_training import RandomForestModel

DATA_FILE_PATH = '/opt/airflow/dags/data/flights.csv'
RAW_DATA_PATH = '/opt/airflow/dags/data/raw.csv'
PROCESSED_DATA_PATH = '/opt/airflow/dags/data/processed_data.npz'
MODEL_PATH = '/opt/airflow/dags/data/model.pkl'
METRICS_PATH = '/opt/airflow/dags/data/metrics.json'


def load_data():
    loader = DataLoader(DATA_FILE_PATH)
    df = loader.load_data()
    loader.save_data(df, RAW_DATA_PATH)


def transform_data():
    transformer = DataTransformer(RAW_DATA_PATH)
    transformer.process_and_save(PROCESSED_DATA_PATH)


def train_model():
    model = RandomForestModel(PROCESSED_DATA_PATH)
    model.train_and_save(MODEL_PATH, METRICS_PATH)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id='flight_price_prediction_dag',
    default_args=default_args,
    description='Flight Price Prediction (Lightweight Demo)',
    schedule=None,
    catchup=False
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag
)

transform_data_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=transform_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag
)

load_data_task >> transform_data_task >> train_model_task
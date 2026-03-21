import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataTransformer:
    def __init__(self, input_path):
        self.input_path = input_path

    def process_and_save(self, output_path):
        df = pd.read_csv(self.input_path)

        df['date'] = pd.to_datetime(df['date'])

        df['week_day'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['week_no'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day

        df.rename(columns={"to": "destination"}, inplace=True)

        df['flight_speed'] = round(df['distance'] / df['time'], 2)

        df = pd.get_dummies(df, columns=['from', 'destination', 'flightType', 'agency'])

        df.drop(columns=['time', 'flight_speed', 'month', 'year', 'distance', 'date'], inplace=True)

        X = df.drop('price', axis=1)
        y = df['price']

        # Simple safe handling (avoid column mismatch errors)
        X = X.fillna(0)

        X = X.select_dtypes(include=['int64', 'float64', 'uint8'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        np.savez(
            output_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.values,
            y_test=y_test.values
        )
import numpy as np
import json
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RandomForestModel:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = np.load(self.data_path)

        return (
            data['X_train'],
            data['X_test'],
            data['y_train'],
            data['y_test']
        )

    def train_and_save(self, model_path, metrics_path):
        X_train, X_test, y_train, y_test = self.load_data()

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred))
        }

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("Model and metrics saved!")
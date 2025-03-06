from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np

class SERPModel:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = ["keyword_density", "readability_score", "page_speed", "ctr", "backlinks"]
        X = self.df[features]
        y = self.df["serp_rank"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_mlp_classifier(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        y_train = np.digitize(y_train, bins=[10, 30])  # Categorize ranks into Top, Mid, Low
        y_test = np.digitize(y_test, bins=[10, 30])
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        return model

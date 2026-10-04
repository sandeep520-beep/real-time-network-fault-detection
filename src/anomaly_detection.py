from sklearn.ensemble import IsolationForest


class NetworkAnomalyDetector:

    def __init__(self, contamination=0.05):

        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )

    def train(self, X):

        self.model.fit(X)

    def predict(self, X):

        predictions = self.model.predict(X)

        return predictions

    def detect_anomalies(self, X):

        predictions = self.predict(X)

        return predictions == -1

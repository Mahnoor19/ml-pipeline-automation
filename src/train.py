from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from preprocess import load_data, preprocess_data

def train_model():
    data = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(model, 'models/iris_model.joblib')
    joblib.dump(scaler, 'models/iris_scaler.joblib')
    
    return accuracy

if __name__ == "__main__":
    train_model()

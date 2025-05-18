from src.preprocess import load_data, preprocess_data

def test_load_data():
    data = load_data()
    assert data.shape == (150, 5)
    assert not data.isnull().any().any()

def test_preprocess_data():
    data = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(data)
    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)

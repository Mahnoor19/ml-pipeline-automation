from train import train_model  # new
from preprocess import load_data, preprocess_data  # new

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.9

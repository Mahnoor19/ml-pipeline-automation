from train import train_model  
from preprocess import load_data, preprocess_data  

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.9

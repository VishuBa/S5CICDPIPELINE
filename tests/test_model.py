import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MNISTModel, count_parameters

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Check output shape
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Check parameter count
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"
    
    print("All architecture tests passed!")

def test_model_accuracy():
    from src.test import test_model
    accuracy = test_model()
    assert accuracy > 10, f"Model accuracy is {accuracy}%, should be greater than 10%"

if __name__ == "__main__":
    test_model_architecture()
    test_model_accuracy()
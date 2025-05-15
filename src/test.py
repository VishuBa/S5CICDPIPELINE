import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTModel, count_parameters

def test_model():
    # Set device to CPU
    device = torch.device('cpu')
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Add download=True to ensure dataset is downloaded
    test_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Load model
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load('models/mnist_model_latest.pth', map_location=device))
    model.eval()
    
    # Test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    accuracy = test_model()
    print(f'Test Accuracy: {accuracy:.2f}%')
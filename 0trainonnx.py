import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    # --- 1. Device Configuration ---
    # Set the device to a CUDA-enabled GPU if available, otherwise use the CPU.
    # This is the core step to ensure the script runs on your NVIDIA RTX GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")
    if torch.cuda.is_available():
        print(f"--- GPU: {torch.cuda.get_device_name(0)} ---")

    # --- 2. Hyperparameters ---
    num_epochs = 25
    batch_size = 128
    learning_rate = 0.001

    # --- 3. Data Loading and Augmentation ---
    # Define transformations for the training data to increase dataset variance.
    # This includes random horizontal flips and rotations.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # Normalization values are standard for CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # For the test set, we only need to convert to a tensor and normalize.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders to feed data to the model in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # CIFAR-10 class labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- 4. Model Definition ---
    # A more complex CNN with multiple convolutional blocks, batch normalization, and dropout.
    class ComplexCNN(nn.Module):
        def __init__(self):
            super(ComplexCNN, self).__init__()
            # Block 1
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # Block 2
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # Block 3
            self.conv_block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # Fully connected layers
            self.fc_block = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, len(classes))
            )

        def forward(self, x):
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.fc_block(x)
            return x

    # Instantiate the model and move it to the configured device (GPU)
    model = ComplexCNN().to(device)

    # --- 5. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler to adjust the learning rate during training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
        # Adjust the learning rate
        scheduler.step()
        print(f'Epoch {epoch+1} average loss: {running_loss / n_total_steps:.4f}')


    training_time = time.time() - start_time
    print(f"--- Finished Training in {training_time:.2f}s ---")

    # --- 7. Evaluation ---
    print("\n--- Evaluating Model on Test Data ---")
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for images, labels in test_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

        for i in range(len(classes)):
            if n_class_samples[i] > 0:
                class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]:5s} : {class_acc:.2f} %')
            else:
                print(f'Accuracy of {classes[i]:5s} : N/A (no samples)')

    # --- 8. Save the Model ---
    model_path = 'cifar10_cnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n--- Model saved to {model_path} ---")
    
    # --- 9. Export to ONNX ---
    print("\n--- Exporting Model to ONNX format ---")
    model.eval() # Set the model to evaluation mode
    
    # Create a dummy input tensor that matches the model's input dimensions
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    onnx_model_path = 'cifar10_cnn_model.onnx'

    # Export the model
    torch.onnx.export(model,                       # model being run
                      dummy_input,                 # model input (or a tuple for multiple inputs)
                      onnx_model_path,             # where to save the model
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=11,            # the ONNX version to export the model to
                      do_constant_folding=True,    # whether to execute constant folding for optimization
                      input_names=['input'],       # the model's input names
                      output_names=['output'],     # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print(f"--- Model successfully exported to {onnx_model_path} ---")


if __name__ == '__main__':
    main()



import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
import os

def main():
    """
    Main function to run inference on a sample image using the ONNX model.
    """
    # CIFAR-10 class labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    onnx_model_path = 'cifar10_cnn_model.onnx'

    # Check if the ONNX model file exists
    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX model file not found at '{onnx_model_path}'")
        print("Please run the 'pytorch_image_classifier.py' script first to train and export the model.")
        return

    print("\n--- Running Inference with ONNX Runtime on a Sample Image ---")
    
    # Use ONNX Runtime to run inference, leveraging the GPU provider if available
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"Running on: {ort_session.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Read a local sample image
    local_image_path = 'airplane.png' # IMPORTANT: Place an image with this name in the same folder
    print(f"Loading sample image from: {local_image_path}")


    # Check if the image file exists
    if not os.path.exists(local_image_path):
        print(f"Error: Image file not found at '{local_image_path}'")
        print("Please place a sample image file (e.g., 'sample_car.jpg') in the same directory as the script.")
        return

    try:
        img = Image.open(local_image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open or process the image file: {e}")
        return

    # Download a sample image
    #image_url = "https://www.publicdomainpictures.net/pictures/20000/nahled/car-1327915903434.jpg"
    #print(f"Downloading sample image from: {image_url}")
    #try:
    #    response = requests.get(image_url)
    #    response.raise_for_status() # Raise an exception for bad status codes
    #    img = Image.open(BytesIO(response.content)).convert("RGB")
    #except requests.exceptions.RequestException as e:
    #    print(f"Could not download image: {e}")
    #    return

    # Preprocess the image to match the model's input requirements
    # These transformations must be identical to the ones used for the test set in the training script
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0) # Add a batch dimension

    # Prepare inputs for ONNX Runtime. The input name must match the one specified during export.
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: img_tensor.cpu().numpy()}
    
    # Run inference
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Get the prediction
    prediction = np.argmax(ort_outs[0])
    predicted_class = classes[prediction]
    
    print(f"\n--- Inference Result ---")
    print(f"Predicted class for the sample image: '{predicted_class}'")

if __name__ == '__main__':
    main()


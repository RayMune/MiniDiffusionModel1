import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils  # Add this line
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Add this line

# Define the diffusion model

    # The rest of the code remains the same...


# Define a function to generate sample images
def generate_samples(model, num_samples=10):
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise as input
            noise = torch.randn(1, 3, 64, 64).to(device)  # Assuming input size 64x64x3
            # Generate sample image by passing noise through the diffusion model
            generated_image = model(noise)
            # Clamp pixel values to valid range
            generated_image = torch.clamp(generated_image, -1, 1)
            # Convert tensor to numpy array and transpose to (height, width, channels) format
            image_np = generated_image.squeeze().permute(1, 2, 0).cpu().numpy()
            # Display the generated image
            plt.imshow((image_np + 1) / 2)  # Scale the pixel values to [0, 1]
            plt.axis('off')
            plt.show()

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformation to preprocess images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a common size
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root='C:\\GANs\\images1', transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the diffusion model
    model = DiffusionModel().to(device)

    # Train the diffusion model
    train_diffusion_model(model, train_loader)

    # Generate sample images
    generate_samples(model, num_samples=10)

if __name__ == '__main__':
    main()

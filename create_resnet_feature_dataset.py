import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from loguru import logger
import pickle  # Import pickle module for saving data

# Dataset Definition
class DigitDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as f:
            self.data = f.read().splitlines(False)
        
        self.images_paths = []
        self.labels = []
        
        for row in self.data:
            image_path, label = row.split(',')
            self.images_paths.append(image_path)
            self.labels.append(int(label))  
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, self.images_paths[idx]  # Return filename as well

# Image Transformations for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_embeddings_pickle(data, filename="digit_embeddings.pkl"):
    # Save the data to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    # Load Dataset
    dataset = DigitDataset('digits/labels.txt',transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    # Load Pretrained ResNet18 (Feature Extractor)
    logger.info("Loading ResNet18 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
    resnet.to(device)
    resnet.eval()

    # Extract Features
    logger.info("Extracting features...")
    data = []

    with torch.inference_mode():
        for images, lbls, fns in tqdm(dataloader):
            images = images.to(device)
            features = resnet(images)  # Output shape: [batch, 512, 1, 1]
            features = features.view(features.size(0), -1).cpu().numpy()  # Flatten and move to CPU
            
            # Store filename, embedding, and label as tuples
            for i in range(len(fns)):
                data.append((fns[i], int(lbls[i]), features[i]))

    # Save dataset using pickle
    save_embeddings_pickle(data, 'digit_embeddings.pkl')
    logger.info(f"Saved {len(data)} embeddings to digit_embeddings.pkl")

if __name__ == '__main__':
    main()

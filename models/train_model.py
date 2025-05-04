import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mappings (matches your folder structure)
CHAR_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'add': 10, 'div': 11, 'mul': 12, 'sub': 13
}
OPERATOR_MAP = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}

class MathDataset(Dataset):
    def __init__(self, char_dir, expr_dir, is_train=True, img_size=28):
        self.char_images, self.char_labels = self.load_char_data(char_dir)
        self.expr_images, self.expr_labels = self.load_expr_data(expr_dir)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]) if is_train else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.img_size = img_size

    def load_char_data(self, char_dir):
        images, labels = [], []
        for class_name, class_id in CHAR_MAP.items():
            class_dir = os.path.join(char_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(class_id)
        return images, labels

    def load_expr_data(self, expr_dir):
        images, labels = [], []
        if not os.path.exists(expr_dir):
            return images, labels
        
        for expr_folder in os.listdir(expr_dir):
            expr_path = os.path.join(expr_dir, expr_folder)
            if not os.path.isdir(expr_path):
                continue
            
            # Parse expression like "3add5=8" or "8div2=4"
            parts = re.split(r'(add|div|mul|sub|=)', expr_folder)
            if len(parts) >= 4:  # e.g., ['3', 'add', '5', '=8']
                try:
                    operator = OPERATOR_MAP[parts[1]]
                    expression = f"{parts[0]}{operator}{parts[2]}"
                    label = eval(expression)
                    for img_file in os.listdir(expr_path):
                        img_path = os.path.join(expr_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            images.append(img)
                            labels.append(label)
                except:
                    continue
        return images, labels

    def __len__(self):
        return len(self.char_images) + len(self.expr_images)

    def __getitem__(self, idx):
        if idx < len(self.char_images):
            img = self.char_images[idx]
            label = self.char_labels[idx]
            task = 0  # Character classification
        else:
            img = self.expr_images[idx - len(self.char_images)]
            label = self.expr_labels[idx - len(self.char_images)]
            task = 1  # Expression evaluation
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)
        return img, label, task

class EnhancedMathModel(nn.Module):
    def __init__(self):
        super(EnhancedMathModel, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Character classification head
        self.char_head = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(CHAR_MAP))
        )
        
        # Expression evaluation head
        self.expr_head = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x, task):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Initialize output tensor
        outputs = torch.zeros(x.size(0), len(CHAR_MAP) if task[0].item() == 0 else 1, device=x.device)
        
        # Process character classification tasks
        char_mask = (task == 0)
        if char_mask.any():
            outputs[char_mask] = self.char_head(x[char_mask])
        
        # Process expression evaluation tasks
        expr_mask = (task == 1)
        if expr_mask.any():
            outputs[expr_mask] = self.expr_head(x[expr_mask])
        
        return outputs

def train():
    # Initialize dataset
    dataset = MathDataset(
        char_dir='dataset/chars',
        expr_dir='dataset/expressions'
    )
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Model, loss, and optimizer
    model = EnhancedMathModel().to(device)
    char_criterion = nn.CrossEntropyLoss()
    expr_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(25):
        model.train()
        for images, labels, tasks in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            tasks = tasks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, tasks)
            
            # Calculate loss
            loss = torch.zeros(1, device=device)
            char_mask = (tasks == 0)
            expr_mask = (tasks == 1)
            
            if char_mask.any():
                loss += char_criterion(outputs[char_mask], labels[char_mask].long())
            if expr_mask.any():
                loss += expr_criterion(outputs[expr_mask].squeeze(), labels[expr_mask].float())
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            char_correct = char_total = expr_error = expr_count = 0
            
            for images, labels, tasks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                tasks = tasks.to(device)
                outputs = model(images, tasks)
                
                # Character accuracy
                char_mask = (tasks == 0)
                if char_mask.any():
                    _, preds = torch.max(outputs[char_mask], 1)
                    char_total += char_mask.sum().item()
                    char_correct += (preds == labels[char_mask]).sum().item()
                
                # Expression MAE
                expr_mask = (tasks == 1)
                if expr_mask.any():
                    expr_count += expr_mask.sum().item()
                    expr_error += torch.abs(outputs[expr_mask].squeeze() - labels[expr_mask]).sum().item()
            
            print(f'Epoch [{epoch+1}/25], '
                  f'Char Acc: {100*char_correct/max(1, char_total):.2f}%, '
                  f'Expr MAE: {expr_error/max(1, expr_count):.2f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/enhanced_math_model.pth')

if __name__ == '__main__':
    train()
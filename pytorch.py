import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Celeb dataset
attr_file_path = '/Users/aatithayapaliwal/Desktop/pytorch_project/list_attr_celeba.txt'

# column names
column_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

df = pd.read_csv(attr_file_path, sep='\s+', header=None, names=column_names, dtype=str)

num_classes = 10
images_per_class = 1000
min_images_needed = 10000

class_columns = column_names[1:]  

np.random.seed(0)
selected_columns = np.random.choice(class_columns, num_classes, replace=False)

df_filtered = pd.DataFrame()

for col in selected_columns:
    class_df = df[df[col] == '1']
    
    if len(class_df) > images_per_class:
        class_df = class_df.sample(images_per_class)
    elif len(class_df) > 0:
        print(f"Only {len(class_df)} images available for class '{col}', using all available images.")
    
    df_filtered = pd.concat([df_filtered, class_df])

df_filtered = df_filtered.drop_duplicates()

if len(df_filtered) < min_images_needed:
    additional_needed = min_images_needed - len(df_filtered)
    additional_df = df.sample(additional_needed)
    df_filtered = pd.concat([df_filtered, additional_df])

df_filtered = df_filtered.drop_duplicates()

df_filtered.to_csv('filtered_dataset.csv', index=False)

print(f"Filtered dataset saved to 'filtered_dataset.csv' with {len(df_filtered)} images.")

class CelebADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        labels = torch.tensor([int(self.df.iloc[idx, col]) for col in range(1, num_classes+1)], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

img_dir = '/Users/aatithayapaliwal/Desktop/pytorch_project/img_align_celeba'
dataset = CelebADataset(df=df_filtered, img_dir=img_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class PretrainedCNN(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedCNN, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = PretrainedCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0001)  

def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

train_model(model, criterion, optimizer, train_loader, num_epochs=20)

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            labels = torch.argmax(labels, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    
    return y_true, y_pred

y_true, y_pred = evaluate_model(model, test_loader)

conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=selected_columns, yticklabels=selected_columns)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

torch.save(model.state_dict(), 'final_model.pth')
print("Model saved as 'final_model.pth'")

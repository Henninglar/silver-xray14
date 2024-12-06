"""
The main EfficientNet model implementation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import DenseNet169_Weights, DenseNet121_Weights, EfficientNet_B3_Weights
from Components.read_data import ChestXrayDataSet, load_labels
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Paths and configurations
TRAIN_IMAGE_LIST = './Chest-X-rays14-Dataset/train_val_list.txt'
TEST_IMAGE_LIST = './Chest-X-rays14-Dataset/test_list.txt'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Pleural_Thickening', 'Hernia']
LABELS = load_labels("./Chest-X-rays14-Dataset/labels.csv", CLASS_NAMES)
Amount = 25000  # Amount of images to load
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SkipTrain = False
EPOCHS = 10
torch.manual_seed(42)

def main():
    cudnn.benchmark = True  # PyTorch will use the cuDNN autotuner to find the most efficient algorithms for the current hardware and input size
    global SkipTrain
    # Initialize and load the ResNet model
    model = EfficientNetB3(N_CLASSES).cuda()
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile("Models/EfficientNet.pth"):
        print("=> loading pretrained weights")
        checkpoint = torch.load("Models/EfficientNet.pth")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        SkipTrain = True

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss for multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data augmentation and normalization for training
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Prepare the datasets
    train_dataset = ChestXrayDataSet(
                                     TrainingImages=TRAIN_IMAGE_LIST,
                                     labels_dict=LABELS,
                                     amount=Amount,
                                     transform=transforms.Compose([
                                         transforms.Resize(256), #It supports 300x300 but i used 224 to keep it fair.
                                         transforms.RandomCrop(224), #It supports 300x300 but i used 224 to keep it fair.
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = ChestXrayDataSet(
                                    ValidationImages=TEST_IMAGE_LIST,
                                    labels_dict=LABELS,
                                    amount=Amount,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train is set to: {SkipTrain}")
    if SkipTrain:
        print("Skipping training. Starting evaluation directly.")
        validate(test_loader, model)
        return

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train(train_loader, model, criterion, optimizer)
        validate(test_loader, model)

    save_final_model(model, optimizer, EPOCHS)

def save_final_model(model, optimizer, epoch):
    """Save the final model checkpoint."""
    checkpoint_path = "Models/EfficientNet.pth"  # Save the final model
    state = {
        'epoch': epoch,  # Save the final epoch
        'state_dict': model.state_dict(),  # Model weights
        'optimizer': optimizer.state_dict()  # Optimizer state
    }
    torch.save(state, checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")


def train(train_loader, model, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()  # Set model to training mode
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Training Loss: {running_loss / len(train_loader):.4f}")


def validate(test_loader, model):
    """Validate the model on the test set."""
    model.eval()  # Set model to evaluation mode
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.cuda()
            targets = targets.cuda()

            outputs = model(images)
            gt = torch.cat((gt, targets), 0)
            pred = torch.cat((pred, outputs.data), 0)

    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    # Compute AUROC
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print(f'The average AUROC is {AUROC_avg:.3f}')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]:.3f}')

    pred_binarized = (pred_np >= 0.3).astype(int)
    # Flatten ground truth and predictions for overall precision, recall, and F1-score
    gt_flat = gt_np.ravel()
    pred_flat = pred_binarized.ravel()
    # Compute overall precision, recall, and F1-score
    avg_precision = precision_score(gt_flat, pred_flat, zero_division=0)
    avg_recall = recall_score(gt_flat, pred_flat, zero_division=0)
    avg_f1_score = f1_score(gt_flat, pred_flat, zero_division=0)
    # Print overall metrics
    print("\nOverall Metrics Across All Classes:")
    print(f"  Average Precision: {avg_precision:.3f}")
    print(f"  Average Recall: {avg_recall:.3f}")
    print(f"  Average F1 Score: {avg_f1_score:.3f}")
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class EfficientNetB3(nn.Module):
    def __init__(self, out_size):
        super(EfficientNetB3, self).__init__()
        # Load pre-trained DenseNet121 model
        self.efficientnet_b3 = torchvision.models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        num_ftrs = self.efficientnet_b3.classifier[1].in_features
        self.efficientnet_b3.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet_b3(x)


if __name__ == '__main__':
    main()

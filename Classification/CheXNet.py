"""
The main CheXNet model implementation with fine tuning and run on our augmentation. Added CAM heatmap
"""

import os
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from Components.read_data import ChestXrayDataSet, load_labels, ChestXraySpecificImagesDataSet
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Paths and configurations
TRAIN_IMAGE_LIST = './Chest-X-rays14-Dataset/train_val_list.txt'
TEST_IMAGE_LIST = './Chest-X-rays14-Dataset/test_list.txt'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Pleural_Thickening', 'Hernia']
LABELS = load_labels("./Chest-X-rays14-Dataset/labels.csv",CLASS_NAMES)
Amount = 25000 #Amount of images to load
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SkipTrain = False
EPOCHS = 10
CHECKPOINT_PATH = './Models/Chexnet_Default.pth'  # Path for pretrained ChexNet model.
torch.manual_seed(42)




normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def denormalize(tensor):
    """Denormalizes an image tensor using the same mean and std as the normalize transform."""
    mean = torch.tensor(normalize.mean).view(3, 1, 1).cuda()
    std = torch.tensor(normalize.std).view(3, 1, 1).cuda()
    return tensor * std + mean


def main():
    cudnn.benchmark = True #PyTorch will use the cuDNN autotuner to find the most efficient algorithms for the current hardware and input size

    # Initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # Load the pretrained weights from the checkpoint
    if os.path.isfile(CHECKPOINT_PATH):
        print("=> loading checkpoint from '{}'".format(CHECKPOINT_PATH))
        checkpoint = torch.load(CHECKPOINT_PATH,weights_only=True)
        model.load_state_dict(checkpoint['state_dict'], strict=False)  # Load weights without strict matching
        print("=> loaded checkpoint '{}'" .format(CHECKPOINT_PATH))
    else:
        print("=> no checkpoint found at '{}'".format(CHECKPOINT_PATH))

    if os.path.isfile("./Models/Finetuned_ChexNet.pth"):
        global SkipTrain
        print("=> loading pretrained weights")
        checkpoint = torch.load("./Models/Finetuned_ChexNet.pth")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        SkipTrain = True

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data augmentation and normalization for training
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Init is executed and a list of image_names is created.
    # The transformation options and label dictionary are prepared and stored for use by the trainloader
    train_dataset = ChestXrayDataSet(
                                     TrainingImages=TRAIN_IMAGE_LIST,
                                    labels_dict=LABELS,
                                    amount=Amount,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
    # __getitem__ in read_data is called and will fetch the image from train_dataset, apply transformations, and return both the image and its corresponding label.
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

    # Retrives specific images that we can use to compare against the Grounnd truth.
    specific_dataset = ChestXraySpecificImagesDataSet(
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    specific_loader = DataLoader(dataset=specific_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                 pin_memory=True)

    print(f"Train is set to: {SkipTrain}")
    if SkipTrain: #I.E if train is NOT true.
        print("Skipping training. Starting evaluation directly.")
        validate(test_loader, model)
        # Provide CAM heatmap of specific images.
        example_images = next(iter(specific_loader))
        generate_cam_visualizations(model, example_images)
        return

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train(train_loader, model, criterion, optimizer)
        validate(test_loader, model)

    save_final_model(model, optimizer, EPOCHS)

def save_final_model(model, optimizer, epoch):
    """Save the final model checkpoint."""
    checkpoint_path = "./Models/Finetuned_ChexNet.pth"  # Save the final model
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

def generate_cam_visualizations(model, images):
    """Generate CAM visualizations for a batch of images."""
    model.eval()  # Ensure model is in evaluation mode
    cam_extractor = SmoothGradCAMpp(model, target_layer="module.densenet121.features")

    images = images.cuda()
    outputs = model(images)  # Run forward pass for the batch

    for idx in range(images.size(0)):
        # Get the highest predicted class for each individual image
        class_idx = outputs[idx].argmax().item()
        activation_map = cam_extractor(class_idx, outputs[idx])[0]

        # Display original image and CAM overlay
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        denorm_image = denormalize(images[idx].unsqueeze(0)).squeeze().cpu().detach().permute(1, 2, 0).numpy()
        axes[0].imshow(denorm_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Generate overlay for each CAM
        for name, cam in zip(cam_extractor.target_names, activation_map):
            # Ensure activation map is single-channel
            result = overlay_mask(
                to_pil_image(denorm_image),      # Convert original image to PIL
                to_pil_image(cam.squeeze(0), mode='F'),  # Convert CAM to single-channel PIL
                alpha=0.5
            )
            axes[1].imshow(result)

        axes[1].set_title("CAM Heatmap Overlay")
        axes[1].axis("off")
        plt.show()

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()

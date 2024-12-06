import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
from DensNet121 import DenseNet121
from Resnet import ResNet50
from EfficientNet import EfficientNetB3

from Components.read_data import ChestXrayDataSet, load_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
N_CLASSES = 14
Amount = 25000
BATCH_SIZE = 64
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Pleural_Thickening', 'Hernia']

LABELS = load_labels("./Chest-X-rays14-Dataset/labels.csv", CLASS_NAMES)
TEST_IMAGE_LIST = './Chest-X-rays14-Dataset/test_list.txt'


# Helper functions
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


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

    # Compute AUROC
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print(f'The average AUROC is {AUROC_avg:.3f}')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]:.3f}')
    return pred, gt
# Main execution block
if __name__ == '__main__':
    # Initialize models
    densenet_model = DenseNet121(N_CLASSES).cuda()
    densenet_model = torch.nn.DataParallel(densenet_model).cuda()
    resnet_model = ResNet50(N_CLASSES).cuda()
    resnet_model = torch.nn.DataParallel(resnet_model).cuda()
    efficientnet_model = EfficientNetB3(N_CLASSES).cuda()
    efficientnet_model = torch.nn.DataParallel(efficientnet_model).cuda()


    # Load pretrained weights
    if os.path.isfile("Models/DensNet.pth"):
        print("=> Loading pretrained weights...")
        checkpoint_densenet = torch.load('Models/DensNet.pth')
        densenet_model.load_state_dict(checkpoint_densenet['state_dict'], strict=False)

    if os.path.isfile("./Models/Final_ResNet_model.pth"):
        print("=> loading pretrained weights")
        checkpoint = torch.load("./Models/Final_ResNet_model.pth")
        resnet_model.load_state_dict(checkpoint['state_dict'], strict=False)

    if os.path.isfile("Models/EfficientNet.pth"):
        print("=> loading pretrained weights")
        checkpoint = torch.load("Models/EfficientNet.pth")
        efficientnet_model.load_state_dict(checkpoint['state_dict'], strict=False)
        SkipTrain = True


    # Normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Load test data
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

    # Validate the model
    pred_densenet, gt_densenet = validate(test_loader, densenet_model)
    pred_resnet, gt_resnet = validate(test_loader, resnet_model)
    pred_efficientnet, gt_efficientnet = validate(test_loader, efficientnet_model)


    weights = [0.4, 0.2, 0.4]  # Example weights for DenseNet, ResNet, EfficientNet
    final_pred_weighted = (weights[0] * pred_densenet +
                           weights[1] * pred_resnet +
                           weights[2] * pred_efficientnet)

    # Ensemble Method 2: Simple Averaging
    final_pred_avg = (pred_densenet + pred_resnet + pred_efficientnet) / 3

    # Ensemble Method 3: Majority Voting

    # Compute AUROC for each ensemble method
    AUROCs_weighted = compute_AUCs(gt_densenet, final_pred_weighted)
    AUROCs_avg = compute_AUCs(gt_densenet, final_pred_avg)

    # Binarize predictions for both ensemble methods
    final_pred_avg_binarized = (final_pred_avg >= 0.3).int()
    final_pred_weighted_binarized = (final_pred_weighted >= 0.3).int()

    # Flatten the ground truth and predictions for overall metric calculations
    gt_flat = gt_densenet.cpu().numpy().ravel()  # Flatten ground truth
    weighted_pred_flat = final_pred_weighted_binarized.cpu().numpy().ravel()  # Flatten weighted predictions
    avg_pred_flat = final_pred_avg_binarized.cpu().numpy().ravel()  # Flatten average predictions

    # Calculate overall precision, recall, and F1-score for weighted averaging
    weighted_precision = precision_score(gt_flat, weighted_pred_flat, zero_division=0)
    weighted_recall = recall_score(gt_flat, weighted_pred_flat, zero_division=0)
    weighted_f1 = f1_score(gt_flat, weighted_pred_flat, zero_division=0)

    # Calculate overall precision, recall, and F1-score for simple averaging
    avg_precision = precision_score(gt_flat, avg_pred_flat, zero_division=0)
    avg_recall = recall_score(gt_flat, avg_pred_flat, zero_division=0)
    avg_f1 = f1_score(gt_flat, avg_pred_flat, zero_division=0)

    # Print AUROC results and overall metrics for each method
    print("\nWeighted Averaging Ensemble Metrics:")
    print(f'The average AUROC for Weighted Averaging is {np.mean(AUROCs_weighted):.3f}')
    print(f"  Average Precision: {weighted_precision:.3f}")
    print(f"  Average Recall: {weighted_recall:.3f}")
    print(f"  Average F1 Score: {weighted_f1:.3f}")

    print("\nSimple Averaging Ensemble Metrics:")
    print(f'The average AUROC for Simple Averaging is {np.mean(AUROCs_avg):.3f}')
    print(f"  Average Precision: {avg_precision:.3f}")
    print(f"  Average Recall: {avg_recall:.3f}")
    print(f"  Average F1 Score: {avg_f1:.3f}")
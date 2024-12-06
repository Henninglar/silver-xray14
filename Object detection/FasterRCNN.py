import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torch_snippets.torch_loader import Report
from matplotlib import patches
import os
import torchvision
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

ROOT_DIRECTORY = os.getcwd()
TRAINING_DIRECTORY = './Chest-X-rays14-Dataset/ImagesWithAnnotationsBackup/Training'
VALIDATION_DIRECTORY = './Chest-X-rays14-Dataset/ImagesWithAnnotationsBackup/Validation'
ANNOTATION_DIRECTORY = "./Chest-X-rays14-Dataset"
BATCH_SIZE = 16
# Specifies root, training, validation, and annotation directories. Sets batch size for data loading.

"""# Dataset Prep"""

# Load bounding box annotations
bbox_df = pd.read_csv(os.path.join(ROOT_DIRECTORY, ANNOTATION_DIRECTORY, 'BBox_List_2017.csv')) #READS CSV
bbox_df = bbox_df.iloc[:, :6] # Takes first 6 rows
bbox_df.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h'] #Gives rows better names

# WE DO THIS TO GIVE EACH CLASS A ID LIKE 1,2,3 etc
class_mapping = {label: i + 1 for i, label in enumerate(bbox_df['Finding Label'].unique())}  # Background is 0

# Reverse mapping: Class ID -> Finding Label
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

#THIS LINE APPLIES THE ID TO THE dataframe
bbox_df['class_id'] = bbox_df['Finding Label'].map(class_mapping)

# Dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, bbox_df, transform=None):
        self.root_dir = root_dir #Directory
        self.bbox_df = bbox_df #DataFrame containing bounding box annotations.
        self.transform = transform #transform options
        self.image_files = bbox_df['Image Index'].unique() #A list of unique image file names, derived from the Image Index column of the DataFrame.

    def __len__(self):
        return len(self.image_files) #Returns the total number of unique images in the dataset.

    def __getitem__(self, idx): #Retrieves an image and its corresponding bounding box annotations AND LABEL for a given index
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # Get all bounding boxes and labels for this image
        boxes = []
        labels = []
        for _, row in self.bbox_df[self.bbox_df['Image Index'] == img_name].iterrows():
            x_min = int(row['x'])
            y_min = int(row['y'])
            x_max = int(x_min + row['w'])
            y_max = int(y_min + row['h'])
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(row['class_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.as_tensor([idx])

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels, "image_id":image_id}
        return image, target

# Used to return A tuple containing:
# A list of image tensors.
# A list of target dictionaries.
def collate_fn(batch):
    return tuple(zip(*batch))

# Define transformations
transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to Tensor first
])

# Load datasets
# bbox_df=bbox_df[bbox_df['Ima.. simply ensures that only data from the correct set is loaded.
train_dataset = ChestXrayDataset(root_dir=os.path.join(ROOT_DIRECTORY, TRAINING_DIRECTORY), bbox_df=bbox_df[bbox_df['Image Index'].isin(os.listdir(os.path.join(ROOT_DIRECTORY, TRAINING_DIRECTORY)))], transform=transform)
val_dataset = ChestXrayDataset(root_dir=os.path.join(ROOT_DIRECTORY, VALIDATION_DIRECTORY), bbox_df=bbox_df[bbox_df['Image Index'].isin(os.listdir(os.path.join(ROOT_DIRECTORY, VALIDATION_DIRECTORY)))], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  # Batch size of 2
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def get_object_detection_model(num_classes,feature_extraction = True):
    # Load the pretrained faster r-cnn model.
    WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=WEIGHTS)
    # If True, the pre-trained weights will be frozen.
    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False
    # Replace the original 91 class top layer with a new layer
    # tailored for num_classes.
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   num_classes)
    return model

def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y

def train_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses

@torch.no_grad()
def validate_batch(batch, model, device):
    model.eval()
    X, y = unbatch(batch, device = device)
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    return loss, losses

def train_fasterrcnn(model,optimizer,n_epochs,train_loader,test_loader = None, log = None, keys = None,device = device):
    if log is None:
        log = Report(n_epochs)
    if keys is None:
        # FasterRCNN loss names.
        keys = ["loss_classifier",
                   "loss_box_reg",
                "loss_objectness",
               "loss_rpn_box_reg"]
    model.to(device)
    for epoch in range(n_epochs):
        N = len(train_loader)
        for ix, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model,
                                  optimizer, device)
            # Record the current train loss.
            pos = epoch + (ix + 1) / N
            log.record(pos = pos, trn_loss = loss.item(),
                       end = "\r")
        if test_loader is not None:
            N = len(test_loader)
            for ix, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model, device)
                # Record the current validation loss.
                pos = epoch + (ix + 1) / N
                log.record(pos = pos, val_loss = loss.item(),
                           end = "\r")
    log.report_avgs(epoch + 1)
    return log

""" used to train model.
NUM_EPOCHS = 50
        
model = get_object_detection_model(num_classes = 9,feature_extraction = True) #If i swap this to false training time goes to 18h+ due to updated ALL layers. Keep at TRUE!.
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
log = train_fasterrcnn(model = model,
                       optimizer = optimizer, 
                    n_epochs = NUM_EPOCHS,
                     train_loader = train_loader, 
                        test_loader = val_loader,
                     log = None, keys = None,
                             device = device)

#torch.save(model.state_dict(), "Models/faster_rcnn_chest_xray_multiclass.pth")
"""

@torch.no_grad()
def predict_batch(batch, model, device):
    model.to(device)
    model.eval()
    X, _ = unbatch(batch, device = device)
    predictions = model(X)
    return predictions

def decode_prediction(prediction,score_threshold = 0.2,nms_iou_threshold = 0.2):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    # Remove any low-score predictions.
    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    # Remove any overlapping bounding boxes using NMS.
    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(boxes = boxes, scores = scores,
                                iou_threshold = nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    return (boxes.cpu().numpy(),labels.cpu().numpy(),scores.cpu().numpy())


## MAIN PART STARTS HERE.

num_classes = 9 #8 + background = 9
model_50_epochs = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model_50_epochs.load_state_dict(torch.load("./Models/faster_rcnn_chest_xray_multiclass_epochs_50.pth", weights_only=True))


"""Prediction based on Training Dataset"""
map_metric = MeanAveragePrecision()
model = model_50_epochs
# For train loader
for i, (image, target) in enumerate(train_loader):
    with torch.no_grad():
        predictions = predict_batch((image, target), model, device)
        # Decode and prepare predictions for metric calculation
        decoded_predictions = [decode_prediction(pred) for pred in predictions]

        # Format predictions and targets for TorchMetrics
        processed_preds = [{"boxes": torch.tensor(pred[0], device=device),
                            "scores": torch.tensor(pred[2], device=device),
                            "labels": torch.tensor(pred[1], device=device)} for pred in decoded_predictions]

        processed_targets = [{"boxes": target[i]["boxes"].to(device),
                              "labels": target[i]["labels"].to(device)} for i in range(len(target))]

        # Update the mAP metric with predictions and targets for this batch
        map_metric.update(processed_preds, processed_targets)

# Compute and print the mAP@0.5 for the train set
map_train_results = map_metric.compute()
print(f"TRAIN STATS: mAP@0.5: {map_train_results['map_50']:.4f}, Precision: {map_train_results['map']:.4f}, Recall: {map_train_results['mar_100']:.4f}")


"""Prediction based on Validation Dataset"""
map_metric = MeanAveragePrecision(iou_thresholds=[0.5],class_metrics=True)
model = model_50_epochs
# For validation loader
for i, (image, target) in enumerate(val_loader):
    with torch.no_grad():
        predictions = predict_batch((image, target), model, device)
        decoded_predictions = [decode_prediction(pred) for pred in predictions]

        processed_preds = [{"boxes": torch.tensor(pred[0], device=device),
                            "scores": torch.tensor(pred[2], device=device),
                            "labels": torch.tensor(pred[1], device=device)} for pred in decoded_predictions]

        processed_targets = [{"boxes": target[i]["boxes"].to(device),
                              "labels": target[i]["labels"].to(device)} for i in range(len(target))]

        map_metric.update(processed_preds, processed_targets)

# Compute and print the mAP@0.5 for the validation set
map_valid_results = map_metric.compute()
# Display per-class mAP

classmap = {
    1: 'Atelectasis',
    2: 'Cardiomegaly',
    3: 'Effusion',
    4: 'Infiltrate',
    5: 'Mass',
    6: 'Nodule',
    7: 'Pneumonia',
    8: 'Pneumothorax'
}


print(f"VALID STATS: mAP@0.5: {map_valid_results['map_50']:.4f}, Precision: {map_valid_results['map']:.4f}, Recall: {map_valid_results['mar_100']:.4f}")
print("\nPer-Class mAP@0.5:")
if 'map_per_class' in map_valid_results:
    for class_id in range(len(map_valid_results['map_per_class'])):
        class_map = map_valid_results['map_per_class'][class_id]
        class_name = classmap.get(class_id+1)
        print(f" - {class_name} (Class {class_id}): mAP@0.5 = {class_map:.4f}")



"""Plotting boundary boxes on the images in validation set."""
model = model_50_epochs
for i, (image, target) in enumerate(val_loader):
    model.eval()
    with torch.no_grad():
        predictions = predict_batch((image, target), model, device)

    # Loop over each individual image in the batch
    for j in range(len(image)):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the ground truth with bounding boxes
        ax[0].imshow(image[j].permute(1, 2, 0).cpu().numpy())
        for box in target[j]['boxes']:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            label = reverse_class_mapping[target[j]['labels'][0].item()]
            ax[0].add_patch(rect)
            ax[0].text(box[0], box[1] - 5, label, color='r')
        ax[0].set_title('Ground Truth')

        # Plot the predictions with bounding boxes
        ax[1].imshow(image[j].permute(1, 2, 0).cpu().numpy())
        boxes, labels, scores = decode_prediction(predictions[j])
        for k, box in enumerate(boxes):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor='b',
                facecolor='none'
            )
            ax[1].add_patch(rect)
            ax[1].text(box[0], box[1] - 5, f"{reverse_class_mapping[labels[k]]}: {scores[k]:.2f}", color='b')
        ax[1].set_title('Prediction')

        plt.show()

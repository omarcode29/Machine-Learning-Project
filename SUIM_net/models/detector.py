import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define the transform for the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
trainset = torchvision.datasets.CocoDetection(root='./data', annFile='./annotations/train.json', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Load the pre-trained model and freeze all layers except the last one
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.roi_heads.box_predictor.cls_score.weight.requires_grad = True
model.roi_heads.box_predictor.bbox_pred.weight.requires_grad = True

# Set the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropy

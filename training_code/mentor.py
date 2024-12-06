import os
import sys
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader_cam import datasetLoader
from tqdm import tqdm
import shutil
import cv2
sys.path.append("../")

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=20)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=False, default= '',type=str)
parser.add_argument('-datasetPath', required=False, default= '',type=str)
parser.add_argument('-outputPath', required=False, default= '',type=str)
parser.add_argument('-heatmaps', required=False, default= '',type=str)
parser.add_argument('-alpha', required=False, default=1.0,type=float)
parser.add_argument('-network', default= 'resnet',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-loss_fn', default= 'mse',type=str)
parser.add_argument('-device', default= 'cuda',type=str)

args = parser.parse_args()
device = torch.device(args.device)

print(args)

hmap_loss_fn = args.loss_fn

print(args)
print("Loss function used:",hmap_loss_fn)

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output
  return hook

# Definition of model architecture
if args.network == "mentor-resnet152":
    im_size = 224
    map_size = 7
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    weights = torch.load("path/to/final_model.pth",
        map_location = device)
    weights = weights['state_dict']
    weights.pop("segmentation_head.0.weight")
    weights.pop("segmentation_head.0.bias")
    for k in list(weights.keys()):
        if k.startswith('decoder'):
            del weights[k]
    for k in list(weights.keys()):
        if k.startswith('encoder'):
            new_key = k.replace('encoder.', '')
            weights[new_key] = weights.pop(k)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)

elif args.network == "mentor-inceptionv4":
    import timm
    model = timm.create_model('inception_v4', pretrained=True, num_classes=2)
    im_size = 224
    map_size = 8
    weights = torch.load("path/to/final_model.pth",
        map_location = device)
    weights = weights['state_dict']
    weights.pop("segmentation_head.0.weight")
    weights.pop("segmentation_head.0.bias")
    for k in list(weights.keys()):
        if k.startswith('decoder'):
            del weights[k]

    #print(len(weights.keys()))
    # renaming the keys
    for k in list(weights.keys()):
        if k.startswith('encoder'):
            new_key = k.replace('encoder.', '')
            weights[new_key] = weights.pop(k)

    model.load_state_dict(weights, strict=False)
    print(model)
    model = model.to(device)


elif args.network == "mentor-efficientnet":
    im_size = 224
    map_size = 8
    model = models.efficientnet_b7(pretrained=True)
    # original weights
    original = torch.load("efficientnet_b7_lukemelas-c5b4e57e.pth", map_location = device)
    original.pop("classifier.1.weight")
    original.pop("classifier.1.bias")
    # mentor weights
    weights = torch.load("path/to/final_model.pth",
        map_location = device)
    weights = weights['state_dict']
    weights.pop("segmentation_head.0.weight")
    weights.pop("segmentation_head.0.bias")
    #sys.exit()
    # deleting all decoder weights
    for k in list(weights.keys()):
        if k.startswith('decoder'):
            del weights[k]
        # deleting all decoder weights
    print(len(weights.keys()))
    # renaming weights
    for i, k in enumerate(list(original.keys())):
        value = list(weights.values())[i]
        original[k] = value
    #print(weights.keys())
    #sys.exit()
    model.load_state_dict(original, strict=False)
    #num_ftrs = 1280
    model.classifier = nn.Linear(2560, args.nClasses)
    model = model.to(device)


else:
    print("Invalid...exiting")
    sys.exit()

#print(model)
#sys.exit()

# Create destination folder
os.makedirs(args.outputPath,exist_ok=True)

# Creation of Log folder: used to save the trained model
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)


# Creation of result folder: used to save the performance of trained model on the test set
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)


if os.path.exists(result_path + "/model_Loss.jpg") and os.path.exists(log_path + "/final_model.pth"):
    print("Training already completed for this setup, exiting...")
    sys.exit()

if os.path.exists(os.path.join(args.outputPath , 'sigmoid')):
    print("Deleting any previous tests...")
    shutil.rmtree(os.path.join(args.outputPath , 'sigmoid'))


class_assgn = {'normal':0,'abnormal':1}

# Dataloader for train and test data
dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',c2i=class_assgn,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataloader = {'train': dl, 'test':test}


# Description of hyperparameters
lr = 0.005
#lr = 0.001
solver = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6, momentum=0.9)

#solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

criterion = nn.CrossEntropyLoss()
criterion_hmap = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
criterion_hmap_ce = nn.CrossEntropyLoss()
criterion_hmap_mse = nn.MSELoss()
criterion_hmap_l1 = nn.L1Loss()
#criterion_hmap_ssim = SSIMLoss(n_channels=1,window_size=map_size).to(device)

# File for logging the training process
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

    

#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################


train_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
alpha = args.alpha
print("Alpha value:",alpha)
if alpha == 1.0:
    print("Only using classification loss")
else:
    print("Using classification loss and heatmap loss")
train_step = 0
val_step = 0
for epoch in range(args.nEpochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
            if args.network == "xception":
                model.model.train()
        else:
            model.eval()
            if args.network == "xception":
                model.model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(train):
            for batch_idx, (data, cls, imageName, hmap) in enumerate(tqdm(dataloader[phase])):

                # Data and ground truth
                data = data.to(device)
                cls = cls.to(device)
                hmap = hmap.to(device)
                
                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                class_loss = criterion(outputs, cls)                       

                # Running model over data
                if phase == 'train' and alpha != 1.0:
                    if args.network == "resnet152":
                        features = activation['features']
                        params = list(model.fc.parameters())[0]
                    elif args.network == "inceptionv4":
                        features = activation['features']
                        params = list(model.last_linear_two.parameters())[0]
                    elif args.network == "efficientnet":
                        features = activation['features']
                        params = list(model.classifier.parameters())[0]
                    else:
                        print("INVALID ARCHITECTURE:",args.network)
                        sys.exit()

                    bz, nc, h, w = features.shape

                    beforeDot =  features.reshape((bz, nc, h*w))
                    cams = []
                    for ids,bd in enumerate(beforeDot):
                        weight = params[cls[ids]]
                        cam = torch.matmul(weight, bd)
                        cam_img = cam.reshape(h, w)
                        cam_img = cam_img - torch.min(cam_img)
                        if torch.max(cam_img) != 0:
                            cam_img = cam_img / torch.max(cam_img)
                        cams.append(cam_img)
                    cams = torch.stack(cams)
                    if hmap_loss_fn == 'mse': # 1
                        hmap_loss = (criterion_hmap_mse(cams,hmap))
                    else:
                        print("Invalid loss selected... Exiting.")
                        sys.exit()
                else:
                    hmap_loss = 0
            
                # Optimization of weights for training data
                if phase == 'train':
                    if alpha != 1.0:
                        loss = (alpha)*(class_loss) + (1-alpha)*(hmap_loss)
                    else:
                        loss = class_loss
                    train_step += 1
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    loss = class_loss
                    val_step += 1
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)

                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            # if args.network != "xception":
            lr_sched.step()
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy =accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,'final_model.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': solver.state_dict(),
    }
    with open(os.path.join(log_path,'model_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(states, os.path.join(log_path,'current_model.pth'))



# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path,'model_Loss.jpg'))


# Evaluation of test set utilizing the trained model
obvResult = evaluation()
errorIndex, predictScore, threshold = obvResult.get_result(args.network, testImgNames, testTrueLabels, testPredScores, result_path)



import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    device = torch.device('cuda')
    print(device)
    parser.add_argument('-imageFolder', default='',type=str)
    parser.add_argument('-modelPath',  default='',type=str)
    parser.add_argument('-csv',  default="",type=str)
    parser.add_argument('-output_scores',  default="",type=str)
    parser.add_argument('-network',  default="resnet",type=str)
    args = parser.parse_args()

    os.makedirs(args.output_scores,exist_ok=True)

    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath, map_location = device)

    if args.network == "resnet152" or "mentor-resnet152":
        im_size = 224
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    elif args.network == "inceptionv4":
        import timm
        #model = timm.create_model('inception_v4', pretrained=True, num_classes=2)
        model = timm.create_model('inception_v4', pretrained=True, num_classes=256)
        model.last_act = nn.ReLU(inplace=True)
        model.last_linear_two = nn.Linear(in_features=256, out_features=2, bias=True)
        model = model.to(device)

        #im_size = 224

    elif args.network == "efficientnet":
        im_size = 224
        model = models.efficientnet_b7(pretrained=True)
        #num_ftrs = 1280
        model.classifier = nn.Linear(2560, 2)


    else:
        print("Invalid model selection...exiting")
        sys.exit()
    
    model.load_state_dict(weights['state_dict'])
    model = model.to(device)
    model.eval()

    if args.network == "xception":
        # Transformation specified for the pre-processing
        transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
    elif args.network == "inceptionv4":
        # Letting Hugging Face Pre-process
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)

    else:
        # Transformation specified for the pre-processing
        transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    sigmoid = nn.Sigmoid()

    #File,PAScore,TRUE,Model,Training_Type,Dataset

    # imageFiles = glob.glob(os.path.join(args.imageFolder,'*.jpg'))
    imagesScores = []
    for file in glob.glob(args.csv):
        #imagesScores=[]
        Dataset = file
        Dataset = Dataset[:-3]
        imageCSV = open(file,"r")
        #if "celeba-hq" in Dataset:
        #    TRUE = 0
        #elif "ffhq" in Dataset:
        #    TRUE = 0
        #else:
        #    TRUE = 1
        with open(file) as _file:
            for entry in _file:
                #print("--------------")
                TRUE = entry.split(",")[1]
                #print("--------------")
                tokens = entry.split(",")
                if tokens[0] != 'test':
                    continue
                upd_name = tokens[-1].replace("\n","")
                # upd_name = upd_name.replace(upd_name.split(".")[-1],"png")
                imgFile = args.imageFolder + upd_name

                # Read the image
                image = Image.open(imgFile).convert('RGB')
                # Image transformation
                tranformImage = transform(image)
                image.close()

                ## START COMMENTING HERE

                tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
                tranformImage = tranformImage.to(device)

                # Output from single binary CNN model
                with torch.no_grad():
                    output = model(tranformImage)
                #predicted_class = torch.argmax(output).item() 
                PAScore = sigmoid(output).detach().cpu().numpy()[:, 1]
                #imagesScores.append([imgFile, predicted_class, TRUE, Dataset])
                imagesScores.append([imgFile, PAScore[0], TRUE, Dataset])

    # Writing the scores in the csv file
    save_directory = args.output_scores + "all.csv"
    with open(save_directory,'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)

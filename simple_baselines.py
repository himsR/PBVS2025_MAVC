import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
from sklearn.preprocessing import LabelBinarizer
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
from torchvision.transforms import v2
import numpy as np

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
import os
from focal_loss import FocalLoss 


def parse_args():
    """
    Parse input arguments
    Returns
    -------
    args : object
        Parsed args
    """
    h = {
        "program": "Simple Baselines training",
        "train_folder": "Path to training data folder.",
        "val_folder": "Path to val data folder",
        "batch_size": "Number of images to load per batch. Set according to your PC GPU memory available. If you get "
                      "out-of-memory errors, lower the value. defaults to 64",
        "epochs": "How many epochs to train for. Once every training image has been shown to the CNN once, an epoch "
                  "has passed. Defaults to 15",
        "test_folder": "Path to test data folder",
        "num_workers": "Number of workers to load in batches of data. Change according to GPU usage",
        "test_only": "Set to true if you want to test a loaded model. Make sure to pass in model path",
        "eval_only": "Set to true if you want to test a loaded model. Make sure to pass in model path",
        "model_path": "Path to your model",
        "learning_rate": "The learning rate of your model. Tune it if it's overfitting or not learning enough",
        "resume_from_weight": "resume training from a weight"}
    parser = argparse.ArgumentParser(description=h['program'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_folder', help=h["train_folder"], type=str)
    parser.add_argument('--val_folder', help=h["val_folder"], type=str)
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=256)
    parser.add_argument('--epochs', help=h["epochs"], type=int, default=5)
    parser.add_argument('--test_folder', help=h["test_folder"], type=str)
    parser.add_argument('--num_workers', help=h["num_workers"], type=int, default=5)
    parser.add_argument('--test_only', help=h["test_only"], type=bool, default=False)
    parser.add_argument('--eval_only', help=h["eval_only"], type=bool, default=False)
    parser.add_argument('--model_path', help=h["num_workers"], type=str),
    parser.add_argument('--resume_from_weight', help=h["resume_from_weight"], type=str, default=""),
    parser.add_argument('--learning_rate', help=h["learning_rate"], type=float, default=0.003)

    args = parser.parse_args()

    return args

def load_train_data(train_data_path, batch_size, shuffle=True):
    # Convert images to tensors, normalize, and resize them
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers)

    return train_data_loader


def load_test_data(test_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    names = test_data.imgs
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return test_data_loader, names



def eval(model, val_data):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    cls_scores = {}
    all_labels = []
    all_pred = []
    all_scores = []
    for data in tqdm(val_data):
        images, labels = data
        images = images.to(device)
        outputs = model(images)
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        all_scores.append(soft_outputs.detach().cpu().numpy())
        score, predicted = torch.max(outputs.data, 1)
        for i, l in enumerate(labels):
            l = int(l)
            all_labels.append(l)
            all_pred.append(int(predicted[i]))
            if l not in cls_scores:
                cls_scores[l] = []

            if l == int(predicted[i]):
                cls_scores[l].append(1)
            else:
                cls_scores[l].append(0)
   
    all_scores = np.concatenate(all_scores)

    micro_roc_auc_ovr = roc_auc_score(
        all_labels,
        all_scores,
        multi_class="ovr",
        average="micro",
    )

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

    total = []
    output_line = ""
    for i in range(10):
        output_line += "cls {}: {}\n".format(i, sum(cls_scores[i]) / len(cls_scores[i]))
        total += cls_scores[i]

    print(output_line)
    print("total Score: {}".format(sum(total) / len(total)))

        
def train():
    args = parse_args()
    train_data = load_train_data(args.train_folder, args.batch_size)
    
    val_data = load_train_data(args.val_folder, args.batch_size)
    train_losses = []

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    if args.resume_from_weight != "":
        model = torch.load(args.resume_from_weight)
    else:
        model = torchvision.models.efficientnet_v2_l(weights="IMAGENET1K_V1") #models.vit_h_14(pretrained=True)
        model.classifier[-1] = nn.Linear(1280,10)

    cls_freq = [] 
    files = os.listdir(args.train_folder)
    files.sort()
    for l in files:
        if l != ".DS_Store":
            cls_freq.append(len(os.listdir(args.train_folder + '/' + l)))

    total_num_files = sum(cls_freq)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        if type(model) != torch.nn.DataParallel:
            model = nn.DataParallel(model)
    model.to(device)


    c = 0
    NUM_CLASSES = 10
    eval(model, val_data)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        with tqdm(train_data, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # get the inputs
                inputs, labels = inputs.to(device), labels.to(device)

                
                cutmix = v2.CutMix(num_classes=NUM_CLASSES)
                mixup = v2.MixUp(num_classes=NUM_CLASSES)
                cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + get predictions + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs.float(), labels_one_hot.float())

                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()

                correct = (predictions == labels).sum().item()
                accuracy = correct / args.batch_size

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                
                
            eval(model, val_data)
        if args.model_path is not None:
            torch.save(model, args.model_path + "_epoch{}".format(epoch))
        else:
            torch.save(model, 'unicornmodel.pth')


    print('Finished Training')
    if args.model_path is not None:
        torch.save(model, args.model_path)
    else:
        torch.save(model, 'unicornmodel.pth')

def test(model_path):
    args = parse_args()
    test_data, files = load_test_data(args.test_folder, args.batch_size)
    files = np.array(files)[:,0]
    names = []
    for file in files:
        _, name = os.path.split(file)
        names += [name[6:-4]]
        #names += [name]
    names = np.array(names)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = torch.load(args.model_path)
    model.to(device)
    model.eval()
    network_output = np.array([])
    scores = np.array([])
    correct = 0
    total = 0
    run_start = time.time()
    results_raw = []
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'class_id', 'score'])
        with torch.no_grad():
            for data in test_data:
                images, _ = data
                total += images.shape[0]
                images = images.to(device)
                outputs = model(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)

                results_raw.append(outputs.detach().cpu().numpy())
                score, predicted = torch.max(outputs.data, 1)

                network_output = np.concatenate((network_output,predicted.detach().cpu().numpy()))
                scores = np.concatenate((scores,score.detach().cpu().numpy()))

        results_raw = np.concatenate(results_raw)

        for idx, name in enumerate(names):
            # Gotcha16683523.png
            writer.writerow([name, int(network_output[idx]), scores[idx]])
    total_time = time.time() - run_start
    print("ran for {}".format(total_time))
    print("avg time for running: {}".format(total_time / total))


def eval_pipeline(model_path):
    args = parse_args()
    val_data = load_train_data(args.val_folder, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = torch.load(args.model_path)
    model.to(device)
    eval(model, val_data)




if __name__ == "__main__":
    args = parse_args()
    if args.test_only:
        test(args.model_path)
    elif args.eval_only:
        eval_pipeline(args.model_path)
    else:
        train()

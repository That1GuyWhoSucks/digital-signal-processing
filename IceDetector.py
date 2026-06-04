import random
import shutil
from os import listdir
from os.path import join
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from Utils import ImageClassificationCategories, ImageClassification

WD = r"D:\training"

MODEL_SETUP_DATA = Tuple[models.resnet18, transforms.Compose, torch.device]


def select_eval():
    ice_train: str = join(WD, "train", "Ice")
    ice_val: str = fr"{WD}\eval\Ice"
    no_ice_train: str = fr"{WD}\train\NoIce"
    no_ice_val: str = fr"{WD}\eval\NoIce"
    for file in listdir(ice_val):
        shutil.move(fr"{ice_val}\{file}", fr"{ice_train}\{file}")
    for file in listdir(no_ice_val):
        shutil.move(fr"{no_ice_val}\{file}", fr"{no_ice_train}\{file}")
    files: List[str] = listdir(ice_train)
    if len(files) < 150:
        raise Exception("Not enough images for training.")
    random.shuffle(files)
    for file in files[0:150]:
        shutil.move(fr"{ice_train}\{file}", fr"{ice_val}\{file}")
    files = listdir(no_ice_train)
    if len(files) < 100:
        raise Exception("Not enough images for training.")
    random.shuffle(files)
    for file in files[0:100]:
        shutil.move(fr"{no_ice_train}\{file}", fr"{no_ice_val}\{file}")


def train():
    device: torch.device = torch.device("cpu")
    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset: datasets.ImageFolder = datasets.ImageFolder(fr"{WD}/train", transform=transform)
    val_dataset: datasets.ImageFolder = datasets.ImageFolder(fr"{WD}/eval", transform=transform)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=32)

    model: models.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(5):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        correct: int = 0
        total: int = 0

        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs: torch.Tensor = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy after iter {i}: {correct / total}")

    torch.save(model.state_dict(), "ice_detector.pth")


def label():
    i: int = 0
    j: int = 0
    k: int = 0
    dat_layer: List[str] = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw"))
    hor_layer: List[str] = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}"))
    fil_layer: List[str] = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}"))

    while True:
        if k == len(fil_layer):
            k = 0
            j += 1
            if j == len(hor_layer):
                j = 0
                i += 1
                if i == len(dat_layer):
                    break
            dat_layer = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw"))
            hor_layer = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}"))
            fil_layer = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}"))
        elif k < 0:
            j -= 1
            if j < 0:
                i -= 1
                if i < 0:
                    i = 0
                hor_layer = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}"))
                j = len(hor_layer) - 1
            fil_layer = sorted(listdir(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}"))
            k = len(fil_layer) - 1

        im: cv.UMat = cv.imread(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}\{fil_layer[k]}")
        cv.imshow("img", im)
        cv.waitKey(1)
        inp = input(f"{i},{j},{k}: ")
        if inp == "q":
            shutil.copy(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}\{fil_layer[k]}", r"D:\training\NoIce")
            k += 1
        elif inp == "a":
            shutil.copy(fr"D:\SWIFT09_2024Oct27-30_HLY2403\COM-7\Raw\{dat_layer[i]}\{hor_layer[j]}\{fil_layer[k]}", r"D:\training\Ice")
            k += 1
        elif inp == "z":
            k += 1
        elif inp == "<":
            k -= 1


def setup_model() -> MODEL_SETUP_DATA:
    device: torch.device = torch.device("cpu")
    model: models.resnet18 = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    model.load_state_dict(torch.load("ice_detector.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, transform, device


def predict(setup_data: MODEL_SETUP_DATA, img_path: str, NIGHT_THRESH: float=30) -> ImageClassification:
    model: models.resnet18
    transform: transforms.Compose
    device: torch.device
    model, transform, device = setup_data
    image: Image = Image.open(img_path).convert("RGB")

    avg_grey: float = np.mean(np.asarray(image.convert("L")))
    
    if avg_grey < NIGHT_THRESH:
        return ImageClassification(classification=ImageClassificationCategories.night, confidence=1 - (avg_grey / (2 * NIGHT_THRESH)))

    tensor: torch.Tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs: torch.Tensor = model(tensor)

        probabilities: torch.Tensor = torch.softmax(outputs, dim=1)

        predicted_index: int = torch.argmax(probabilities, dim=1).item()

        confidence: float = probabilities[0][predicted_index].item()

    predicted_class: ImageClassificationCategories = [ImageClassificationCategories.ice, ImageClassificationCategories.no_ice][predicted_index]
    return ImageClassification(classification=predicted_class, confidence=confidence)


def test_model() -> None:
    ice_val: str = fr"{WD}\eval\Ice"
    no_ice_val: str = fr"{WD}\eval\NoIce"
    setup_data: MODEL_SETUP_DATA = setup_model()
    score: int = 0
    max_score: int = 0
    for file in listdir(ice_val):
        if predict(setup_data, fr"{ice_val}\{file}").classification == ImageClassificationCategories.ice:
            score += 1
        max_score += 1
    for file in listdir(no_ice_val):
        if predict(setup_data, fr"{no_ice_val}\{file}").classification == ImageClassificationCategories.no_ice:
            score += 1
        max_score += 1
    print(score, max_score)


if __name__ == "__main__":
    # label()
    # select_eval()
    # train()
    # test_model()
    exit()

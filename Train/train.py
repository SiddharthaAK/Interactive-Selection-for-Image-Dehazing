import os
import datetime

from PIL import Image
import pathlib
from DehazingDataset import DatasetType, DehazingDataset
from DehazingModel import AODnet

# TODO: Abstract this in DehazingModel.py
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tu_data

import torchvision.transforms as transforms
import torchvision.transforms.functional as tv_functional
from torchmetrics.image import StructuralSimilarityIndexMeasure

import cv2.ximgproc


def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def Preprocess(image: Image.Image) -> torch.Tensor:
    # Contrast Enhancement
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
            # transforms.functional.equalize
        ]
    )
    transformedImage = transform(image)

    # Gamma Correction
    gammaCorrectedImage = transforms.functional.adjust_gamma(transformedImage, 2.2)

    # Histogram Stretching
    min_val = gammaCorrectedImage.min()
    max_val = gammaCorrectedImage.max()
    stretchedImage = (gammaCorrectedImage - min_val) / (max_val - min_val)

    # for x in stretchedImage:
    #     for y in x:
    #         print(y)

    # Guided Filtering
    gFilter = cv2.ximgproc.createGuidedFilter(guide=stretchedImage.permute(1, 2, 0).numpy(), radius=3, eps=0.01)
    filteredImage = gFilter.filter(src=stretchedImage.permute(1, 2, 0).numpy())
    return torch.from_numpy(filteredImage).permute(2, 0, 1)


        # -------------------------------------------------------------------
        # save per epochs model
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
         os.mkdir(os.path.join(path, net_name))
    torch.save(
        {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
        f=os.path.join(path, net_name, "{}_{}.pth".format("AOD", epoch)),
    )
    # -------------------------------------------------------------------


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")

    datasetPath = GetProjectDir() / "dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k"
    trainingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess, verbose=False)
    validationDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Validation, transformFn=Preprocess, verbose=False)


    # TODO: Abstract this in DehazingModel.py
    trainingDataLoader = tu_data.DataLoader(trainingDataset, batch_size=32, shuffle=True, num_workers=3)
    validationDataLoader = tu_data.DataLoader(validationDataset, batch_size=32, shuffle=True, num_workers=3)

    print(len(trainingDataset), len(validationDataset))

    # Instantiate the AODNet model
    model = AODnet().to(device)
    print(model)
    
    best_ssim = 0.0
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 50

    train_number = len(trainingDataLoader)

    print("Started Training...")
    model.train()
    for epoch in range(EPOCHS):
        # -------------------------------------------------------------------
        # start training
        for step, (haze_image, ori_image) in enumerate(trainingDataLoader):
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = model(haze_image)
            loss = criterion(dehaze_image, ori_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 10 == 0 or step + 1 == train_number:
                print(
                    "Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}".format(
                        epoch + 1, EPOCHS, step + 1, train_number, optimizer.param_groups[0]["lr"], loss.item()
                    )
                )

        # -------------------------------------------------------------------
        # Validation loop
    print("Epoch: {}/{} | Validation Model Saving Images".format(epoch + 1, EPOCHS))
    model.eval()
    for step, (haze_image, ori_image) in enumerate(validationDataLoader):
        if step > 10:  # only save image 10 times
            break
        ori_image, haze_image = ori_image.to(device), haze_image.to(device)
        dehaze_image = model(haze_image)

        ssim = StructuralSimilarityIndexMeasure().to(device)
        ssim_val = ssim(dehaze_image, ori_image)
        print(f"SSIM: {ssim_val}")

        # Save only the model with the highest SSIM
        if ssim_val > best_ssim:
            best_ssim = ssim_val
            best_model_path = os.path.join(
                GetProjectDir() / "saved_models",
                "{}_best_model_{}.pth".format(epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            )

        torchvision.utils.save_image(
            torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0), nrow=ori_image.shape[0]),
            os.path.join(GetProjectDir() / "output", "{}_{}.jpg".format(epoch + 1, step)),
        )

    model.train()

    # Save the best model after all epochs
    if best_model_path:
        save_model(epoch, GetProjectDir() / "saved_models", model, optimizer, best_model_path)


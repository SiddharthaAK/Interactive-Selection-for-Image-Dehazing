import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tu_data
import torchvision.utils
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as psnr
from DehazingDataset import DatasetType, DehazingDataset
#from DehazingModel_1 import AODnet
from Preprocess_1 import Preprocess
from Model_ResNet import Model

def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent

def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save(
        {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
        f=os.path.join(path, net_name, "{}_{}.pth".format("DCP_ResNet", epoch)),
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasetPath = GetProjectDir() / "dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k"
    trainingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess, verbose=False)
    validationDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Validation, transformFn=Preprocess, verbose=False)

    trainingDataLoader = tu_data.DataLoader(trainingDataset, batch_size=2, shuffle=True, num_workers=3)
    validationDataLoader = tu_data.DataLoader(validationDataset, batch_size=2, shuffle=True, num_workers=3)

    print(len(trainingDataset), len(validationDataset))

    model = Model().to(device)
    print(model)

    best_ssim = 0.0
    best_psnr = 0.0
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20

    train_number = len(trainingDataLoader)

    print("Started Training...")
    model.train()

    for epoch in range(EPOCHS):
        for step, (haze_image, ori_image) in enumerate(trainingDataLoader):
            try:
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

            except FileNotFoundError as e:
                print(f"Error: {e}. Skipping this batch.")

        print("Epoch: {}/{} | Validation Model Saving Images".format(epoch + 1, EPOCHS))
        model.eval()
        total_ssim = 0.0
        total_psnr = 0.0

        for step, (haze_image, ori_image) in enumerate(validationDataLoader):
            try:
                if step > 10:  
                    break
                ori_image, haze_image = ori_image.to(device), haze_image.to(device)
                dehaze_image = model(haze_image)

                ssim = StructuralSimilarityIndexMeasure().to(device)
                ssim_val = ssim(dehaze_image, ori_image)
                total_ssim += ssim_val

                psnr_val = psnr(torch.clamp(dehaze_image, 0, 1).cpu().detach().numpy(), torch.clamp(ori_image, 0, 1).cpu().detach().numpy())

                total_psnr += psnr_val

                torchvision.utils.save_image(
                    torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0), nrow=ori_image.shape[0]),
                    os.path.join(GetProjectDir() / "output", "{}_{}.jpg".format(epoch + 1, step)),
                )

            except FileNotFoundError as e:
                print(f"Error: {e}. Skipping this batch.")

        average_ssim = total_ssim / (step + 1)
        average_psnr = total_psnr / (step + 1)
        print(f"Epoch: {epoch + 1}, Average SSIM: {average_ssim}, Average PSNR: {average_psnr}")
        model.train()

        # Save the model if both SSIM and PSNR are improved
        if average_ssim > best_ssim and average_psnr > best_psnr:
            best_ssim = average_ssim
            best_psnr = average_psnr
            save_model(epoch, GetProjectDir() / "saved_models", model, optimizer, "best_model")

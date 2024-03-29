import torch.nn as nn
import torch
from preprocess import generate_noise, LATENT_DIM
from tqdm import tqdm
import numpy as np
import gc
from torch.utils.data import DataLoader

from src.fid_metric import calculate_fid
from src.is_metric import inception_score
from src.preprocess import IS_PREPROCESSOR, FID_PREPROCESSOR
from src.utils import CustomFakeDataset

#
def train(gen_model, dis_model,  gen_optimizer, dis_optimizer,
          criterion, loader, device):
    gen_model.to(device)
    dis_model.to(device)

    dis_losses, gen_losses = [], []
    process = tqdm(loader)
    for batch in process:
        batch = {k: v.to(device) for k, v in batch.items()}

        real_images = batch['images']
        real_gender_labels = batch['labels']
        
        batch_size = real_images.shape[0]
        real_images_labels = torch.ones(batch_size, 1).to(device)

        fake_gender_labels = torch.randint(0, 2, size=(batch_size,1)).to(device)
        fake_imags_labels = torch.zeros(batch_size, 1).to(device)
        
        noise = generate_noise(batch_size).to(device)
        fake_images = gen_model(noise, fake_gender_labels).detach().to(device)
        # print(fake_images.shape)

        # Обучение дискриминатора
        dis_model.zero_grad()

        # Дискриминатор обрабатывает реальные и сгенерированные изображения
        real_outputs = dis_model(real_images, real_gender_labels)
        fake_outputs = dis_model(fake_images, fake_gender_labels)

        # Вычисление функции потерь для дискриминатора и обновление параметров
        discriminator_loss = criterion(real_outputs, real_images_labels) + criterion(fake_outputs, fake_imags_labels)
        discriminator_loss.backward()
        dis_optimizer.step()

        noise = generate_noise(batch_size).to(device)
        fake_images = gen_model(noise, fake_gender_labels)

        # Обучение генератора
        gen_model.zero_grad()
        fake_outputs = dis_model(fake_images, fake_gender_labels)

        # Вычисление функции потерь для генератора и обновление параметров
        generator_loss = criterion(fake_outputs, real_images_labels)
        generator_loss.backward()
        gen_optimizer.step()

        dis_losses.append(discriminator_loss.item())
        gen_losses.append(generator_loss.item())
        process.set_postfix({"dis_avg_loss": np.mean(dis_losses), "gen_avg_loss": np.mean(gen_losses)})
        gc.collect()
        torch.cuda.empty_cache()

    return gen_losses, dis_losses

# TODO
def evaluate(gen_model, real_loader, real_dataset_size, bs, device):
    gen_model.eval()

    print("Start Evaluation")

    print("Calculation of IS...")
    fake_dataset = CustomFakeDataset(gen_model, real_dataset_size, IS_PREPROCESSOR, device)
    fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=False)
    is_value = inception_score(fake_loader, device)
    
    torch.cuda.empty_cache()
    gc.collect()

    print("Inception Score: ",is_value)

    print("Calculation of FID...")
    fake_dataset = CustomFakeDataset(gen_model, real_dataset_size, FID_PREPROCESSOR, device)
    fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=False)

    try:
        fid_value = calculate_fid(real_loader, fake_loader, real_dataset_size, bs)
    except ValueError:
        print("Imaginary number!")
        fid_value = 1000000000

    torch.cuda.empty_cache()
    gc.collect()

    print("Fréchet Inception Distance,: ", fid_value )

    return fid_value, is_value
import sys
import os
import torch
import numpy as np
import random

#ROOT_DIR = '.'
ROOT_DIR = '/home/ubuntu/ImgGen/lab4'

sys.path.insert(0, f"{ROOT_DIR}/src")

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)

from src.neural_nets import *
from src.utils import *
from src.train import *
from src.preprocess import *
from time import time

import json
from torch.utils.data import DataLoader

CONFIG_FILE_JSON = f'{ROOT_DIR}/src/learning_config.json'
LOGS_DIR = f'{ROOT_DIR}/logs'
DATA_DIR = f'{ROOT_DIR}/data'
IMAGES_DIR = f'{DATA_DIR}/croped_images'
IMAGES_INFO = f'{DATA_DIR}/images_info.csv'
EXAMPLE_IMAGES = 16

##############################

with open(CONFIG_FILE_JSON, 'r', encoding='utf-8') as fd:
    json_obj = json.loads(fd.read())
learn_config = LearningConfig(**json_obj)

##############################

SAVE_DIR = f"{LOGS_DIR}/{learn_config.run_name}"
PLOTS_DIR = f'{SAVE_DIR}/plots'
BEST_MODEL_SAVE_PATH = f"{SAVE_DIR}/best_model.pt"
LAST_MODEL_SAVE_PATH = f"{SAVE_DIR}/last_model.pt"
BEST_DIS_SAVE_PATH = f"{SAVE_DIR}/best_dis_model.pt"
LAST_DIS_SAVE_PATH = f"{SAVE_DIR}/last_dis_model.pt"
LOGS_PATH = f"{SAVE_DIR}/logs.txt"
os.mkdir(SAVE_DIR)
os.mkdir(PLOTS_DIR)

print("Saving used config")
with open(f"{SAVE_DIR}/used_config.json", 'w', encoding='utf-8') as fd:
    json.dump(learn_config.__dict__, indent=2, fp=fd)

##############################

print("Init train objectives")

num_classes = len(os.listdir(IMAGES_DIR))
print(num_classes)

#
if learn_config.model_type == 'conditional':
    gen_model = ConditionalGenerator().to(learn_config.device)
    dis_model = ConditionalDiscriminator().to(learn_config.device)
elif learn_config.model_type == 'unconditional':
    gen_model = UnconditionalGenerator().to(learn_config.device)
    dis_model = UnconditionalDiscriminator().to(learn_config.device)

gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=learn_config.gen_lr, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=learn_config.dis_lr, betas=(0.5, 0.999))

criterion = torch.nn.BCELoss()

print("Saving used preprocessor")
with open(f"{SAVE_DIR}/used_prep.txt", 'w', encoding='utf-8') as fd:
    fd.write("\n=======EMBEDDER_ARCH=======\n")
    fd.write(IMAGE_PREPROCESSOR.__str__())

print("Saving used nn-arch")
with open(f"{SAVE_DIR}/used_arch.txt", 'w', encoding='utf-8') as fd:
    fd.write("\n=======GENERATOR_ARCH=======\n")
    fd.write(gen_model.__str__())
    fd.write("\n=======DISCRIMINATOR_ARCH=======\n")
    fd.write(dis_model.__str__())

##############################

print("Load train/eval datasets")

train_dataset = CustomCelebDataset(IMAGES_INFO, IMAGE_PREPROCESSOR, 'train', base_dir=ROOT_DIR)
train_loader = DataLoader(train_dataset, batch_size=learn_config.batch, 
                          collate_fn=custom_collate, shuffle=True, num_workers=2, drop_last=True)

eval_dataset = CustomCelebDataset(IMAGES_INFO, FID_PREPROCESSOR2, 'test', base_dir=ROOT_DIR, omit_label=True)
eval_loader = DataLoader(eval_dataset, batch_size=learn_config.inceptionv3_batch, shuffle=False)

print(len(eval_dataset))

##############################

ml_gen_train, ml_dis_train = [], []
fid_scores, is_scores = [], []
best_score = 100000

for i in range(learn_config.epochs):
    print(f"Epoch {i+1} start:")

    #
    train_s = time()
    gen_losses, dis_losses = train(gen_model, dis_model, gen_optimizer, dis_optimizer, 
                                   criterion, train_loader, learn_config.device)
    train_e = time()

    #
    with torch.no_grad():
        noise = torch.randn(EXAMPLE_IMAGES, LATENT_DIM).to(learn_config.device)
        fake_gender_labels = torch.randint(0, 2, size=(EXAMPLE_IMAGES,1)).to(learn_config.device)
        fake_images = gen_model(noise, fake_gender_labels)
        plot_image_grid(fake_images, PLOTS_DIR, i)

    #
    fid_score, is_score = evaluate(gen_model, eval_loader, len(eval_dataset), learn_config.inceptionv3_batch, learn_config.device)
    eval_e = time()


    if fid_score <= best_score:
        print("Saving new best model!")
        best_score = fid_score
        torch.save(gen_model.state_dict(), BEST_MODEL_SAVE_PATH)
        torch.save(dis_model.state_dict(), BEST_DIS_SAVE_PATH)

    #
    ml_gen_train.append(np.mean(gen_losses))
    ml_dis_train.append(np.mean(dis_losses))

    fid_scores.append(fid_score)
    is_scores.append(is_score)
    print(f"Epoch {i+1} results: gen_tain_loss - {round(ml_gen_train[-1], 5)} | dis_tain_loss - {round(ml_dis_train[-1], 5)}")
    print(f"fid_score: {fid_scores[-1]} | is_score: {is_scores[-1]}")

    # Save train/eval info to logs folder
    epoch_log = {
        'epoch': i+1, 'gen_train_loss': ml_gen_train[-1], 'dis_train_loss': ml_dis_train[-1],
        'fid_score': fid_scores[-1], 'is_score': is_scores[-1] , 'train_time': round(train_e - train_s, 5), 
        'eval_time': round(eval_e - train_e, 5)
        }
    with open(LOGS_PATH,'a',encoding='utf-8') as logfd:
        logfd.write(str(epoch_log) + '\n')

torch.save(gen_model.state_dict(), LAST_MODEL_SAVE_PATH)
torch.save(dis_model.state_dict(), LAST_DIS_SAVE_PATH)

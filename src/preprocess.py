import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt

#
LATENT_DIM = 100

#
IMAGE_PREPROCESSOR = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)])

IS_PREPROCESSOR = T.Compose([T.Resize((299, 299),antialias=True),
                             T.Normalize(0.5, 0.5)])

FID_PREPROCESSOR2 = T.Compose([T.Resize((299, 299),antialias=True),
                               T.ToTensor()])

FID_PREPROCESSOR = lambda x: FID_PREPROCESSOR2(T.ToPILImage()(x*0.5 + 0.5))


#
def generate_noise(n=1):
    return torch.randint(0, 256, size=(n,LATENT_DIM)) / 255

#
def plot_image_grid(tensor, save_dir, epoch_n):
    images = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()

    # denormalize
    images = images*0.5 + 0.5

    B, H, W, C = images.shape
    rows = int(np.sqrt(B))
    cols = int(np.ceil(B / rows))

    fig, axes = plt.subplots(rows, cols)
    fig.subplots_adjust(hspace=0.4)
    for i, ax in enumerate(axes.flat):
        if i < B:
            ax.imshow(images[i])
            ax.axis('off')

    if save_dir is None:
        plt.show()
    else:
        plt.savefig(f'{save_dir}/{epoch_n}.png')

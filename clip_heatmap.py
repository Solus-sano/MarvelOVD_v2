import torch
from PIL import Image
import requests
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import numpy as np
from clip import clip
from clip.clip import tokenize
import cv2
# import ClipTextModel


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)
# for item in model.children():
#     print(item)

activations = {}
gradients = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_input
    return hook

target_layer = model.visual.transformer.resblocks[7].ln_2
target_layer.register_forward_hook(get_activation('ln_2_output'))
target_layer.register_full_backward_hook(get_gradient('ln_2_gradient'))



def get_features_and_grads(img_tensor, class_id=None):
    img_tensor.requires_grad_(True)
    model.zero_grad()
    logits_per_image, logits_per_text = model(img_tensor,tokenize("a photo of person").to(device))
    if class_id is None:
        class_id = logits_per_image.argmax().item()
    logits_per_image[:, class_id].backward()

    act = activations['ln_2_output']
    grad = gradients['ln_2_gradient'][0]
    print(grad[:,0,:].sum(dim=-1))
    weights = torch.mean(grad, dim=[1, 2], keepdim=True)
    grad_cam = torch.mul(weights, act).sum(dim=1, keepdim=True)
    grad_cam = torch.relu(grad_cam.sum(dim=-1)[:,0])
    return grad_cam.cpu().detach()

if __name__ == '__main__':

    img_file = "/data1/liangzhijia/datasets/coco/val2017/000000206025.jpg"
    img = Image.open(img_file)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    heatmap = get_features_and_grads(img_tensor)

    heatmap = torch.tensor(heatmap)
    heatmap = heatmap[1:].reshape(7,7)
    # print(heatmap)
    heatmap = interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=img_tensor.shape[2:], mode='bilinear', align_corners=False)
    # print(heatmap.shape)
    heatmap = heatmap[0].permute(1,2,0).repeat(1,1,3).numpy()
    plt.figure()
    img_with_heatmap = np.float32(heatmap) #+ np.float32(img.resize((heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS))
    img_with_heatmap -= np.min(img_with_heatmap)
    img_with_heatmap /= np.max(img_with_heatmap)
    img_with_heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    plt.imshow(np.float32(img.resize((heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS)))
    plt.imshow(img_with_heatmap)
    plt.savefig("heatmap.jpg")

    # plt.imsave("heatmap.jpg",img_with_heatmap)
    # plt.axis('off')
    # plt.show()


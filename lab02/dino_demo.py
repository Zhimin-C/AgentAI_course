import roboflow
import os
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

# roboflow.login()
rf = roboflow.Roboflow()

project = rf.workspace("team-roboflow").project("coco-128")
dataset = project.version(2).download("coco")

cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd, "COCO-128-2/train/") ### 128 database images
save_dir = os.path.join(cwd, "output/") ### for saving the output images

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(ROOT_DIR)
files = [os.path.join(ROOT_DIR, f) for f in files if f.lower().endswith(".jpg")]

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])]) ### transofmation on the imaege

### Load the model DINOV2

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)
dinov2_vits14.eval()

def load_image(img: str):
    img = Image.open(img)
    img = transform_image(img)[:3].unsqueeze(0)   #### (c, 500, 500 ) -> (1, c, 500, 500) 
    return img

def create_index(files):
    index = faiss.IndexFlatL2(384)  ### compute L2 distance between q and db_embeddings
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14( load_image(file).to(device) )   ###  class-token -> 0th token (1, 384)
            embedding = embeddings.cpu().numpy()

            all_embeddings[file] = embedding

            index.add(embedding)
    return index, all_embeddings

data_index, all_embeddings = create_index(files)

def search_image(index: faiss.IndexFlatL2, query_embeddings: list, top_k: int =3) -> list:
    D, I = index.search(query_embeddings, top_k)
    return I[0] ##[[Idx1, Idx2, Idx3]] -> [Idx1, Idx2, Idx3]

search_file = "COCO-128-2/valid/000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
img = cv2.resize(cv2.imread(search_file), (416, 416))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## visualize the search image
print("Image to search: ", search_file)
plt.imshow(img)
plt.savefig(save_dir + "input_image.jpg")
plt.close()

with torch.no_grad():
    query_embedding = dinov2_vits14(load_image(search_file).to(device))
    indices = search_image(data_index, query_embedding.cpu().numpy())   ## topk = 3 results -> [idx1, idx2, idx3]

    for i, index in enumerate(indices):
        print(f"Image: {i}: {files[index]}")
        img = cv2.resize(cv2.imread(files[index]), (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.savefig(save_dir + f"output_image_{i}.jpg")
        plt.close()





















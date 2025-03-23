import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import joblib
from scipy import stats
import pyvips
from tqdm import tqdm
import argparse
from PIL import PngImagePlugin

# Crop WSI to patches
def is_background(region, threshold=0.95):
    avg_color = region.avg()
    return avg_color > 240 #If the mean color is smaller than 240, it is considered background

def crop_and_save(file_path, save_name,output_dir, tile_size, output_size=1120): #The output size should be greater than 224
    
    image = pyvips.Image.new_from_file(file_path, access='sequential')
    
    width = image.width
    height = image.height

    for top in range(0, height-tile_size+1, tile_size):
        for left in range(0, width-tile_size+1, tile_size):

            crop = image.crop(left, top, tile_size, tile_size)
   
            if not is_background(crop):
                resized_crop = crop.resize(output_size / tile_size) 
                output_path = f"{output_dir}/{save_name}_image_{left//tile_size}_{top//tile_size}.png"
                resized_crop.write_to_file(output_path)
            
def parse_feature_vector(vector_str):
    vector_str = vector_str.strip()[1:-1]
    vector_array = np.fromstring(vector_str, sep=',')
    return vector_array


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))  

if __name__ == '__main__':
    """Process image data and generate predictions of the niche 12 score for each WSI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, required=True, help='Local directory where the UNI model files, containing pytorch_model.bin, are downloaded.')
    parser.add_argument('--prediction_model_path', type=str, default="./statistics/PREDICT_MODEL.pth", required=True, help='Local directory where the prediction model are downloaded.')
    parser.add_argument('--WSI_path', type=str, required=True, help='Path to the whole slide image.')

    parser.add_argument('--cropped_patch_size', type=int, default=3000, required=False, help='Size of the cropped patches. This may depend on the magnification (default is x20).')   
    parser.add_argument('--img_path', type=str, default="../cropped_patch", required=False, help='Path to save cropped patches.')
    parser.add_argument('--out_path', type=str, default='../result', required=False, help='Path to save the final results.')
    parser.add_argument('--filter_RGB', type=int, default=240, required=False, help='Threshold for filtering background noise.')

    args = parser.parse_args()
    local_dir = str(args.local_dir)
    WSI_path = str(args.WSI_path)
    img_path = str(args.img_path)
    out_path = str(args.out_path)
    prediction_model_path = str(args.prediction_model_path)
    cropped_patch_size = int(args.cropped_patch_size)   
    filter_RGB = int(args.filter_RGB)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    path = WSI_path 
    file_names = os.listdir(path) # default WSIs are named as sample names, such as TCGA-25-1625.svs
    sample_df = pd.DataFrame({"sample_name":file_names})
    sample_df['sample_name'] = sample_df['sample_name'].str.split('.').str[0]
    for file_name in tqdm(file_names, desc="Processing files"):
        file_path = os.path.join(path, file_name)
        file_name = file_name.split('.')[0] # remove the extension, e.g. .svs
        crop_and_save(file_path, file_name, img_path, cropped_patch_size)

    # UNI ecodes patches as feature vectors
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()

    results = []
    features = []
    labels = []
    file_name_list = []
    file_names = os.listdir(img_path)

    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

    for file in file_names:
        file_path = os.path.join(img_path, file)
        parts = file.split('.')
        file_name = parts[0]
        print(file_name)
        file_name_list.append(file_name)

        image_path = os.path.join(img_path,f"{file}")
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = transform(image).unsqueeze(dim=0) 
        
        with torch.inference_mode():
            feature_emb = model(image) 

        feature_vector = feature_emb.squeeze().cpu().numpy()
        results.append({
            "file_name":file,
            "feature_vector":feature_vector.tolist() 
        })

    results_df = pd.DataFrame(results)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    results_df.to_csv(os.path.join(out_path,"PATCH_EMBEDDING.csv"))

    results_df = pd.read_csv(os.path.join(out_path,"PATCH_EMBEDDING.csv"))
    features = results_df["feature_vector"].apply(parse_feature_vector)
    X = np.array(features.tolist()) 
    X_test_tensor = torch.tensor(X, dtype=torch.float32)

    model1 = MLP(input_size=X_test_tensor.shape[1])
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model1.parameters(), lr=1e-3)  

    model1.load_state_dict(torch.load(prediction_model_path))
    model1.eval()
    with torch.no_grad():
        y_pred_tensor = model1(X_test_tensor)

    y_pre_MLP = y_pred_tensor.numpy().flatten()

    results_df['pred_niche12_proportion'] = y_pre_MLP
    results_df['Image_ID'] = results_df['file_name'].str.split('_').str[0]  # results_df['file_name'] is named as '{WSI}_{i}_{j}'
    results_df.to_csv(os.path.join(out_path,"PRED_NICHE12_PROPORTION.csv"))

    for idx, row in sample_df.iterrows():
        sample_id = row['sample_name']
        filtered_results = results_df[results_df['Image_ID'] == sample_id]
        pred_label_gt_10_count = (filtered_results['pred_niche12_proportion'] > 20).sum() 
        sample_df.at[idx, 'pred_niche12_score'] = pred_label_gt_10_count*1.0/len(filtered_results)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    sample_df.to_csv(os.path.join(out_path,"PRED_NICHE12_SCORE.csv"))
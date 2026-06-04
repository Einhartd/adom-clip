import os
import json
import urllib.request
import torch
import clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

data_dir = "../data/imagenet-r" 

if not os.path.exists(data_dir):
    print(f"Brak folderu {data_dir}. Podmień ścieżkę na własną, aby uruchomić pełen test.")
    exit()

dataset = ImageFolder(data_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
folder_names = dataset.classes

print("Pobieranie mapowania klas ImageNet...")
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
try:
    response = urllib.request.urlopen(url)
    imagenet_class_index = json.loads(response.read())
    synset_to_name = {v[0]: v[1].replace('_', ' ') for k, v in imagenet_class_index.items()}
except Exception as e:
    print(f"Błąd podczas pobierania etykiet: {e}")
    synset_to_name = {}

class_names = [synset_to_name.get(name, name) for name in folder_names]

prompts_raw = [f"{c}" for c in class_names]

prompts_standard = [f"A rendition of a {c}." for c in class_names]

style_templates = [
    "A sketch of a {}.",
    "A cartoon of a {}.",
    "An origami {}.",
    "A painting of a {}.",
    "A sculpture of a {}.",
    "A toy {}.",
    "A rendition of a {}."
]

print("\nRozpoczynam ekstrakcję cech tekstu...")
with torch.no_grad():
    tokens_raw = clip.tokenize(prompts_raw).to(device)
    text_features_raw = model.encode_text(tokens_raw)
    text_features_raw /= text_features_raw.norm(dim=-1, keepdim=True)
    
    tokens_std = clip.tokenize(prompts_standard).to(device)
    text_features_std = model.encode_text(tokens_std)
    text_features_std /= text_features_std.norm(dim=-1, keepdim=True)
    
    text_features_style = []
    print("Budowanie uśrednionych wektorów stylu dla każdej klasy...")
    for c in class_names:
        class_prompts = [template.format(c) for template in style_templates]
        tokens = clip.tokenize(class_prompts).to(device)
        
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        
        mean_feature = features.mean(dim=0)
        mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
        
        text_features_style.append(mean_feature)
        
    text_features_style = torch.stack(text_features_style).to(device)

correct_raw = 0
correct_std = 0
correct_style = 0
total = 0

print("\nRozpoczynam testowanie obrazów...")
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity_raw = (100.0 * image_features @ text_features_raw.T).softmax(dim=-1)
        similarity_std = (100.0 * image_features @ text_features_std.T).softmax(dim=-1)
        similarity_style = (100.0 * image_features @ text_features_style.T).softmax(dim=-1)
        
        predictions_raw = similarity_raw.argmax(dim=-1)
        predictions_std = similarity_std.argmax(dim=-1)
        predictions_style = similarity_style.argmax(dim=-1)
        
        correct_raw += (predictions_raw == labels).sum().item()
        correct_std += (predictions_std == labels).sum().item()
        correct_style += (predictions_style == labels).sum().item()
        total += labels.size(0)

acc_raw = (correct_raw / total) * 100
acc_std = (correct_std / total) * 100
acc_style = (correct_style / total) * 100

print("\n" + "="*50)
print("WYNIKI KLASYFIKACJI NA ZBIORZE IMAGENET-R")
print("="*50)
print(f"Brak kontekstu (tylko klasa):        {acc_raw:.2f}%")
print(f"Standard (A rendition of a...):      {acc_std:.2f}%")
print(f"Style-Aware (Prompt Ensembling):     {acc_style:.2f}%")
print("="*50)
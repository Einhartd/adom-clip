import os
import urllib.request
import json
import torch
import clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

data_dir = "../data/imagenet-r" 

if not os.path.exists(data_dir):
    print(f"Brak folderu {data_dir}. Pobierz i wypakuj ImageNet-R, a następnie zaktualizuj ścieżkę.")
    exit()

dataset = ImageFolder(data_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

print("Pobieranie mapowania klas ImageNet...")
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
try:
    response = urllib.request.urlopen(url)
    imagenet_class_index = json.loads(response.read())
    synset_to_name = {v[0]: v[1].replace('_', ' ') for k, v in imagenet_class_index.items()}
except Exception as e:
    print(f"Błąd podczas pobierania etykiet: {e}")
    synset_to_name = {}

folder_names = dataset.classes
class_names = [synset_to_name.get(name, name) for name in folder_names]

prompts_raw = [f"{c}" for c in class_names]
prompts_engineered = [f"a rendition of a {c}." for c in class_names]

tokens_raw = clip.tokenize(prompts_raw).to(device)
tokens_engineered = clip.tokenize(prompts_engineered).to(device)

print("\nRozpoczynam ekstrakcję cech tekstu...")
with torch.no_grad():
    text_features_raw = model.encode_text(tokens_raw)
    text_features_raw /= text_features_raw.norm(dim=-1, keepdim=True)
    
    text_features_eng = model.encode_text(tokens_engineered)
    text_features_eng /= text_features_eng.norm(dim=-1, keepdim=True)

correct_raw = 0
correct_eng = 0
total = 0

all_labels = []
all_preds_raw = []
all_preds_eng = []

print("Rozpoczynam testowanie obrazów (ImageNet-R)...")
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity_raw = (100.0 * image_features @ text_features_raw.T).softmax(dim=-1)
        similarity_eng = (100.0 * image_features @ text_features_eng.T).softmax(dim=-1)
        
        predictions_raw = similarity_raw.argmax(dim=-1)
        predictions_eng = similarity_eng.argmax(dim=-1)
        
        correct_raw += (predictions_raw == labels).sum().item()
        correct_eng += (predictions_eng == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds_raw.extend(predictions_raw.cpu().numpy())
        all_preds_eng.extend(predictions_eng.cpu().numpy())

acc_raw = (correct_raw / total) * 100
acc_eng = (correct_eng / total) * 100

print("\n" + "="*50)
print("WYNIKI KLASYFIKACJI ZERO-SHOT CLIP NA IMAGENET-R")
print("="*50)
print(f"Skuteczność (Sama nazwa klasy):          {acc_raw:.2f}%")
print(f"Skuteczność ('a rendition of a...'):     {acc_eng:.2f}%")
print("="*50)

print("\nGenerowanie i zapisywanie macierzy pomyłek...")

cm = confusion_matrix(all_labels, all_preds_eng)

plt.figure(figsize=(16, 16))
sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, 
            xticklabels=False, yticklabels=False)

plt.title("Macierz pomyłek modelu CLIP ('a rendition of a...') na ImageNet-R")
plt.xlabel("Przewidziana klasa (Predicted Label)")
plt.ylabel("Prawdziwa klasa (True Label)")

output_path = "confusion_matrix_clip.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Sukces! Macierz pomyłek została zapisana w pliku: {output_path}")

correct_raw = 0
correct_eng = 0
total = 0

all_labels = []
all_preds_eng = []
all_image_features = []

print("Rozpoczynam testowanie obrazów...")
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        all_image_features.append(image_features.cpu())
        
        similarity_raw = (100.0 * image_features @ text_features_raw.T).softmax(dim=-1)
        similarity_eng = (100.0 * image_features @ text_features_eng.T).softmax(dim=-1)
        
        predictions_raw = similarity_raw.argmax(dim=-1)
        predictions_eng = similarity_eng.argmax(dim=-1)
        
        correct_raw += (predictions_raw == labels).sum().item()
        correct_eng += (predictions_eng == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds_eng.extend(predictions_eng.cpu().numpy())

print("\nObliczanie podobieństwa Intra-class i Inter-class...")

features = torch.cat(all_image_features)
labels_tensor = torch.tensor(all_labels)

similarity_matrix = features @ features.T

same_class_mask = labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)

same_class_mask.fill_diagonal_(False)
diff_class_mask = ~same_class_mask
diff_class_mask.fill_diagonal_(False)

intra_sims = similarity_matrix[same_class_mask].numpy()
inter_sims = similarity_matrix[diff_class_mask].numpy()

if len(inter_sims) > len(intra_sims):
    inter_sims = np.random.choice(inter_sims, size=len(intra_sims), replace=False)

print("Generowanie wykresu podobieństw...")
plt.figure(figsize=(10, 6))

sns.histplot(intra_sims, color='blue', label='Wewnątrzklasowe (Intra-class)', kde=True, stat='density', alpha=0.5, bins=50)
sns.histplot(inter_sims, color='red', label='Międzyklasowe (Inter-class)', kde=True, stat='density', alpha=0.5, bins=50)

plt.title("Rozkład podobieństwa Cosinusowego: Wewnątrz vs Między klasami (CLIP na ImageNet-A)")
plt.xlabel("Podobieństwo Cosinusowe (Cosine Similarity)")
plt.ylabel("Gęstość")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

output_sim_path = "similarity_distribution.png"
plt.savefig(output_sim_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gotowe! Wykres rozkładu zapisano jako: {output_sim_path}")


print("\nObliczanie 4 rozkładów podobieństwa (Obraz vs Tekst)...")

features = torch.cat(all_image_features).cpu() # Wszystkie cechy obrazów [N, 512]
labels_tensor = torch.tensor(all_labels)       # Prawdziwe etykiety [N]

t_raw = text_features_raw.cpu() # Cechy surowych etykiet [200, 512]
t_eng = text_features_eng.cpu() # Cechy inżynierii promptu [200, 512]

sim_raw = features @ t_raw.T # Kształt: [N, 200]
sim_eng = features @ t_eng.T # Kształt: [N, 200]

# 3. Tworzenie maski do oddzielenia prawidłowych klas (Intra) od błędnych (Inter)
N = features.shape[0]
num_classes = t_raw.shape[0]

mask = torch.zeros((N, num_classes), dtype=torch.bool)
mask[torch.arange(N), labels_tensor] = True # Zaznaczamy komórki z prawidłowymi odpowiedziami

# 4. Wyciągnięcie wartości z użyciem maski
intra_raw = sim_raw[mask].numpy()
inter_raw = sim_raw[~mask].numpy()

intra_eng = sim_eng[mask].numpy()
inter_eng = sim_eng[~mask].numpy()

# 5. Balansowanie danych (Inter jest 199 razy więcej niż Intra, co psuje wizualizację)
# Losujemy próbkę wielkości Intra z puli Inter.
if len(inter_raw) > len(intra_raw):
    inter_raw = np.random.choice(inter_raw, size=len(intra_raw), replace=False)
    inter_eng = np.random.choice(inter_eng, size=len(intra_eng), replace=False)

# 6. Rysowanie wykresów (2 sub-ploty obok siebie)
print("Generowanie wykresów...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)

# Wykres 1: Surowe etykiety
sns.histplot(intra_raw, ax=axes[0], color='blue', label='Wewnątrzklasowe (Intra)', kde=True, stat='density', alpha=0.5, bins=50)
sns.histplot(inter_raw, ax=axes[0], color='red', label='Międzyklasowe (Inter)', kde=True, stat='density', alpha=0.5, bins=50)
axes[0].set_title("Surowe Etykiety (Raw Prompts)")
axes[0].set_xlabel("Podobieństwo Cosinusowe")
axes[0].set_ylabel("Gęstość")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.7)

# Wykres 2: Inżynieria promptu
sns.histplot(intra_eng, ax=axes[1], color='green', label='Wewnątrzklasowe (Intra)', kde=True, stat='density', alpha=0.5, bins=50)
sns.histplot(inter_eng, ax=axes[1], color='orange', label='Międzyklasowe (Inter)', kde=True, stat='density', alpha=0.5, bins=50)
axes[1].set_title("Inżynieria Promptów ('This is an example of a...')")
axes[1].set_xlabel("Podobieństwo Cosinusowe")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.suptitle("Wpływ Inżynierii Promptów na dopasowanie Obraz-Tekst (CLIP na ImageNet-A)", fontsize=16)

output_sim_path = "r_similarity_4_distributions.png"
plt.savefig(output_sim_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gotowe! Porównanie rozkładów zapisano jako: {output_sim_path}")
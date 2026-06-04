import torch
import clip
from PIL import Image

# 1. Inicjalizacja środowiska
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
image_path = "data/dog.jpg" 

try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Nie znaleziono pliku: {image_path}")
    exit()

labels_prompts = [
    "a photo of a dog",          # Angielski (Oczekiwany faworyt)
    "fotografia psa",               # Polski
    "ein Foto von einem Hund",   # Niemiecki
    "une photo d'un chien",      # Francuski
    # "zdjęcie samochodu"          # Polski (Zmyłka)
]
labels_prompts = [
    "dog",          # Angielski (Oczekiwany faworyt)
    "pies",               # Polski
    "Hund",   # Niemiecki
    "chien",      # Francuski
    # "zdjęcie samochodu"          # Polski (Zmyłka)
]

text_prompts = clip.tokenize(labels_prompts).to(device)

with torch.no_grad():
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    logits_image, _ = model(image_input, text_prompts)
    probs = logits_image.softmax(dim=-1).cpu().numpy()[0]

for label, prob in zip(labels_prompts, probs):
    print(f"- JĘZYK/PROMPT: '{label:<25}' -> PEWNOŚĆ: {prob * 100:>5.2f}%")
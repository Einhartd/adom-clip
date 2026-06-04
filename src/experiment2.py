import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "data/cup_and_book.jpg" 

try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Nie znaleziono pliku: {image_path}. Zrób zdjęcie i podaj poprawną ścieżkę.")
    exit()

labels_prompts = [
    "a photo of a cup resting on top of a book", 
    "a photo of a book resting on top of a cup",
    "a photo of a cup and a book next to each other",
    "a photo of a cup that looks like a stack of books",
    "a photo of a book that looks like a stack of cups",
]

text_prompts = clip.tokenize(labels_prompts).to(device)

with torch.no_grad():
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    logits_image, _ = model(image_input, text_prompts)
    probs = logits_image.softmax(dim=-1).cpu().numpy()[0]

for label, prob in zip(labels_prompts, probs):
    print(f"Label: '{label}'")
    print(f"Model confidence: {prob * 100:>5.2f}%\n")
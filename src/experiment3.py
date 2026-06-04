import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "data/apple_cartoon.jpg" 

try:
    image = Image.open(image_path)
except FileNotFoundError:
    exit()

labels_prompts = [
    "a photo of an apple",
    "a logo of apple",
    "a photo of an ipod",
    "a cartoon of an apple",
    "a phone of an iphone",
    "a photo of a banana"
]

text_prompts = clip.tokenize(labels_prompts).to(device)

with torch.no_grad():
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    logits_image, _ = model(image_input, text_prompts)
    probs = logits_image.softmax(dim=-1).cpu().numpy()[0]

for label, prob in zip(labels_prompts, probs):
    print(f"- ZAPYTANIE: '{label:<26}' -> PEWNOŚĆ: {prob * 100:>5.2f}%")
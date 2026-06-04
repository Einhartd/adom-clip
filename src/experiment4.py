import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_paths = [
    "data/apple.jpg", 
    "data/3apples.jpg", 
    "data/manyapples.jpg"
]

object_name = "apple"

labels_prompts = [
    f"a photo of one {object_name}",
    f"a photo of two {object_name}s",
    f"a photo of three {object_name}s",
    f"a photo of four {object_name}s",
    f"a photo of five {object_name}s",
    f"a photo of many {object_name}s",
    f"a photo of no {object_name}s"
]

text_prompts = clip.tokenize(labels_prompts).to(device)

for image_path in image_paths:
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"\n[BŁĄD] Nie znaleziono pliku: {image_path}. Pomijam...")
        continue

    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        logits_image, _ = model(image_input, text_prompts)
        probs = logits_image.softmax(dim=-1).cpu().numpy()[0]

    print(f"\n--- Wyniki dla obrazu: {image_path} ---")
    for label, prob in zip(labels_prompts, probs):
        print(f"- PROMPT: '{label:<25}' -> PEWNOŚĆ: {prob * 100:>5.2f}%")
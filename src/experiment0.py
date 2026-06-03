import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "data/paper.jpg"

try:
  image = Image.open(image_path)
except FileNotFoundError:
  print(f"Not found: {image_path}")
  exit()

labels_raw = ["bird", "paper", "newspaper", "origami"]
labels_prompts = [
    "a photo of a bird in nature",
    "a close-up of a printed newspaper",
    "a macro photo of a paper origami bird",
    "a simple piece of white paper"
]

text_raw = clip.tokenize(labels_raw).to(device)
text_prompts = clip.tokenize(labels_prompts).to(device)

with torch.no_grad():
  image_input = preprocess(image).unsqueeze(0).to(device)
  logits_image_raw, _ = model(image_input, text_raw)
  probs_raw = logits_image_raw.softmax(dim=-1).cpu().numpy()[0]

  logits_image_prompts, _ = model(image_input, text_prompts)
  probs_prompts = logits_image_prompts.softmax(dim=-1).cpu().numpy()[0]

print("Raw labels:")
for label, prob in zip(labels_raw, probs_raw):
  print(f"- {label:>40}: {prob * 100:>5.2f}%")

print("Prompted labels:")
for label, prob in zip(labels_prompts, probs_prompts):
  print(f"- {label:>40}: {prob * 100:>5.2f}%")
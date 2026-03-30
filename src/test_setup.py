import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Detected device: {device.upper()}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    image_path = Path(__file__).resolve().parent.parent / "dog.png"
    image = Image.open(image_path)
    
    image_input = preprocess(image).unsqueeze(0).to(device) #type:ignore
    
    labels = ["a diagram", "a dog", "a cat"]
    text = clip.tokenize(labels).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image_input, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    for label, prob in zip(labels, probs):
        print(f"- {label}: {prob * 100:.2f}%")
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.imshow(image)
    ax1.set_title("Wejściowy obraz")
    ax1.axis("off")
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax2.bar(labels, probs, color=colors)
    ax2.set_title("Wynik klasyfikacji Zero-shot")
    ax2.set_ylabel("Prawdopodobieństwo")
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("test_output.png")
    
    plt.show()
    
if __name__ == "__main__":
    main()
    
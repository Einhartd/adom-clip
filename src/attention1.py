import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

# 1. Ładowanie modelu (z poprawką "eager", aby wyciągnąć atencję)
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(
    model_name, 
    use_safetensors=True,       
    attn_implementation="eager"
).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Pobranie zbioru w formacie Parquet (brak błędu bezpieczeństwa skryptów)
print("Ładowanie zbioru Flickr30k...")
dataset = load_dataset("lmms-lab/flickr30k", split="test")

random_indices = random.sample(range(len(dataset)), 10)

print("Generowanie połączonych map uwagi (Obraz + Tekst)...")
# Tworzymy 4 kolumny!
fig, axes = plt.subplots(10, 4, figsize=(22, 40))
fig.tight_layout(pad=7.0)

for row_idx, data_idx in enumerate(random_indices):
    item = dataset[data_idx]
    image = item['image'].convert("RGB")
    caption = item['caption'][0]

    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True).to(device)

    # 3. Osobne zapytania do wieży wizyjnej i tekstowej
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values'], 
            output_attentions=True
        )
        text_outputs = model.text_model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            output_attentions=True
        )

    # ==========================================
    # 4A. ATENCJA OBRAZU
    # ==========================================
    last_layer_vision_attn = vision_outputs.attentions[-1]
    vision_attn_avg = last_layer_vision_attn.mean(dim=1).squeeze(0)
    cls_attention = vision_attn_avg[0, 1:].cpu().numpy()
    
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_grid = cls_attention.reshape(grid_size, grid_size)
    
    # Dodajemy maleńką wartość (1e-8), aby uniknąć dzielenia przez zero przy całkowicie czarnych mapach
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    
    img_np = np.array(image)
    attention_resized = cv2.resize(attention_grid, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)

    # ==========================================
    # 4B. ATENCJA TEKSTU
    # ==========================================
    last_layer_text_attn = text_outputs.attentions[-1]
    text_attn_avg = last_layer_text_attn.mean(dim=1).squeeze(0) # [Długość sekwencji, Długość sekwencji]
    
    # W CLIP ostateczna reprezentacja tekstu znajduje się w tokenie najwyższego indeksu (EOS)
    eos_idx = inputs['input_ids'][0].argmax().item()
    
    # Wyciągamy wektor mówiący, jak bardzo token EOS zwracał uwagę na inne słowa
    text_attention = text_attn_avg[eos_idx].cpu().numpy()
    
    # Konwersja numerycznych ID z powrotem na czytelne słowa
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Odfiltrowujemy padding
    seq_len = inputs['attention_mask'][0].sum().item()
    
    # Wycinamy pierwszy token (Start) i ostatni (End), zostawiając same słowa (od 1 do seq_len - 1)
    text_attention = text_attention[1 : seq_len - 1]
    tokens = tokens[1 : seq_len - 1]
    
    # (Opcjonalnie) Czyszczenie tokenów CLIP-a z symboli '</w>'
    clean_tokens = [t.replace('</w>', '') for t in tokens]

    # ==========================================
    # 5. RYSOWANIE WYNIKÓW
    # ==========================================
    # Kolumna 1: Oryginał
    axes[row_idx, 0].imshow(img_np)
    axes[row_idx, 0].set_title(f"Oryginał\nPrompt: '{caption}'", fontsize=10, wrap=True)
    axes[row_idx, 0].axis("off")
    
    # Kolumna 2: Atencja Obrazu (Samo skupienie)
    axes[row_idx, 1].imshow(attention_resized, cmap='jet')
    axes[row_idx, 1].set_title("Atencja Obrazu (ViT)", fontsize=10)
    axes[row_idx, 1].axis("off")
    
    # Kolumna 3: Nałożenie Obrazu
    axes[row_idx, 2].imshow(img_np)
    axes[row_idx, 2].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[row_idx, 2].set_title("Oryginał + Atencja", fontsize=10)
    axes[row_idx, 2].axis("off")
    
    plot_len = len(text_attention) # Obliczamy faktyczną ilość słupków po odcięciu
    
    axes[row_idx, 3].bar(range(plot_len), text_attention, color='mediumpurple')
    axes[row_idx, 3].set_xticks(range(plot_len))
    axes[row_idx, 3].set_xticklabels(clean_tokens, rotation=45, ha='right', fontsize=9)
    axes[row_idx, 3].set_title("Atencja Tekstu (Ważność słów)", fontsize=10)
    axes[row_idx, 3].set_ylabel("Waga")
    axes[row_idx, 3].grid(axis='y', linestyle='--', alpha=0.5)

output_path = "flickr_attention_image_and_text.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Sukces! Wygenerowano plik z mapami: {output_path}")
# adom-clip: Eksperymenty z modelem multimodalnym CLIP

Projekt dedykowany jest badaniu możliwości, ograniczeń oraz optymalizacji multimodalnego modelu CLIP (Contrastive Language-Image Pre-training). Architektura repozytorium obejmuje analizę reprezentacji przestrzennej (embeddings), testy klasyfikacji zero-shot na klasycznych benchmarkach (MNIST, Fashion-MNIST) oraz badanie odporności modelu na przesunięcia domenowe i zadania związane z liczeniem obiektów.

## Struktura Projektu

    ├── data/                                  # Zbiór obrazów testowych (różne domeny i dystrybucje)
    ├── src/                                   # Notebooki badawcze i skrypty źródłowe
    │   ├── 1.mixed_experiments.ipynb          # Analiza dystrybucji i podobieństwa tekst-obraz
    │   ├── 2.exp_pca_ciphar_10.ipynb          # Redukcja wymiarowości (PCA) na osadzeniach CIFAR-10
    │   ├── 3.mnist-number-experiment.ipynb    # Klasyfikacja cyfr (MNIST) za pomocą CLIP
    │   └── 4.mnist-fashion-experiment.ipynb   # Klasyfikacja odzieży (Fashion-MNIST)
    │
    ├── environment.yml                        # Konfiguracja bazowego środowiska Conda
    └── requirements.txt                       # Szczegółowe zależności PyPI (zarządzane przez uv)

## Przegląd Eksperymentów

### 1. Eksperymenty Mieszane (`1.mixed_experiments.ipynb`)
Badanie ogólnych zdolności asocjacyjnych modelu CLIP przy użyciu zróżnicowanego zestawu danych jakościowych z katalogu `data/`. Analiza obejmuje:
* Weryfikację dopasowania tekstu do obrazu dla obiektów codziennego użytku (`coffee.jpg`, `cup_and_book.jpg`).
* Analizę zachowania modelu wobec obiektów abstrakcyjnych, schematycznych i geometrycznych (`origami.png`).
* Testy odporności na specyficzne domeny (zdjęcia satelitarne `sat_europe.jpg`, wycinki prasowe `newspaper.jpg`).
* Wstępne testy zdolności zliczania obiektów (seria plików `Xapples.jpg`).

### 2. Analiza Przestrzeni Ukrytej na CIFAR-10 (`2.exp_pca_ciphar_10.ipynb`)
Ekstrakcja wektorów cech (embeddings) dla zbioru danych CIFAR-10. Zastosowanie algorytmu PCA (Principal Component Analysis) do wizualizacji i oceny stopnia separowalności poszczególnych klas w przestrzeni ukrytej CLIP.

### 3. Klasyfikacja Cyfr (`3.mnist-number-experiment.ipynb`)
Eksperymenty nad klasyfikacją zero-shot na zbiorze MNIST. Analiza skupia się na specyfice interpretowania symboli numerycznych przez model uczony na danych internetowych oraz technikach inżynierii promptów (prompt engineering) w celu poprawy dokładności klasyfikacji trudnych klas.

### 4. Klasyfikacja Odzieży (`4.mnist-fashion-experiment.ipynb`)
Weryfikacja działania modelu na zbiorze Fashion-MNIST. Ewaluacja zdolności CLIP do rozpoznawania cech fasonów tekstyliów i odzieży bez wcześniejszego dostrajania (fine-tuningu).

---

## Instalacja i Konfiguracja

Środowisko projektowe zarządzane jest hybrydowo: Conda odpowiada za stabilne środowisko bazowe i interpreter Pythona, natomiast szybki instalator `uv` zarządza pakietami PyPI.

### Krok 1: Utworzenie środowiska Conda
Utwórz bazowe środowisko na podstawie pliku konfiguracyjnego:
`conda env create -f environment.yml`

### Krok 2: Aktywacja środowiska
`conda activate adom-clip`

### Krok 3: Instalacja zależności za pomocą `uv`
Użyj `uv` do błyskawicznej instalacji i synchronizacji bibliotek wymienionych w `requirements.txt` (w tym `torch`, `torchvision` oraz `datasets`):
`uv pip install -r requirements.txt`

## Kluczowe Wyniki i Narzędzia Wizualizacji
Wewnątrz notebooków generowane są macierze pomyłek (confusion matrices) oraz wykresy rozkładów podobieństwa (similarity distributions), które pozwalają na bieżąco monitorować jakość dopasowania reprezentacji tekstowych do wizualnych.
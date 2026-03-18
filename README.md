# Image Dataset Preprocessing & Augmentation Pipeline

Pipeline Python pour préparer et augmenter un dataset d'images destiné à l'entraînement de modèles de Deep Learning.

## Fonctionnalités

Le projet se compose de deux étapes enchaînées automatiquement :

### Étape 1 — Prétraitement des images (Data Preprocessing)

| Opération | Description |
|---|---|
| **Segmentation (SAM)** | Détection et extraction automatique des objets via le modèle SAM (Segment Anything Model) |
| **Débruitage** | Suppression du bruit avec `cv2.fastNlMeansDenoisingColored` |
| **Amélioration du contraste** | Égalisation adaptative (CLAHE) sur le canal L en espace LAB |
| **Netteté (Sharpening)** | Filtre de netteté par convolution |
| **Padding & Redimensionnement** | Mise à l'échelle carrée (224×224) avec fond noir |

### Étape 2 — Augmentation des données (Data Augmentation)

Génération d'images augmentées à partir des images prétraitées :

- **Rotation** aléatoire (probabilité : 50 %, ±25°)
- **Flou** (probabilité : 10 %)
- **Bruit aléatoire** (probabilité : 50 %)
- **Flip horizontal** (probabilité : 30 %)
- **Flip vertical** (optionnel)

Les probabilités et opérations sont configurables dans `augmentation_config.py`.

## Prérequis

- Python 3.8+
- GPU compatible CUDA (recommandé pour SAM)
- Poids du modèle SAM (`sam3.pt` par défaut)

## Installation

```bash
git clone <repo-url>
cd py-image-dataset-generator
pip install -r requirements.txt
```

## Utilisation

### Pipeline complète (preprocessing + augmentation)

```bash
python pipeline.py -input=path/to/raw/images -output=path/to/augmented -limit=500
```

### Paramètres

| Paramètre | Description |
|---|---|
| `-input`, `-i` **(requis)** | Dossier contenant les images brutes |
| `-output`, `-o` | Dossier de destination pour les images augmentées (défaut : `output`) |
| `-limit`, `-l` | Nombre d'images augmentées à générer (défaut : 500) |
| `-sam_weights` | Chemin vers les poids SAM (défaut : `sam3.pt`) |
| `--skip-preprocessing` | Sauter le prétraitement et lancer uniquement l'augmentation |
| `--preprocess-only` | Lancer uniquement le prétraitement (pas d'augmentation) |

### Exemples

**Pipeline complète :**
```bash
python pipeline.py -input=mes_images -output=dataset_augmente -limit=1000
```

**Prétraitement seul :**
```bash
python pipeline.py -input=mes_images --preprocess-only
```

**Augmentation seule** (sur des images déjà prétraitées) :
```bash
python pipeline.py -input=data_preprocessing/preprocessed/padding --skip-preprocessing -output=dataset_augmente -limit=2000
```

### Augmentation seule (mode standalone)

```bash
python augmentation.py -folder=mon_dossier -limit=10000 -dest=dossier_sortie
```

### Pipeline personnalisée en Python

```python
from augmentation.augmentation import DatasetGenerator

pipeline = DatasetGenerator(
    folder_path="images/preprocessed/",
    num_files=5000,
    save_to_disk=True,
    folder_destination="images/results"
)
pipeline.rotate(probability=0.5, max_left_degree=25, max_right_degree=25)
pipeline.random_noise(probability=0.5)
pipeline.blur(probability=0.5)
pipeline.horizontal_flip(probability=0.2)
pipeline.execute()
```

## Structure du projet

```
├── pipeline.py                  # Point d'entrée principal (preprocessing → augmentation)
├── augmentation.py              # Point d'entrée standalone pour l'augmentation
├── augmentation_config.py       # Configuration des opérations d'augmentation
├── requirements.txt             # Dépendances Python
├── augmentation/
│   ├── augmentation.py          # Classe DatasetGenerator
│   └── operations.py            # Opérations d'augmentation (Rotate, Blur, Flip, etc.)
├── data_preprocessing/
│   ├── datapreprocessing.py     # Pipeline de prétraitement (segmentation → padding)
│   └── preprocessed/            # Sorties intermédiaires du prétraitement
│       ├── segmentation/
│       ├── denoise/
│       ├── contrast/
│       ├── sharpened/
│       └── padding/
├── utils/
│   └── utils.py                 # Utilitaires (fichiers, barre de progression)
└── tests/
    └── utils/
        └── test_string_utils.py
```

## Configuration

Modifiez `augmentation_config.py` pour ajuster les opérations d'augmentation :

```python
DEFAULT_OPERATIONS = [
    'rotate',
    'blur',
    'random_noise',
    'horizontal_flip',
    # 'vertical_flip'
]

DEFAULT_ROTATE_PROBABILITY = 0.5
DEFAULT_ROTATE_MAX_LEFT_DEGREE = 25
DEFAULT_ROTATE_MAX_RIGHT_DEGREE = 25
DEFAULT_BLUR_PROBABILITY = 0.1
DEFAULT_RANDOM_NOISE_PROBABILITY = 0.5
DEFAULT_HORIZONTAL_FLIP_PROBABILITY = 0.3
DEFAULT_VERTICAL_FLIP_PROBABILITY = 0.3
```

## Dépendances

- `scipy` — Calcul scientifique
- `scikit-image` — Traitement d'images (IO, transformations)
- `opencv-python` — Vision par ordinateur (débruitage, contours, etc.)
- `numpy` — Manipulation de tableaux
- `Pillow` — Manipulation d'images (padding, resize)
- `ultralytics` — Modèle SAM pour la segmentation

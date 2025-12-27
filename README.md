# HTR-Local - Transcription de manuscrits

Application de transcription de manuscrits cursifs français (1880-1920) utilisant l'OCR et l'IA.

## Fonctionnalités

- **Segmentation** : Détection automatique des lignes de texte (Kraken)
- **OCR** : Reconnaissance de l'écriture manuscrite française (TrOCR)
- **Correction LLM** : Amélioration du texte avec GPT-4o
- **Résumé** : Analyse et résumé des plaintes/documents
- **Export PDF** : PDF searchable (texte sélectionnable sur l'image)

## Installation

### Prérequis

- Python 3.10 ou plus récent
- Une clé API OpenAI (pour la correction LLM)

### Étapes

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd trocr
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv

   # Sur Mac/Linux :
   source venv/bin/activate

   # Sur Windows :
   venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer la clé API OpenAI**
   ```bash
   cp .env.example .env
   ```
   Puis éditer `.env` et remplacer `sk-xxx` par ta vraie clé API.

5. **Lancer l'application**
   ```bash
   streamlit run app/main.py
   ```

   L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`

## Utilisation

1. **Upload** : Importer une image de manuscrit (PNG, JPG, TIFF)
2. **Segmentation** : Détecter les lignes de texte
3. **Transcription** : Lancer l'OCR sur chaque ligne
4. **Correction** :
   - *Correction globale* : Le LLM reconstruit un texte lisible
   - *Ligne par ligne* : Correction individuelle de chaque ligne
   - *Résumé* : Analyse et extraction des informations clés
5. **Export** : Générer un PDF searchable ou un rapport

## Notes

- Le premier lancement télécharge les modèles (~2 Go), ça peut prendre quelques minutes
- Sur Mac avec puce Apple Silicon, l'OCR utilise le GPU (MPS)
- Les documents sont sauvegardés dans une base SQLite locale

## Structure du projet

```
trocr/
├── app/
│   ├── main.py              # Interface Streamlit
│   ├── database/
│   │   ├── models.py        # Modèles SQLModel
│   │   └── db.py            # Connexion DB
│   └── services/
│       ├── segmentation.py  # Kraken
│       ├── ocr.py           # TrOCR
│       ├── llm.py           # OpenAI
│       └── pdf.py           # Export PDF
├── data/                    # Données locales
├── requirements.txt
└── .env                     # Clé API (à créer)
```

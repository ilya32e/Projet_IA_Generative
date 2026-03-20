# AISCA - Cartographie Semantique des Competences et Recommandation de Metiers

AISCA est une application Streamlit qui analyse un profil utilisateur par appariement semantique, calcule des scores de couverture des competences, recommande les 3 profils de metiers les plus pertinents et genere un plan de progression ainsi qu'une courte bio professionnelle avec un usage controle de la GenAI.

Le projet repose sur un pipeline NLP local base sur des embeddings SBERT et la similarite cosinus, avec une couche GenAI contrainte pour limiter les appels et reutiliser les reponses mises en cache.

## Fonctionnalites

- Questionnaire hybride combinant texte libre, auto-evaluation de type Likert, questions guidees et choix multiples
- Stockage structure des donnees en JSON et CSV locaux
- Analyse semantique locale avec embeddings SBERT et similarite cosinus
- Scoring pondere par bloc de competences
- Recommandation des 3 metiers les plus pertinents
- Visualisations des resultats avec radar et barres
- Generation d'un plan de progression
- Generation d'une bio professionnelle courte
- Cache local des generations pour reduire les appels repetes

## Stack Technique

- Python
- Streamlit
- Pandas
- Plotly
- scikit-learn
- SentenceTransformers
- Transformers
- Google GenAI SDK

## Structure du Projet

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
|-- data/
|   |-- raw/
|   `-- processed/
|-- cache/
|-- scripts/
|   |-- prepare_data.py
|   `-- demo_run.py
|-- src/
|   |-- analytics.py
|   |-- config.py
|   |-- data_pipeline.py
|   |-- genai.py
|   |-- recommender.py
|   `-- semantic_engine.py
`-- tests/
    `-- test_pipeline.py
```

## Fonctionnement

1. L'utilisateur remplit un questionnaire hybride sur ses competences, outils, projets et experiences.
2. Les reponses sont normalisees puis enregistrees localement dans un format structure.
3. Les preuves utilisateur sont comparees a un referentiel de competences via des embeddings semantiques.
4. Les scores de couverture sont agreges par bloc de competences.
5. Le systeme classe les profils de metiers les plus pertinents et retourne le top 3.
6. Sur demande, le module GenAI genere :
   - un plan de progression centre sur les competences les plus faibles
   - une courte bio professionnelle basee sur le profil analyse

## Conception de la Partie GenAI

La couche generative est volontairement contrainte :

- les sorties sont mises en cache localement dans `cache/genai_cache.json`
- les requetes identiques reutilisent les generations deja obtenues
- la generation du plan est verrouillee a un seul appel logique par profil analyse
- la generation de la bio est verrouillee a un seul appel logique par profil analyse
- Gemini peut etre utilise si une cle API est disponible
- un modele local de secours est disponible si Gemini n'est pas configure

## Installation

Version recommandee : Python 3.11 ou plus

Installer les dependances :

```bash
pip install -r requirements.txt
```

Preparer les donnees traitees :

```bash
python scripts/prepare_data.py
```

## Configuration

Creer un fichier `.env` a partir de `.env.example`.

Exemple de variables :

```env
SBERT_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
GEMINI_API_KEY=
GENAI_PROVIDER=gemini
GEMINI_MODEL_NAME=gemini-2.5-flash
LOCAL_GENAI_MODEL_NAME=google/flan-t5-small
GENAI_MODEL_NAME=gemini-2.5-flash
SEMANTIC_BACKEND=auto
SEMANTIC_THRESHOLD=0.52
TOP_N_JOBS=3
```

Notes :

- si `GEMINI_API_KEY` est vide, l'application bascule sur un comportement local de secours pour la generation
- l'analyse semantique fonctionne meme sans cle API GenAI

## Lancer l'Application

```bash
streamlit run app.py
```

Par defaut, l'application est accessible a l'adresse :

```text
http://localhost:8501
```

## Lancer la Demo Console

```bash
python scripts/demo_run.py
```

## Lancer les Tests

```bash
python -m unittest discover -s tests
```

La suite de tests actuelle couvre :

- la coherence de la preparation des donnees
- le comportement de recommandation du top 3
- la stabilite de l'identifiant de soumission
- la reutilisation du cache GenAI
- le verrouillage a un seul appel pour le plan
- l'independance du classement vis-a-vis du metier cible choisi

## Modules Principaux

- [app.py](app.py) : interface utilisateur Streamlit
- [src/data_pipeline.py](src/data_pipeline.py) : preparation, normalisation et stockage local des donnees
- [src/semantic_engine.py](src/semantic_engine.py) : moteur d'encodage semantique et de similarite
- [src/recommender.py](src/recommender.py) : scoring, agregation et recommandation de metiers
- [src/genai.py](src/genai.py) : logique de generation, cache et fallback de provider
- [src/config.py](src/config.py) : configuration generale et chemins du projet

## Perimetre Actuel

L'application Streamlit actuelle se concentre sur les exigences coeur du projet :

- questionnaire
- analyse semantique
- scoring des competences
- recommandation des 3 metiers
- generation du plan de progression
- generation de la bio professionnelle

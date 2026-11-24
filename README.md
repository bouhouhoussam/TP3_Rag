# RAG Multimodal Agentique

Application RAG multimodale et agentique qui extrait, indexe et interroge
le texte ainsi que les images de l'article *Attention Is All You Need*.

**Stack :** PyMuPDF · OpenCLIP (ViT-B-32) · ChromaDB · LangGraph · Ollama

## Progression
- [x] Étape 0 : Initialisation du projet
- [X] Étape 1 : Extraction du contenu multimodal (texte et images)
- [X] Étape 2 : Génération des embeddings CLIP dans un espace latent partagé
- [X] Étape 3 : Indexation dans ChromaDB
- [X] Étape 4 : Recherche multimodale par similarité cosinus
- [X] Étape 5 : Transformation du retriever en outil agentique
- [X] Étape 6 : Agent ReAct avec LangGraph
- [X] Étape 7 : Tests et validation

## Installation
    uv sync
## Structure
- `data/` : PDF source
- `notebooks/` : un notebook par étape, avec les sorties
- `src/embeddings.py` : encodage CLIP (texte et image) et découpage en chunks

## Limites connues
- Le modèle local `llama3.2:3b` (3 milliards de paramètres, CPU) appelle parfois l'outil `consulter_pdf` même pour une simple salutation.
- L'agent récupère les figures (chemin, page) mais ne les analyse pas visuellement : le modèle de l'agent ne traite que du texte.
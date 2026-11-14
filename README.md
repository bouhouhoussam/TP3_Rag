# RAG Multimodal Agentique

Application RAG multimodale et agentique qui extrait, indexe et interroge
le texte ainsi que les images de l'article *Attention Is All You Need*.

**Stack :** PyMuPDF · OpenCLIP (ViT-B-32) · ChromaDB · LangGraph · Ollama

## Progression
- [x] Étape 0 : Initialisation du projet
- [X] Étape 1 : Extraction du contenu multimodal (texte et images)
- [X] Étape 2 : Génération des embeddings CLIP dans un espace latent partagé
- [X] Étape 3 : Indexation dans ChromaDB
- [ ] Étape 4 : Recherche multimodale par similarité cosinus
- [ ] Étape 5 : Transformation du retriever en outil agentique
- [ ] Étape 6 : Agent ReAct avec LangGraph
- [ ] Étape 7 : Tests et validation

## Installation
    uv sync
## Structure
- `data/` : PDF source
- `notebooks/` : un notebook par étape, avec les sorties
- `src/embeddings.py` : encodage CLIP (texte et image) et découpage en chunks
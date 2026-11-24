import json
from typing import Literal
from langchain_core.tools import tool

from src.recherche import rechercher


def recuperer_sources(question, k=6, type_resultat=None):
    resultats = rechercher(texte=question, type_resultat=type_resultat, k=k)
    sources = []
    for i, r in enumerate(resultats, start=1):
        source = {"reference": f"S{i}"}
        source.update(r)
        sources.append(source)
    return sources


@tool
def consulter_pdf(
    question: str,
    type_resultat: Literal["tous", "texte", "image"] = "tous",
    k: int = 4,
) -> str:
    """
    Rechercher dans le PDF indexé (article "Attention Is All You Need")
    les passages de texte et les figures liés à la question.
    Utiliser cet outil pour toute question sur le contenu du document.
    type_resultat : "tous", "texte" ou "image".
    Renvoie un JSON avec les sources (référence, page, similarité, contenu).
    """
    filtre = None if type_resultat == "tous" else type_resultat
    k = max(1, min(k, 8))
    try:
        sources = recuperer_sources(question, k=k, type_resultat=filtre)
    except Exception as erreur:
        return json.dumps({"erreur": str(erreur), "sources": []}, ensure_ascii=False)
    return json.dumps({"question": question, "sources": sources}, ensure_ascii=False)
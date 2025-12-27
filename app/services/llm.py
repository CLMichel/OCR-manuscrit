"""Service de correction LLM avec OpenAI."""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Charger le .env
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Client OpenAI
_client = None


def get_client() -> OpenAI:
    """Retourne le client OpenAI (singleton)."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non définie dans .env")
        _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT_BASE = """Tu es un expert paléographe français spécialisé dans les manuscrits cursifs du début du XXe siècle (1880-1920).

Ta tâche est de corriger le texte OCR brut d'un manuscrit. Tu dois :

1. CORRIGER les erreurs de lecture OCR évidentes (lettres mal reconnues, mots tronqués)
2. RESPECTER l'orthographe et la grammaire de l'époque (ne PAS moderniser)
3. CONSERVER la ponctuation originale telle qu'elle apparaît
4. GARDER les formulations d'époque (tournures anciennes, vocabulaire désuet)
5. NE PAS ajouter de ponctuation ou mots manquants sauf si évident du contexte

Réponds UNIQUEMENT avec le texte corrigé, sans explication ni commentaire."""


def build_system_prompt(document_context: str = "") -> str:
    """Construit le prompt système avec le contexte documentaire."""
    if document_context:
        return f"""{SYSTEM_PROMPT_BASE}

CONTEXTE DU DOCUMENT :
{document_context}

Utilise ce contexte pour mieux interpréter les noms propres, lieux, vocabulaire technique et abréviations."""
    return SYSTEM_PROMPT_BASE


def correct_text(
    raw_text: str,
    context_before: str = "",
    context_after: str = "",
    document_context: str = ""
) -> str:
    """
    Corrige un texte OCR brut avec le LLM.

    Args:
        raw_text: Texte OCR à corriger
        context_before: Lignes précédentes pour le contexte
        context_after: Lignes suivantes pour le contexte
        document_context: Contexte général du document

    Returns:
        Texte corrigé
    """
    client = get_client()

    # Construire le prompt système avec contexte documentaire
    system_prompt = build_system_prompt(document_context)

    # Construire le message utilisateur avec contexte des lignes
    user_message = ""
    if context_before:
        user_message += f"[Contexte précédent : {context_before}]\n\n"

    user_message += f"Texte OCR à corriger :\n{raw_text}"

    if context_after:
        user_message += f"\n\n[Contexte suivant : {context_after}]"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def correct_transcriptions(
    transcriptions: list[dict],
    context_window: int = 2,
    document_context: str = "",
    progress_callback=None
) -> list[dict]:
    """
    Corrige une liste de transcriptions avec contexte.

    Args:
        transcriptions: Liste de dicts avec line_number, text, etc.
        context_window: Nombre de lignes de contexte avant/après
        document_context: Contexte général du document
        progress_callback: Fonction appelée avec (current, total)

    Returns:
        Liste de dicts avec text_corrected ajouté
    """
    results = []
    total = len(transcriptions)

    for i, trans in enumerate(transcriptions):
        # Extraire le contexte des lignes voisines
        context_before = " ".join([
            t["text"] for t in transcriptions[max(0, i - context_window):i]
        ])
        context_after = " ".join([
            t["text"] for t in transcriptions[i + 1:i + 1 + context_window]
        ])

        # Corriger avec tous les contextes
        corrected = correct_text(
            trans["text"],
            context_before=context_before,
            context_after=context_after,
            document_context=document_context
        )

        # Ajouter le résultat
        result = trans.copy()
        result["text_corrected"] = corrected
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


SUMMARY_PROMPT = """Tu es un historien spécialisé dans les archives ferroviaires françaises du début du XXe siècle.

Tu reçois la transcription d'une page d'un livre de plaintes d'une gare.

Analyse le texte et pour CHAQUE plainte identifiée, donne :
- **Date** : si mentionnée
- **Plaignant** : nom ou description si disponible
- **Objet** : le sujet de la plainte en 1 phrase
- **Résumé** : ce qui s'est passé en 2-3 phrases
- **Personnes mentionnées** : noms de chefs de gare, employés, etc.

Si le texte contient plusieurs plaintes, sépare-les clairement.
Si certaines informations sont illisibles ou manquantes, indique-le.

Réponds en français avec un formatage clair."""


GLOBAL_CORRECTION_PROMPT = """Tu es un expert paléographe français spécialisé dans les manuscrits cursifs du début du XXe siècle.

Tu reçois le résultat brut d'un OCR sur un manuscrit ancien. Ce texte contient beaucoup d'erreurs de reconnaissance.

Ta tâche est de RECONSTRUIRE un texte LISIBLE et COMPRÉHENSIBLE à partir de ce brouillon OCR :

1. CORRIGE les erreurs OCR évidentes (lettres mal reconnues, mots tronqués ou fusionnés)
2. RECONSTITUE les mots incomplets ou mal découpés
3. AJOUTE la ponctuation nécessaire pour la lisibilité (points, virgules)
4. FUSIONNE les lignes en paragraphes cohérents si nécessaire
5. GARDE le vocabulaire et les tournures d'époque (ne modernise pas le style)
6. Si un passage reste vraiment illisible, indique [illisible]

Le but est d'obtenir un texte que quelqu'un peut LIRE et COMPRENDRE facilement.

Réponds UNIQUEMENT avec le texte reconstitué, sans commentaire ni explication."""


def correct_full_text(raw_text: str, document_context: str = "", use_gpt4: bool = False) -> str:
    """
    Corrige un texte complet en une seule requête.

    Args:
        raw_text: Texte OCR complet à corriger
        document_context: Contexte général du document
        use_gpt4: Utiliser gpt-4o (meilleur) au lieu de gpt-4o-mini

    Returns:
        Texte corrigé complet
    """
    client = get_client()

    # Construire le prompt avec contexte
    system_prompt = GLOBAL_CORRECTION_PROMPT
    if document_context:
        system_prompt += f"\n\nCONTEXTE DU DOCUMENT :\n{document_context}"

    model = "gpt-4o" if use_gpt4 else "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reconstitue ce texte OCR en un texte lisible :\n\n{raw_text}"}
        ],
        temperature=0.4,
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()


def summarize_text(text: str, document_context: str = "", use_gpt4: bool = False) -> str:
    """
    Résume et analyse le texte transcrit.

    Args:
        text: Texte transcrit (brut ou corrigé)
        document_context: Contexte général du document
        use_gpt4: Utiliser gpt-4o au lieu de gpt-4o-mini

    Returns:
        Résumé structuré des plaintes
    """
    client = get_client()

    system_prompt = SUMMARY_PROMPT
    if document_context:
        system_prompt += f"\n\nCONTEXTE DU DOCUMENT :\n{document_context}"

    model = "gpt-4o" if use_gpt4 else "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyse et résume ce texte :\n\n{text}"}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content.strip()

import os
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# Charger les variables d'environnement
load_dotenv()

# Configuration des APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Clé API Gemini non trouvée. Veuillez la définir dans le fichier .env")
if not TAVILY_API_KEY:
    raise ValueError("Clé API Tavily non trouvée. Veuillez la définir dans le fichier .env")

# Initialiser les clients
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- Définition de l'état de l'agent ---
class AgentState(TypedDict):
    company_name: str
    report: str
    search_queries: List[str]
    research_data: List[Dict[str, str]]
    csv_data: List[Dict[str, str]]

# --- Nœuds du Graphe ---

def planning_node(state: AgentState):
    """
    Génère une liste de requêtes de recherche ciblées pour l'analyse concurrentielle.
    """
    print("--- ÉTAPE : PLANIFICATION DE LA RECHERCHE ---")
    company = state['company_name']
    
    prompt = f"""
    En tant qu'analyste stratégique, crée un plan de recherche pour une analyse concurrentielle complète de "{company}".
    Génère une liste de 5 à 7 requêtes de recherche précises pour l'API Tavily Search.
    Les requêtes doivent couvrir les aspects suivants :
    1.  **Identité et Positionnement :** Site officiel, slogan, et pages de réseaux sociaux.
    2.  **Services et Offres :** Principaux services, domaines d'expertise.
    3.  **Présence Locale  :** Informations de contact, dirigeants locaux, mandats spécifiques.
    4.  **Publications et Rapports :** Études de cas, livres blancs sur des sujets clés (IA, transfo. digitale, etc.).
    5.  **Clients et Projets :** Annonces de nouveaux clients, contrats ou partenariats.

    Réponds uniquement avec la liste des requêtes, une par ligne.
    """
    
    response = llm.invoke(prompt)
    queries = [q for q in response.content.strip().split('\n') if q]
    print(f"Requêtes de recherche planifiées :\n" + "\n".join(f"- {q}" for q in queries))
    
    return {"search_queries": queries}

def research_node(state: AgentState):
    """
    Exécute les requêtes de recherche planifiées en utilisant l'API Tavily.
    """
    print("--- ÉTAPE : EXÉCUTION DE LA RECHERCHE ---")
    queries = state['search_queries']
    all_results = []
    
    for query in queries:
        print(f"Recherche en cours pour : '{query}'")
        try:
            response = tavily_client.search(query=query, search_depth="advanced", max_results=3)
            results = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
            all_results.extend(results)
        except Exception as e:
            print(f"Erreur lors de la recherche pour la requête '{query}': {e}")
            
    print(f"Recherche terminée. {len(all_results)} résultats trouvés.")
    return {"research_data": all_results}

def report_node(state: AgentState):
    """
    Synthétise les données de recherche en un rapport d'analyse concurrentielle structuré
    et extrait les informations clés pour un fichier CSV.
    """
    print("--- ÉTAPE : GÉNÉRATION DU RAPPORT ET DES DONNÉES CSV ---")
    company = state['company_name']
    research_data = state['research_data']

    # Formatter les données de recherche pour le prompt
    formatted_data = "\n\n".join([f"Source URL: {res['url']}\nContent: {res['content'][:1000]}..." for res in research_data]) # Augmenter le contenu pour plus de contexte

    prompt = f"""
    En tant qu'analyste de marché expert, rédige un rapport d'analyse concurrentielle détaillé sur "{company}"
    en te basant exclusivement sur les données de recherche suivantes.

    Données de recherche brutes :
    ---
    {formatted_data}
    ---

    Instructions pour le rapport Markdown :
    1.  **Introduction :** Présente brièvement l'objectif du rapport.
    2.  **Positionnement et Valeurs :** Identifie le slogan, la mission et les valeurs de l'entreprise.
    3.  **Domaines d'Expertise :** Liste les principaux services et spécialités.
    4.  **Analyse de la Présence de {company} global :** Résume les informations relatives à la présence globale de {company} (dirigeants, projets, contact).
    5.  **Publications et Leadership Intellectuel :** Mentionne les thèmes des rapports ou études publiés.
    6.  **Clients et Projets Clés :** Identifie les clients ou partenariats notables mentionnés.
    7.  **Conclusion Synthétique :** Fournis une conclusion sur la stratégie globale de {company} basée sur les informations collectées.

    Le rapport final doit être structuré, professionnel, facile à lire et formaté en Markdown.
    Pour chaque section (Positionnement, Domaines d'Expertise, etc.), inclue une sous-section "Sources :" à la fin, listant les URLs pertinentes qui ont été utilisées pour cette section. Si une information provient d'une source spécifique, cite cette source directement.

    ---
    En plus du rapport Markdown, génère une structure JSON contenant les informations clés extraites, avec leurs catégories et les URLs des sources principales.
    Le format JSON doit être une liste d'objets, chaque objet ayant les clés "Catégorie", "Information", et "Source URL".
    Les catégories doivent être : "Identité", "Services", "Présence Locale", "Publications", "Clients".
    Si une information n'a pas de source directe ou claire, utilise "N/A" pour "Source URL".

    Exemple de format JSON :
    ```json
    [
        {{
            "Catégorie": "Identité",
            "Information": "Slogan: Building a better working world",
            "Source URL": "https://www.ey.com/en_gl"
        }},
        {{
            "Catégorie": "Services",
            "Information": "Audit, services en matière de changement climatique et de développement durable (CCaSS)",
            "Source URL": "https://www.ey.com/en_gl/assurance"
        }}
    ]
    ```
    Place le JSON entre les balises <JSON_DATA> et </JSON_DATA>.
    """

    response = llm.invoke(prompt).content
    
    # Séparer le rapport Markdown et les données JSON
    markdown_report = ""
    json_data_str = ""
    
    json_start_tag = "<JSON_DATA>"
    json_end_tag = "</JSON_DATA>"
    
    if json_start_tag in response and json_end_tag in response:
        parts = response.split(json_start_tag)
        markdown_report = parts[0].strip()
        
        if len(parts) > 1:
            json_part = parts[1].split(json_end_tag)
            json_data_str = json_part[0].strip()
            if len(json_part) > 1:
                markdown_report += json_part[1].strip() # Ajouter le reste du markdown si présent après le JSON
    else:
        markdown_report = response.strip() # Si pas de balises, tout est considéré comme markdown

    csv_data = []
    try:
        if json_data_str:
            # Nettoyer le JSON pour s'assurer qu'il est valide (parfois le LLM ajoute des caractères)
            # Supprimer les éventuels blocs de code Markdown autour du JSON
            if json_data_str.startswith("```json"):
                json_data_str = json_data_str[len("```json"):].strip()
            if json_data_str.endswith("```"):
                json_data_str = json_data_str[:-len("```")].strip()
            
            csv_data = json.loads(json_data_str)
            print("Données CSV extraites avec succès.")
        else:
            print("Aucune donnée JSON trouvée pour le CSV.")
    except json.JSONDecodeError as e:
        print(f"Erreur lors du parsing JSON pour le CSV : {e}")
        print(f"JSON brut tenté de parser : {json_data_str[:500]}...") # Afficher un extrait pour le débogage
        csv_data = [] # Assurez-vous que csv_data est une liste vide en cas d'erreur

    print("Rapport généré.")
    return {"report": markdown_report, "csv_data": csv_data}

# --- Construction du Graphe ---
workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_node)
workflow.add_node("research", research_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("planning")
workflow.add_edge("planning", "research")
workflow.add_edge("research", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# Pour des tests directs
if __name__ == '__main__':
    inputs = {"company_name": "EY"}
    final_state = app.invoke(inputs)
    
    print("\n--- RAPPORT FINAL ---")
    print(final_state['report'])

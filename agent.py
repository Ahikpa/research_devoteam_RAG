import os
import json
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# Charger les variables d'environnement
load_dotenv()

# Configuration des APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Une ou plusieurs clés API (Gemini, Tavily) sont manquantes dans le fichier .env")

# Initialiser les clients
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- Définition de l'état de l'agent (mis à jour) ---
class AgentState(TypedDict):
    company_name: str
    report: str
    search_queries: List[str]
    research_data: List[Dict[str, str]]
    specialized_research_data: Dict # Pour les données simulées
    validated_data: str # Pour la sortie du juge
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
    3.  **Présence Locale :** Informations de contact, dirigeants locaux, mandats spécifiques.
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
    print("--- ÉTAPE : EXÉCUTION DE LA RECHERCHE WEB (TAVILY) ---")
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
            
    print(f"Recherche web terminée. {len(all_results)} résultats trouvés.")
    return {"research_data": all_results}

def specialized_research_node(state: AgentState):
    """
    Simule la récupération de données factuelles depuis une API comme Crunchbase ou Owler.
    REMPLACEZ CECI PAR UN VRAI APPEL API QUAND VOUS AUREZ UNE CLÉ.
    """
    print("--- ÉTAPE : RECHERCHE SPÉCIALISÉE (SIMULATION) ---")
    company_name = state['company_name'].lower()
    
    # --- Base de données simulée ---
    mock_database = {
        "devoteam": {
            "nom": "Devoteam",
            "description_courte": "Société de conseil en technologies spécialisée dans la transformation digitale et le cloud.",
            "site_web": "https://www.devoteam.com/",
            "secteurs": ["IT Consulting", "Digital Transformation", "Cloud Computing", "Cybersecurity"],
            "total_financement_usd": 15000000,
            "date_creation": "1995-01-01",
            "concurrents": ["Capgemini", "Accenture", "Sopra Steria"]
        },
        "ey": {
            "nom": "Ernst & Young (EY)",
            "description_courte": "Un des plus grands réseaux de services professionnels au monde (Big Four).",
            "site_web": "https://www.ey.com/",
            "secteurs": ["Professional Services", "Audit", "Tax", "Consulting"],
            "total_financement_usd": 0, # Partenariat, pas de levées de fonds typiques
            "date_creation": "1989-01-01",
            "concurrents": ["PwC", "Deloitte", "KPMG"]
        }
    }
    
    # Récupérer les données pour l'entreprise demandée
    company_data = mock_database.get(company_name, {"error": f"Aucune donnée simulée trouvée pour {company_name}"})
    
    print(f"Données simulées récupérées pour {company_name}.")
    return {"specialized_research_data": company_data}

def judging_node(state: AgentState):
    """
    Agit comme un juge pour comparer, valider et synthétiser les informations.
    """
    print("--- ÉTAPE : JUGEMENT ET VALIDATION DES DONNÉES ---")
    company = state['company_name']
    web_data = state['research_data']
    specialized_data = state['specialized_research_data']

    formatted_web_data = "\n\n".join([f"Source URL: {res['url']}\nContent: {res['content'][:1000]}..." for res in web_data])
    formatted_specialized_data = json.dumps(specialized_data, indent=2, ensure_ascii=False)

    prompt = f"""
    En tant qu'analyste expert, tu es chargé de valider et de synthétiser des informations sur l'entreprise "{company}" à partir de deux sources distinctes.

    **Source 1 : Recherche Web Générale (Tavily)**
    Contient des informations qualitatives, des articles de presse, le positionnement marketing, etc.
    ---
    {formatted_web_data}
    ---

    **Source 2 : Base de Données Structurée (Simulation d'API)**
    Contient des faits précis et vérifiables (données financières, secteurs, date de création).
    ---
    {formatted_specialized_data}
    ---

    **Ta mission est de produire une synthèse validée qui servira de base unique pour le rapport final.**
    1.  **Priorise la Source 2 (Données simulées)** pour les données factuelles comme le total du financement, la date de création, les secteurs d'activité officiels.
    2.  **Utilise la Source 1 (Tavily)** pour comprendre le positionnement, le slogan, les services détaillés, les projets clients et la culture d'entreprise.
    3.  **Identifie et Résous les Contradictions.** Si les sources se contredisent, fais confiance à la Source 2 ou mentionne la divergence si elle est significative.
    4.  **Enrichis les Données.** Combine les informations pour créer une vue complète.
    5.  **Structure ta sortie.** Rédige un texte clair et cohérent, organisé par thèmes (Identité, Finances, Services, Positionnement, etc.). Ce texte sera la seule source de vérité pour le rapport final.

    Produis uniquement cette synthèse validée.
    """
    
    response = llm.invoke(prompt)
    validated_text = response.content.strip()
    print("Synthèse validée générée.")
    
    return {"validated_data": validated_text}

def report_node(state: AgentState):
    """
    Génère le rapport final et les données CSV à partir de la synthèse validée.
    """
    print("--- ÉTAPE : GÉNÉRATION DU RAPPORT FINAL ---")
    company = state['company_name']
    validated_data = state['validated_data']

    prompt = f"""
    En tant qu'analyste de marché, utilise la synthèse validée suivante pour créer un rapport d'analyse concurrentielle final sur "{company}".
    La synthèse a déjà été vérifiée et croisée à partir de plusieurs sources. Ta seule tâche est de la mettre en forme.

    Synthèse Validée :
    ---
    {validated_data}
    ---

    Instructions pour le rapport Markdown :
    1.  Structure le rapport avec des sections claires (Introduction, Positionnement, Domaines d'Expertise, etc.).
    2.  Assure-toi que le formatage Markdown est propre et professionnel.
    3.  N'ajoute PAS d'informations qui ne sont pas dans la synthèse fournie.

    ---
    En plus du rapport, génère une structure JSON contenant les informations clés extraites de la synthèse, avec leurs catégories.
    Le format JSON doit être une liste d'objets (Catégorie, Information). La source n'est plus nécessaire car les données sont déjà validées.

    Exemple de format JSON :
    ```json
    [
        {{"Catégorie": "Identité", "Information": "Slogan: Building a better working world"}},
        {{"Catégorie": "Services", "Information": "Audit, services en matière de changement climatique et de développement durable (CCaSS)"}}
    ]
    ```
    Place le JSON entre les balises <JSON_DATA> et </JSON_DATA>.
    """

    response = llm.invoke(prompt).content
    
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
    else:
        markdown_report = response.strip()

    csv_data = []
    try:
        if json_data_str:
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
        csv_data = []

    print("Rapport final généré.")
    return {"report": markdown_report, "csv_data": csv_data}

# --- Construction du Graphe (mis à jour) ---
workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_node)
workflow.add_node("research", research_node)
workflow.add_node("specialized_research", specialized_research_node)
workflow.add_node("judging", judging_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("planning")
workflow.add_edge("planning", "research")
workflow.add_edge("research", "specialized_research")
workflow.add_edge("specialized_research", "judging")
workflow.add_edge("judging", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# Pour des tests directs
if __name__ == '__main__':
    # Utiliser une entreprise présente dans la base de données simulée pour le test
    inputs = {"company_name": "devoteam"} 
    final_state = app.invoke(inputs)
    
    print("\n--- RAPPORT FINAL ---")
    print(final_state['report'])
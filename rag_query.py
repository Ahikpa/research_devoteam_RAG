import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration et Initialisation ---
load_dotenv()

# Charger les clés API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Clé API Gemini non trouvée. Veuillez la définir dans le fichier .env")

# Constantes
DB_FAISS_PATH = 'vectorstore/db_faiss'
DOCS_PATH = '.' # Le répertoire courant

# Initialiser le LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# --- Fonctions du Système RAG ---

def create_vector_db():
    """
    Charge les documents, les divise, crée les embeddings et les stocke dans FAISS.
    """
    print("--- Démarrage de la création de la base de données vectorielle ---")

    # Charger les documents Markdown et CSV
    md_files = glob.glob(os.path.join(DOCS_PATH, '*.md'))
    csv_files = glob.glob(os.path.join(DOCS_PATH, '*.csv'))
    
    loaders = []
    if md_files:
        loaders.append(DirectoryLoader(DOCS_PATH, glob='*.md', loader_cls=UnstructuredMarkdownLoader, show_progress=True))
    if csv_files:
        for filename in csv_files:
            loaders.append(CSVLoader(file_path=filename, encoding="utf-8", csv_args={'delimiter': ','}))

    if not loaders:
        print("Aucun document .md ou .csv trouvé. Le script va s'arrêter.")
        return None

    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    print(f"Nombre de documents chargés : {len(documents)}")

    # Diviser les documents en fragments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"Nombre de fragments de texte créés : {len(texts)}")

    # Créer les embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Créer et sauvegarder la base de données FAISS
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"--- Base de données vectorielle créée et sauvegardée dans '{DB_FAISS_PATH}' ---")
    return db

def get_qa_chain():
    """
    Charge la base de données vectorielle et initialise la chaîne de questions-réponses.
    """
    print("--- Initialisation de la chaîne de Q&A ---")
    
    # Créer les embeddings (doit correspondre à ceux utilisés pour la création)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    # Charger la base de données FAISS
    if not os.path.exists(DB_FAISS_PATH):
        print("Base de données non trouvée. Veuillez d'abord la créer.")
        return None, None
        
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Base de données vectorielle chargée.")

    # Créer le retriever
    retriever = db.as_retriever(search_kwargs={'k': 3}) # Récupère les 3 fragments les plus pertinents

    # Créer la chaîne de Q&A
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_custom_prompt()}
    )
    print("--- Chaîne de Q&A prête ---")
    return chain, db

def get_custom_prompt():
    """
    Crée un prompt personnalisé pour guider le LLM.
    """
    from langchain.prompts import PromptTemplate
    custom_prompt_template = """
    Utilise les informations suivantes pour répondre à la question de l'utilisateur.
    Si tu ne connais pas la réponse, dis simplement "Je n'ai pas trouvé d'information à ce sujet dans les documents fournis.", n'essaie pas d'inventer une réponse.
    Sois aussi concis et précis que possible.

    Contexte : {context}
    Question : {question}

    Réponse utile :
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


# --- Point d'Entrée Principal ---

if __name__ == '__main__':
    # Créer la base de données si elle n'existe pas
    if not os.path.exists(DB_FAISS_PATH):
        create_vector_db()

    # Obtenir la chaîne de Q&A
    qa_chain, db = get_qa_chain()

    if qa_chain:
        print("\n--- Système de dialogue RAG ---")
        print("Posez vos questions sur les analyses concurrentielles. Tapez 'exit' ou 'quit' pour quitter.")
        
        while True:
            query = input("\nVotre question : ")
            if query.lower() in ['exit', 'quit']:
                break
            
            # Exécuter la chaîne
            result = qa_chain({"query": query})
            
            # Afficher la réponse
            print("\nRéponse :")
            print(result["result"])
            
            # (Optionnel) Afficher les sources
            print("\nSources utilisées :")
            for doc in result["source_documents"]:
                # Tenter d'extraire le nom du fichier source
                source_name = doc.metadata.get('source', 'Source inconnue')
                print(f"- {os.path.basename(source_name)}")


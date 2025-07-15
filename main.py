import os
import csv
from agent import app
from datetime import datetime

def main():
    """
    Point d'entrée principal pour lancer l'agent d'analyse concurrentielle.
    """
    company_to_analyze = "devoteam"
    
    print(f"--- Lancement de l'analyse concurrentielle pour : '{company_to_analyze}' ---")
    
    # Définir l'input pour l'agent
    inputs = {"company_name": company_to_analyze}
    
    print("Traitement en cours... L'agent collecte et synthétise les informations.")
    
    # Exécuter l'agent
    final_state = app.invoke(inputs)
    
    # Extraire le rapport final et les données CSV
    report_content = final_state.get('report', "Le rapport n'a pas pu être généré.")
    csv_data = final_state.get('csv_data', [])
    
    # Afficher le rapport dans la console
    print("\n--- RAPPORT DE SYNTHÈSE FINAL ---")
    print(report_content)
    
    # Sauvegarder le rapport Markdown dans un fichier
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"analyse_concurrentielle_{company_to_analyze.lower()}_{timestamp}.md"
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"\n--- Rapport Markdown sauvegardé avec succès dans le fichier : {report_filename} ---")
        print(f"Chemin absolu : {os.path.abspath(report_filename)}")
        
    except Exception as e:
        print(f"\n--- Erreur lors de la sauvegarde du rapport Markdown : {e} ---")

    # Sauvegarder les données CSV dans un fichier
    if csv_data:
        try:
            csv_filename = f"analyse_concurrentielle_{company_to_analyze.lower()}_{timestamp}.csv"
            # Déterminer les en-têtes du CSV à partir des clés du premier dictionnaire
            headers = csv_data[0].keys() if csv_data else []
            
            with open(csv_filename, "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_data)
                
            print(f"\n--- Données CSV sauvegardées avec succès dans le fichier : {csv_filename} ---")
            print(f"Chemin absolu : {os.path.abspath(csv_filename)}")
            
        except Exception as e:
            print(f"\n--- Erreur lors de la sauvegarde des données CSV : {e} ---")
    else:
        print("\n--- Aucune donnée CSV à sauvegarder. ---")

if __name__ == "__main__":
    main()

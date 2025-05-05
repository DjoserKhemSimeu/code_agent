from flask import Flask, render_template, request, jsonify
from huggingface_hub import login
from transformers import ReactCodeAgent,CodeAgent, HfApiEngine, tool
from ragatouille import RAGPretrainedModel
import pandas as pd
import torch
import re



# Initialisation du moteur LLM et de l'agent
login("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")
torch.cuda.empty_cache()
llm_engine = HfApiEngine(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
agent = ReactCodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=False, additional_authorized_imports=["csv"])
RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2-64")
RAG_fashion = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2-64")

# Chargement des données de la collection à partir d'un fichier TSV
collection_data = pd.read_csv("Data/collection.tsv", delimiter='\t', header=None, names=["id", "text"])
fashion_data = pd.read_csv("Data/fashion_advice.tsv", delimiter='\t', header=None, names=["id", "text"])
docs = collection_data["text"].tolist()
docs_fashion = fashion_data["text"].tolist()
RAG.index(docs, index_name="demo")
torch.cuda.empty_cache()
RAG_fashion.index(docs_fashion, index_name="advice")
torch.cuda.empty_cache()
@tool
def clothes_finder_RAG(query: str) -> str:
    """
    Cette fonction recherche des articles de vêtements pertinents en fonction d'une requête donnée.
    Elle utilise une collection pré-indexée de descriptions de vêtements pour trouver des correspondances.

    Args:
        query: La requête de recherche décrivant l'article de vêtement.
    Output:
        Une chaîne contenant une liste classée de 2 éléments.
    """
     
    torch.cuda.empty_cache()

    results = RAG.search(query)
    contents = [result['content'] for result in results]
    combined_contents = "\n".join(contents[:2])
    return combined_contents

agent.toolbox.add_tool(clothes_finder_RAG)


@tool
def fashion_advisor_RAG(query: str) -> str:
    """
    Cette fonction recherche des conseils en mode pertinents en fonction d'une requête donnée.
    Elle utilise une collection pré-indexée de descriptions de conseils en mode pour trouver des correspondances.

    Args:
        query: La requête de recherche décrivant le conseil en mode recherché.
    Output:
        Une chaîne contenant une liste classée de 2 éléments.
    """
     
    torch.cuda.empty_cache()

    results = RAG_fashion.search(query)
    contents = [result['content'] for result in results]
    combined_contents = "\n".join(contents[:2])
    return combined_contents

agent.toolbox.add_tool(fashion_advisor_RAG)
 
# Liste pour stocker les messages de la conversation
conversation = [{"role": "agent", "text": "Bonjour ! Je suis CelIA, votre agent IA spécialisé dans la mode. Je vous aide à trouver des looks tendance, des conseils stylés et à explorer l'univers de la mode selon vos envies. À vos côtés pour une expérience mode sur mesure ! 👗✨", "rag_results": []}]
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('chat.html', messages=conversation)

@app.route('/send_message', methods=['POST'])

def send_message():
    user_input = request.form['message']
    if user_input.lower() == 'exit':
        conversation.append({"role": "user", "text": "exit", "rag_results": []})
        conversation.append({"role": "agent", "text": "Au revoir !", "rag_results": []})
        return jsonify({"status": "exit"})

    conversation.append({"role": "user", "text": user_input, "rag_results": []})

    torch.cuda.empty_cache()  # Libérer la mémoire GPU
    result = agent.run(
        user_input,
        return_generated_code=False,
        user_style="fashion addict",
        output_format = "little sentence to introduce the output+ '\n' + output of the RAG"
    )
    torch.cuda.empty_cache()  # Libérer la mémoire GPU après l'exécution
    
    # Analyser l'output pour détecter les résultats du RAG
    rag_results,text = extract_rag_results(result)


    conversation.append({"role": "agent", "text": text, "rag_results": rag_results})
    return jsonify({"status": "ok", "messages": conversation})
def extract_rag_results(text):
    # Utiliser une expression régulière pour extraire les résultats du RAG
    pattern = re.compile(r"img_(\d+);([^;]+);[^;]+;[^;]+;[^;]+;[^;]+;[^;]+;([^;]+);[^;]+;[^;]+;[^;]+;[^;]+;[^;]+;([^;]+);[^;]+;[^;]+;[^;]+;[^;]+;(.+)")
    matches = pattern.findall(text)
 
    rag_results = []
    for match in matches:
        img_number = match[0]
        boutique = match[1]
        description = match[4]
        if boutique == "elvira":
            image_path = f"/static/images/elvira/img_{img_number}.jpeg"
        else:
            image_path = f"/static/images/{boutique}/img_{img_number}.jpg"
        rag_results.append({"image_path": image_path, "description": description})

    # Supprimer les séquences repérées du texte
    text = pattern.sub("", text)
    print("Text:",text)
    return rag_results ,text


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    app.run(host='0.0.0.0', port=8080)

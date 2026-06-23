import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Base pequena para enxergar o RAG funcionando na prática durante a aula.
documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]

# O modelo de embeddings transforma texto em vetor, capturando o significado da frase.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Para rodar, preciso ter a variável GROQ_API_KEY configurada no ambiente.
client = Groq()

# Aqui eu gero os embeddings antes da pergunta chegar, como se fosse a indexação da base.
doc_embeddings = model.encode(documents)
doc_embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, top_k=3):
    # A pergunta também vira embedding para ser comparada com os documentos.
    query_embedding = model.encode([query])[0]

    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim))

    # Aqui acontece o retrieval: pego os trechos mais parecidos com a pergunta.
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [(documents[i], sim) for i, sim in similarities[:top_k]]


def generate_answer(query, retrieve_docs):
    # O contexto recuperado é enviado junto com a pergunta para guiar a resposta do LLM.
    context = "\n".join([doc for doc, _ in retrieve_docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Você é um especialista em machine learning. Use apenas o contexto fornecido para responder as perguntas.",
            },
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def rag(query, top_k=3):
    # RAG = primeiro recupera contexto, depois gera a resposta.
    retrieved = retrieve(query, top_k)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


# Execução do exemplo da aula.
answer, docs = rag("O que é machine learning?")

print("\n[Resposta gerada pelo LLM]")
print(answer)

print("\n[Anotação da aula] Documentos recuperados pelo retriever:")
for doc, similarity in docs:
    print(f" - {similarity:.3f}: {doc}")

"""
Anotações de saída esperada:

[Resposta gerada pelo LLM]
Machine learning é um campo da inteligência artificial que permite que computadores aprendam
padrões a partir de dados e melhorem seu desempenho sem serem explicitamente programados.

[Anotação da aula] Documentos recuperados pelo retriever:
 - 0.xxx: Machine learning é um campo da inteligência artificial...
 - 0.xxx: O aprendizado de máquina dá aos sistemas...
 - 0.xxx: Em vez de seguir apenas regras fixas...

Observação:
Os valores de similaridade podem mudar um pouco dependendo da versão das bibliotecas/modelo.
"""
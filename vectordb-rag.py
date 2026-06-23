from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Base pequena para demonstrar o fluxo RAG usando uma vector database.
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

client = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Aqui o Qdrant entra como a base vetorial.
# Usei persistência em disco para simular uma base real, não apenas memória.
# qdrant = QdrantClient(":memory:")
qdrant = QdrantClient(path="db/data")

vector_size = model.get_sentence_embedding_dimension()

# A collection define o tamanho dos vetores e a métrica usada para comparar similaridade.
# Como a base está em disco, apago a collection anterior para conseguir repetir a aula do zero.
if qdrant.collection_exists(collection_name="ml_documents"):
    qdrant.delete_collection(collection_name="ml_documents")

qdrant.create_collection(
    collection_name="ml_documents",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

points = []
for idx, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    points.append(PointStruct(id=idx, vector=embedding, payload={"text": doc}))

# Nesta etapa eu salvo os documentos vetorizados no Qdrant.
qdrant.upsert(collection_name="ml_documents", points=points, wait=True)


def retrieve(query, top_k=3):
    # A pergunta também vira vetor para o Qdrant encontrar os textos mais parecidos.
    query_embedding = model.encode(query).tolist()
    search_result = qdrant.query_points(
        collection_name="ml_documents",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return [(hit.payload["text"], hit.score) for hit in search_result.points]


def generate_answer(query, retrieve_docs):
    # O contexto vem dos pontos recuperados no Qdrant e guia a resposta do LLM.
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
    # Fluxo completo: busca semântica no Qdrant e geração da resposta com Groq.
    retrieved = retrieve(query, top_k)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


# Execução do exemplo da aula.
answer, docs = rag("O que é machine learning?")

print("\n[Resposta gerada pelo LLM]")
print(answer)

print("\n[Anotação da aula] Documentos recuperados no Qdrant:")
for doc, similarity in docs:
    print(f" - {similarity:.3f}: {doc}")

"""
Anotações de saída esperada:

[Resposta gerada pelo LLM]
Machine learning é um campo da inteligência artificial que permite que computadores aprendam
padrões a partir de dados, melhorem seu desempenho e descubram relações nos dados.

[Anotação da aula] Documentos recuperados no Qdrant:
 - 0.xxx: Machine learning é um campo da inteligência artificial...
 - 0.xxx: O aprendizado de máquina dá aos sistemas...
 - 0.xxx: Em vez de seguir apenas regras fixas...

Observação:
O Qdrant fica responsável por guardar os embeddings e fazer a busca por similaridade.
Os valores de score podem mudar um pouco dependendo da versão das bibliotecas/modelo.
"""
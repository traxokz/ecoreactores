from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from :class:`~langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter

# Configura tu LLM (cambio sugerido si usas GPT-4-all)
llm = OpenAI()

# Cargar documentos (asegúrate de convertirlos a texto previamente)
with open("chatbot/normas/abc_pot.txt", "r", encoding="utf-8") as f:
    texto_documento = f.read()

# Dividir el texto en fragmentos para la búsqueda
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
textos_divididos = text_splitter.split_text(texto_documento)

# Crear un índice de búsqueda
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(textos_divididos, embeddings)

# Cadena de pregunta-respuesta
chain = load_qa_chain(llm, chain_type="stuff")

# Función para hacer preguntas al chatbot
def preguntar_al_chatbot(pregunta):
    docs = vectorstore.similarity_search(pregunta)
    respuesta = chain.run(input_documents=docs, question=pregunta)
    return respuesta

# Ejemplo de uso
pregunta = "¿Cuál es la normativa sobre residuos orgánicos en Bogotá?"
print(preguntar_al_chatbot(pregunta))

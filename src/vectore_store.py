from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
def store_func():
    embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')
    print("INITIALZING VECTOR STORE....")

    vector_store = Chroma(
        collection_name = "test_main",
        embedding_function= embeddings,
        persist_directory= "./chroma_db"
    )

    return vector_store
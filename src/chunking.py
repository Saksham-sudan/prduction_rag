from langchain_text_splitters import CharacterTextSplitter
from ingestion import ingestion_func

def chunking_func():
    document = ingestion_func()
    print("CHUNKING DOCUMENT....")
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap = 100, separator="\n")
    documents = text_splitter.split_documents(document)

    return documents
from src.vectore_store import store_func
from langchain.tools import tool

store = store_func()

@tool(response_format= 'content_and_artifact')
def retriver_func(input_query: str) -> tuple[str, list[str]]:
    """Searches the input keyword from the vector database and return relevant context regarding the keyword.

    Args:
        input_query: serach keyword to look for
    """
    retrived_final = []
    print("RETRIVAL.....")

    retrived = store.similarity_search(input_query, k=7)
    for i in range(len(retrived)): 
        retrived_content = retrived[i].page_content
        retrived_final.append(retrived_content)
        
    return str(retrived_final), retrived_final
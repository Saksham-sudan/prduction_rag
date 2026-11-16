import time
from chunking import chunking_func
from vectore_store import store_func

def embedding_func():
    chunks = chunking_func()
    db = store_func()
    batch = []

    print("EMBEDDING DOCUMENT....")
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == 3:
            print("Embedding batch")
            db.add_documents(documents= batch)
            batch = []
            time.sleep(2.0)
            
    if len(batch) > 0:
        db.add_documents(documents=batch)

embedding_func()
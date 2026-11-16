import sys
import types
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import sys
def silence_unraisable_hook(unraisable):
    if unraisable.object is not None and "__del__" in str(unraisable.object):
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = silence_unraisable_hook

from ragas.testset import TestsetGenerator
from src.ingestion import ingestion_func
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

document = ingestion_func()
llm = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite'
)
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')

generator = TestsetGenerator.from_langchain(llm, embedding)
testset = generator.generate_with_langchain_docs(testset_size = 9, documents= document)


df = testset.to_pandas()
df.to_csv(r"C:\Users\saksh\Documents\projects\Production RAG\eval\eval_testset.csv")
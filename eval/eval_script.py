import sys
import types
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import pandas as pd
from ragas.metrics import (context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    context_entity_recall,
    NoiseSensitivity
    )
from langchain_google_genai import ChatGoogleGenerativeAI
from src.llm_call import call_func
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain.rate_limiters import InMemoryRateLimiter

df = pd.read_csv(r"C:\Users\saksh\Documents\projects\Production RAG\eval\eval_testset.csv")
def eval_data():
    retrived_list = []
    answers_list = []
    for i in range(len(df)):
        response = call_func(df.user_input[i])
        retrived = response[1]
        answers = response[0].text
        retrived_list.append(retrived)
        answers_list.append(answers)
    data = {
        "question": df.user_input,
        "contexts": retrived_list,
        "answer": answers_list,
        "ground_truth": df.reference
    }
    data = Dataset.from_dict(data)
    return data

def eval_func():
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.22, 
        check_every_n_seconds=0.1,
        max_bucket_size=1
    )

    ds = eval_data()
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite', rate_limiter=rate_limiter), bypass_n = True)
    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', rate_limiter=rate_limiter))

    run_config = RunConfig(
        timeout= 2880,
        max_retries= 5
    )

    metrics =[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        context_entity_recall,
        NoiseSensitivity()
    ]
    result = evaluate(
        dataset=ds,
        metrics= metrics,
        llm= llm,
        embeddings= embeddings,
        run_config= run_config
    )
    df = result.to_pandas()
    return df

df = eval_func()
df.to_csv(r"C:\Users\saksh\Documents\projects\Production RAG\eval\eval_result.csv")
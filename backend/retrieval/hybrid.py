from __future__ import annotations

from typing import Callable, List, Optional

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain.schema.runnable import Runnable
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from qdrant_client import models

from backend.retrieval.cache import LimitedSemanticCache

DEFAULT_PROMPT = """
**Role**
    - you are a helpful assistant and your role is to answer user questions with the provided documents.
**Objective** 
    - rely solely on the supplied context.
**Expected solution**
    - answer with three sections: answer, explanation, sources.
    
question : {question}
context : {context}
response :
"""


def build_hybrid_retriever(
    vectorstore: QdrantVectorStore,
    k: int = 50,
) -> ContextualCompressionRetriever:
    reranker = FlashrankRerank()
    base_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "hybrid_fusion": models.FusionQuery(fusion=models.Fusion.RRF),
        }
    )
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('pageno')})\n{doc.page_content}"
            for doc in docs
        ]
    )


def build_qa_chain(
    retriever: ContextualCompressionRetriever,
    llm,
    cache: Optional[LimitedSemanticCache] = None,
    top_k: int = 20,
    prompt_template: str = DEFAULT_PROMPT,
) -> Runnable:
    def retrieve(query: str) -> List[Document]:
        if cache:
            cached = cache.get(query)
            if cached:
                return cached[:top_k]
        docs = retriever.invoke(query)
        docs = docs[:top_k]
        if cache:
            cache.put(query, docs)
        return docs

    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    chain = (
        {
            "context": RunnableLambda(retrieve) | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


__all__ = ["build_hybrid_retriever", "build_qa_chain", "DEFAULT_PROMPT"]

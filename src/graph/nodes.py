from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.api.schemas import SearchResult
from src.config import settings
from src.graph.state import (
    BatchDocumentGradeOutput,
    DocumentGradeOutput,
    GraphState,
    QueryAnalysisOutput,
)
from src.llm.models import get_llm
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("graph_nodes")


class RAGNodes:
    """Encapsulates the execution logic for all nodes in the Self-Corrective LangGraph workflow."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm = llm or get_llm()

    def query_analysis_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: Analyzes the raw question, identifies intent, and produces an optimized search query."""
        raw_question = state.get("question", "").strip()
        logger.info(f"--- [Node: Query Analysis] Analyzing question: '{raw_question}' ---")

        system_prompt = (
            "You are an expert technical query analyzer for an AI documentation assistant. "
            "Analyze the user's question and produce:\n"
            "1. 'intent': Classify into one of ['conceptual', 'how-to', 'troubleshooting', 'api_reference'].\n"
            "2. 'search_query': Rewrite unclear or conversational phrasing into an expanded, keyword-rich "
            "search query tailored for semantic similarity search in technical library docs."
        )
        user_prompt = f"User Question: {raw_question}"

        structured_llm = self.llm.with_structured_output(QueryAnalysisOutput)
        try:
            analysis: QueryAnalysisOutput = structured_llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            intent = analysis.intent
            search_query = analysis.search_query.strip() or raw_question
        except Exception as e:
            logger.warning(f"Query analysis LLM failed ({e}). Falling back to raw question.")
            intent = "conceptual"
            search_query = raw_question

        logger.info(f"Query Intent: '{intent}' | Refined Search Query: '{search_query}'")
        return {
            "search_query": search_query,
            "query_intent": intent,
        }

    def retrieval_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Searches ChromaDB using the analyzed query and retrieves top-k chunks."""
        search_query = state.get("search_query") or state.get("question", "")
        logger.info(f"--- [Node: Retrieval] Searching ChromaDB for: '{search_query}' ---")

        results: List[SearchResult] = self.vector_store.similarity_search(
            query=search_query,
            top_k=settings.DEFAULT_TOP_K,
            score_threshold=settings.SIMILARITY_SCORE_THRESHOLD,
        )

        logger.info(f"Retrieved {len(results)} chunks from vector store.")
        return {
            "documents": results,
        }

    def document_grading_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Evaluates retrieved chunks for relevance to the question (Self-Corrective RAG).

        Uses single-pass multi-document relevance evaluation for high throughput and zero rate-limiting.
        """
        question = state.get("question", "")
        documents = state.get("documents", [])
        logger.info(f"--- [Node: Document Grading] Grading {len(documents)} retrieved chunks ---")

        if not documents:
            logger.warning("No documents retrieved to grade.")
            return {"relevant_documents": []}

        system_prompt = (
            "You are a strict technical documentation relevance evaluator. "
            "Given the user's question and a set of candidate document chunks numbered [Chunk 1], [Chunk 2], etc., "
            "determine which chunk indices contain information, concepts, or code relevant to answering the question. "
            "Return the 1-based indices in 'relevant_indices' (e.g. [1, 3]), or an empty list [] if none are relevant. "
            "Provide a concise explanation in 'reasoning'."
        )

        chunks_text = "\n\n".join(
            f"--- [Chunk {idx}] (from {doc.metadata.source_filename}) ---\n{doc.content}"
            for idx, doc in enumerate(documents, 1)
        )
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Candidate Document Chunks:\n{chunks_text}\n\n"
            "Evaluate each chunk and return the list of relevant chunk numbers in 'relevant_indices'."
        )

        relevant_docs: List[SearchResult] = []
        try:
            structured_grader = self.llm.with_structured_output(BatchDocumentGradeOutput)
            grade: BatchDocumentGradeOutput = structured_grader.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            rel_indices = set(grade.relevant_indices or [])
            for idx, doc in enumerate(documents, 1):
                if idx in rel_indices:
                    logger.info(f"Chunk #{idx} ({doc.metadata.source_filename}) -> RELEVANT")
                    relevant_docs.append(doc)
                else:
                    logger.info(f"Chunk #{idx} ({doc.metadata.source_filename}) -> IRRELEVANT")
            logger.info(f"Grading verdict: {len(relevant_docs)}/{len(documents)} relevant ({grade.reasoning})")
        except Exception as e:
            logger.warning(f"Single-pass grading failed ({e}). Falling back to per-chunk grading.")
            try:
                per_chunk_grader = self.llm.with_structured_output(DocumentGradeOutput)
                for idx, doc in enumerate(documents, 1):
                    item_prompt = f"User Question: {question}\n\nDocument Chunk ({doc.metadata.source_filename}):\n{doc.content}"
                    single_grade: DocumentGradeOutput = per_chunk_grader.invoke(
                        [SystemMessage(content="Grade if chunk contains relevant context."), HumanMessage(content=item_prompt)]
                    )
                    if single_grade.is_relevant:
                        relevant_docs.append(doc)
            except Exception as inner_e:
                logger.warning(f"Per-chunk fallback also failed ({inner_e}). Retaining all chunks safely.")
                relevant_docs = list(documents)

        logger.info(
            f"Grading complete: {len(relevant_docs)} of {len(documents)} chunks marked relevant."
        )
        return {
            "relevant_documents": relevant_docs,
        }

    def query_rewrite_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Reformulates the search query when retrieved chunks are irrelevant."""
        question = state.get("question", "")
        current_query = state.get("search_query", question)
        current_retries = state.get("retry_count", 0)
        new_retries = current_retries + 1

        logger.info(
            f"--- [Node: Query Rewrite] Retry attempt {new_retries}/{state.get('max_retries', settings.MAX_RETRIES)} ---"
        )

        system_prompt = (
            "You are an expert technical search query optimizer. "
            "The previous search query failed to retrieve relevant documentation chunks. "
            "Rewrite the search query to improve retrieval quality. Add relevant technical synonyms, "
            "standard library or framework terminology, and core concepts. "
            "Respond ONLY with the rewritten search query text."
        )
        user_prompt = (
            f"Original Question: {question}\n"
            f"Previous Search Query: {current_query}\n\n"
            "Rewritten Search Query:"
        )

        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            rewritten_query = content.strip().replace('"', "")
            if not rewritten_query:
                rewritten_query = f"{current_query} documentation guide"
        except Exception as e:
            logger.warning(f"Query rewrite LLM call failed ({e}). Appending fallback keywords.")
            rewritten_query = f"{current_query} documentation guide"

        logger.info(f"Rewritten Query: '{rewritten_query}'")
        return {
            "search_query": rewritten_query,
            "retry_count": new_retries,
        }

    def generation_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Generates a clear, cited answer grounded strictly in relevant retrieved context."""
        question = state.get("question", "")
        relevant_docs = state.get("relevant_documents", [])
        logger.info(f"--- [Node: Generation] Generating answer with {len(relevant_docs)} relevant docs ---")

        # Fallback path if no relevant documents were found after retries
        if not relevant_docs:
            fallback_text = (
                f"I am sorry, but the provided technical documentation does not contain sufficient "
                f"information to answer your question regarding '{question}'. "
                f"Please consult the official documentation directly or rephrase your question."
            )
            logger.info("No relevant context available. Returned safe fallback response.")
            return {
                "generation": fallback_text,
                "citations": [],
                "is_fallback": True,
            }

        # Format context and collect citations
        context_blocks = []
        citations_set = set()
        for idx, doc in enumerate(relevant_docs, 1):
            citations_set.add(doc.metadata.source_filename)
            context_blocks.append(
                f"[Source #{idx}: {doc.metadata.source_filename} - {doc.metadata.document_title}]\n"
                f"{doc.content}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)
        citations_list = sorted(list(citations_set))

        system_prompt = (
            "You are a helpful, precise technical documentation assistant. "
            "Answer the user's question using ONLY the provided technical documentation context. "
            "Rules:\n"
            "1. Ground your answer strictly in the provided context. Do NOT invent or extrapolate facts.\n"
            "2. If the context does not contain enough information, state that clearly.\n"
            "3. Explicitly cite your sources by referencing the source filename(s) "
            "(e.g., [fastapi_overview.md])."
        )
        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"User Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            answer = response.content if isinstance(response.content, str) else str(response.content)
            answer_text = answer.strip()
        except Exception as e:
            logger.error(f"Generation LLM call failed: {e}")
            answer_text = (
                f"An error occurred while generating the answer from the retrieved context. "
                f"Relevant sources: {', '.join(citations_list)}."
            )

        logger.info(f"Answer generated successfully ({len(answer_text)} chars, citations: {citations_list}).")
        return {
            "generation": answer_text,
            "citations": citations_list,
            "is_fallback": False,
        }


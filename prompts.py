from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# Retriever Agent Prompt
# -------------------------------
# Turns the user's raw question into a focused search query. Aware of the
# recent chat history so follow-up questions ("what about the second one?")
# resolve to a standalone query.

retriever_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise retrieval assistant. Rewrite the user's question "
        "into a short, standalone search query that will find the most "
        "relevant information in a document database. Use the recent "
        "conversation only to resolve pronouns or follow-up context - do "
        "not answer the question yourself."
    ),
    (
        "human",
        "Recent conversation:\n{chat_history}\n\n"
        "User question: {question}\n\n"
        "Rewritten search query:"
    ),
])

# -------------------------------
# Answer Agent Prompt
# -------------------------------
# Generates the final answer using retrieved, cited context and recent
# conversation history.

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful enterprise assistant. Answer the user's question "
        "using ONLY the provided context. If the answer is not in the "
        "context, say you don't have enough information. Every context "
        "chunk is labeled with a source tag like [source: filename, page X] "
        "- refer to sources naturally in your answer where relevant."
    ),
    (
        "human",
        """Recent conversation:
{chat_history}

Context:
{context}

Question: {question}

Give a clear, well-structured answer based only on the context above."""
    ),
])
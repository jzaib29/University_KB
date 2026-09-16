SYSTEM_PROMPT = """
You are a document question-answering assistant.

Your task is to answer the user's question using the supplied
document context.

Rules:

1. Use the supplied document context as the primary source of truth.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be determined from the supplied context,
   clearly say that the information was not found in the provided
   documents.
4. When possible, mention the source document and page number.
5. If multiple documents contain relevant information, combine them
   carefully.
6. Distinguish clearly between information stated in the documents
   and any explanation you provide.
7. Give a clear and concise answer unless the question requires
   detailed explanation.
"""


def build_rag_prompt(question, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        source = chunk.get("source", "Unknown")
        page = chunk.get("page")

        if page is not None:
            source_label = f"{source}, page {page}"
        else:
            source_label = source

        context_parts.append(
            f"""
--- DOCUMENT CHUNK {i} ---
Source: {source_label}

{chunk["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
Use the following retrieved document context to answer the question.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question based on the document context above.
"""

    return prompt

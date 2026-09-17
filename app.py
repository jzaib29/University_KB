import streamlit as st

from src.retriever import FAISSRetriever
from src.groq_client import GroqService
from src.prompts import SYSTEM_PROMPT, build_rag_prompt


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="NUST Student Knowledge Base AI Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY was not found. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Load FAISS retriever
# ---------------------------------------------------------

@st.cache_resource
def load_retriever():
    return FAISSRetriever()


@st.cache_resource
def load_groq_client(api_key):
    return GroqService(api_key)


try:
    retriever = load_retriever()
    groq_service = load_groq_client(GROQ_API_KEY)

except Exception as e:
    st.error(f"Application initialization failed: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=5
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=True
    )

    st.divider()

    st.caption(
        "LLM: GPT-OSS 120B via Groq"
    )

    st.caption(
        f"Vectors loaded: {retriever.index.ntotal}"
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📚 NUST Student Knowledge Base AI Assistant")

st.write(
    "Ask questions about the documents contained "
    "in the knowledge base."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the documents..."
)


if question:

    # Save/display user question

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # -----------------------------------------------------
    # Retrieve relevant document chunks
    # -----------------------------------------------------

    with st.spinner("Searching documents..."):

        try:

            results = retriever.search(
                query=question,
                k=top_k
            )

        except Exception as e:

            st.error(
                f"Document retrieval failed: {e}"
            )

            st.stop()


    # -----------------------------------------------------
    # Build RAG prompt
    # -----------------------------------------------------

    rag_prompt = build_rag_prompt(
        question,
        results
    )


    # -----------------------------------------------------
    # Ask Groq
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:

                answer = groq_service.generate_answer(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=rag_prompt
                )

            except Exception as e:

                st.error(
                    f"Groq API request failed: {e}"
                )

                st.stop()


        st.markdown(answer)


        # -------------------------------------------------
        # Display retrieved sources
        # -------------------------------------------------

        if show_sources:

            with st.expander(
                "Retrieved sources"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    source = result.get(
                        "source",
                        "Unknown"
                    )

                    page = result.get("page")

                    score = result.get(
                        "score",
                        0
                    )

                    if page is not None:
                        st.markdown(
                            f"**{i}. {source} — "
                            f"Page {page}**"
                        )
                    else:
                        st.markdown(
                            f"**{i}. {source}**"
                        )

                    st.caption(
                        f"Similarity score: "
                        f"{score:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()


    # -----------------------------------------------------
    # Store assistant answer
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

import streamlit as st

from src.retriever import FAISSRetriever
from src.groq_client import GroqService
from src.prompts import SYSTEM_PROMPT, build_rag_prompt


# =========================================================
# Configuration
# =========================================================

TOP_K = 5
MAX_CONVERSATIONS = 6


st.set_page_config(
    page_title="NUST Student Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# UI Styling
# =========================================================

st.markdown("""
<style>

/* Main content width */
.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 6rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    min-width: 290px;
    max-width: 290px;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Main title */
.main-title {
    font-size: 2.45rem;
    font-weight: 750;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.main-subtitle {
    font-size: 1.05rem;
    opacity: 0.72;
    margin-bottom: 2rem;
}

/* Source text */
.source-box {
    font-size: 0.85rem;
    opacity: 0.75;
    padding-top: 0.3rem;
}

/* Sidebar title */
.sidebar-title {
    font-size: 1.15rem;
    font-weight: 650;
    margin-bottom: 1rem;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# API Key
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "Groq API key is not configured. "
        "Please add GROQ_API_KEY to Streamlit Secrets."
    )
    st.stop()


# =========================================================
# Load Services
# =========================================================

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


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Recent Questions</div>',
        unsafe_allow_html=True
    )

    # Get only user queries
    recent_questions = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ][-MAX_CONVERSATIONS:]

    if recent_questions:

        # Show newest first
        for question_item in reversed(recent_questions):

            # Short preview
            preview = question_item

            if len(preview) > 65:
                preview = preview[:65] + "..."

            st.caption(f"• {preview}")

    else:
        st.caption("Your recent questions will appear here.")

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="main-title">
        📚 NUST Student Knowledge Base
    </div>

    <div class="main-subtitle">
        Ask questions about university policies, procedures,
        regulations and student information.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Empty State
# =========================================================

if not st.session_state.messages:

    st.info(
        "Ask a question below to search the knowledge base."
    )


# =========================================================
# Display Conversation
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Compact sources
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            sources = message["sources"]

            source_labels = []

            seen = set()

            for source in sources:

                filename = source.get(
                    "source",
                    "Unknown source"
                )

                page = source.get("page")

                if page is not None:
                    label = f"{filename} · p. {page}"
                else:
                    label = filename

                # Avoid duplicate source labels
                if label not in seen:
                    seen.add(label)
                    source_labels.append(label)

            if source_labels:

                st.markdown(
                    '<div class="source-box">'
                    '<b>Sources:</b> '
                    + " &nbsp; • &nbsp; ".join(source_labels)
                    + "</div>",
                    unsafe_allow_html=True
                )


# =========================================================
# User Question
# =========================================================

question = st.chat_input(
    "Ask about NUST policies, procedures or student information..."
)


if question:

    # -----------------------------------------------------
    # Add User Message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # -----------------------------------------------------
    # Keep Only Last 6 Conversations
    #
    # Each conversation = user + assistant
    # Therefore max messages = 12
    # -----------------------------------------------------

    max_messages = MAX_CONVERSATIONS * 2

    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = (
            st.session_state.messages[-max_messages:]
        )


    # -----------------------------------------------------
    # Display Question
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.markdown(question)


    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    try:

        with st.spinner("Searching the knowledge base..."):

            results = retriever.search(
                query=question,
                k=TOP_K
            )

    except Exception as e:

        st.error(f"Knowledge base search failed: {e}")
        st.stop()


    # -----------------------------------------------------
    # RAG Prompt
    # -----------------------------------------------------

    rag_prompt = build_rag_prompt(
        question,
        results
    )


    # -----------------------------------------------------
    # Generate Answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner("Preparing answer..."):

                answer = groq_service.generate_answer(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=rag_prompt
                )

        except Exception as e:

            st.error(f"Unable to generate answer: {e}")
            st.stop()

        st.markdown(answer)


        # -------------------------------------------------
        # Compact Sources
        # -------------------------------------------------

        source_labels = []
        seen = set()

        for result in results:

            filename = result.get(
                "source",
                "Unknown source"
            )

            page = result.get("page")

            if page is not None:
                label = f"{filename} · p. {page}"
            else:
                label = filename

            if label not in seen:
                seen.add(label)
                source_labels.append(label)


        if source_labels:

            st.markdown(
                '<div class="source-box">'
                '<b>Sources:</b> '
                + " &nbsp; • &nbsp; ".join(source_labels)
                + "</div>",
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # Save Assistant Response
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": results
    })


    # Final history limit
    if len(st.session_state.messages) > max_messages:

        st.session_state.messages = (
            st.session_state.messages[-max_messages:]
        )

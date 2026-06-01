import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import get_retriever, DB_LOCATION


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="centered"
)


@st.cache_resource
def load_retriever():
    return get_retriever()


@st.cache_resource
def load_chain():
    model = OllamaLLM(
        model="llama3.2",
        temperature=0
    )

    template = """
You are an HR policy assistant.

Answer the user's question using ONLY the HR policy context below.

Rules:
1. Give a brief and simple answer.
2. Do not show document names, page numbers, or source text.
3. If the answer is not available in the context, say:
   "I don't know based on the HR policy documents."
4. Do not use outside knowledge.
5. Do not guess.

HR Policy Context:
{context}

Question:
{question}

Brief Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model


def format_docs(docs):
    if not docs:
        return ""

    return "\n\n---\n\n".join([doc.page_content for doc in docs])


st.title("📘 HR Policy Assistant")
st.write("Ask questions based only on your company HR policy document.")


if not DB_LOCATION.exists():
    st.error("Vector database not found. First run: python build_db.py")
    st.stop()


retriever = load_retriever()
chain = load_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask something about HR policy...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policy..."):
            docs = retriever.invoke(question)
            context = format_docs(docs)

            if not context:
                answer = "I don't know based on the HR policy documents."
            else:
                answer = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

            st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

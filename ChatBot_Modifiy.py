import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader, UnstructuredMarkdownLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# OpenAPI Key
# openai_api_key = st.secrets["OPENAI_API_KEY"]

icon_path = r"C:\Users\tech\Downloads\Bear_test.jpg"

def main():
    st.set_page_config(
        page_title="Officekeeper ChatBot",
        page_icon=icon_path)

    st.title("_OfficeKeeper :red[QA Chat]_")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx', 'txt', 'md'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "주어진 문서에 대해 질문해주세요"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    result = chain({"question": query})
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result.get('source_documents', [])

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(note):
    keeper_list = []

    for keeper in note:
        file_name = keeper.name
        with open(file_name, "wb") as file:
            file.write(keeper.getvalue())
            logger.info(f"Uploaded {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
        elif file_name.endswith('.txt'):
            loader = TextLoader(file_name)
        elif file_name.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        keeper_list.extend(documents)

    return keeper_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        pattern=r"(#+\s)",
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len,
        keep_separator=True
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()

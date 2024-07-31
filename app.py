import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
from langdetect import detect
from googletrans import Translator


load_dotenv()

# Configuring API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.write("API key not found. Make sure your .env file is set up correctly.")
else:
    genai.configure(api_key=api_key)


translator = Translator()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def load_and_process_csv():
    csv_file_path_en = "easypaisa-dpdf.csv" 
    try:
        df_en = pd.read_csv(csv_file_path_en)
    except FileNotFoundError:
        st.write(f"CSV file not found at path: {csv_file_path_en}")
        return None
    
    raw_text_en = "\n".join(df_en['text'].tolist())
    
    raw_text = raw_text_en + "\n"

    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    You are a helpful Easypaisa assistant. Use the provided context and your extensive knowledge base to answer the question efficiently.
    Avoid unnecessary details and ensure all relevant information is included.
    If the answer is not known, simply state, "The answer is not available in the context."
    Do not provide wrong answers. Always end with, "Do let me know if you have any other query."

    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]


def main():
    st.set_page_config(page_title="EazyGuide", layout="wide")
    
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        st.image('easypaisa.png', use_column_width=True)
    with col3:
        st.write("")
    
    st.header("Chat with Easypaisa Assistant chatbot")

    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

 
    vector_store = load_and_process_csv()
    if vector_store is None:
        return

    
    st.sidebar.title("Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.sidebar.write(f"Q: {chat['question']}")
            st.sidebar.write(f"A: {chat['answer']}")
            st.sidebar.write("")

    
    user_question = st.text_input("Feel free to ask a question")
    if user_question:
        with st.spinner("Fetching answer..."):
            try:
                # Detecting language
                lang = detect(user_question)
                if lang == 'ur':  # If input is in Urdu
                    # Translate the question to English
                    translated_question = translator.translate(user_question, src='ur', dest='en').text
                    # Get the response in English
                    reply = user_input(translated_question, vector_store)
                    # Translate the response back to Urdu
                    translated_reply = translator.translate(reply, src='en', dest='ur').text
                    st.write(f"EazyGuide: {translated_reply}")
                    st.session_state.chat_history.append({"question": user_question, "answer": translated_reply})
                else:
                    reply = user_input(user_question, vector_store)
                    st.write(f"EazyGuide: {reply}")
                    st.session_state.chat_history.append({"question": user_question, "answer": reply})
            except Exception as e:
                st.write(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

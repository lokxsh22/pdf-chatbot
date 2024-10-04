import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from dotenv import load_dotenv
import csv
from datetime import datetime
import langsmith
from colorama import init, Fore, Style


# Load environment variables from .env file
load_dotenv()

# Langcmith Client
langsmith_client = langsmith.Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


def log_message(level: str, message: str):
    """Logs messages with timestamps and colors based on severity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "info": Fore.BLUE
    }
    print(f"{colors.get(level, Fore.WHITE)}{timestamp} - {level.capitalize()}: {Style.RESET_ALL}{message}")


# Function to set the Google API key
def set_google_api_key(key: str):
    """Sets the Google API key in the environment and .env file."""
    with open(".env", "a") as f:
        f.write(f"GOOGLE_API_KEY={key}\n")
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)
    st.success("Google API Key set and saved to .env file!")
    #log_message("success", "Google API Key set.")

# Function to remove the Google API key
def remove_google_api_key():
    """Removes the Google API key from the environment and .env file."""
    os.environ.pop("GOOGLE_API_KEY", None)
    with open(".env", "r") as f:
        lines = f.readlines()
    with open(".env", "w") as f:
        for line in lines:
            if not line.startswith("GOOGLE_API_KEY"):
                f.write(line)
    st.success("Google API Key removed.")
    #log_message("success", "Google API Key removed.")

# Read the Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the API client
if google_api_key:
    genai.configure(api_key=google_api_key)

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    """Extracts and returns the text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() if page else ""
                text += page_text
            except IndexError:
                st.warning(f"Warning: Page {page_num} in the PDF is missing or empty.")
            except Exception as e:
                st.error(f"Error processing page {page_num}: {str(e)}")
                
    if not text.strip():
        st.error("No text found in the uploaded PDF documents.")
    return text


# Split text into chunks
def get_text_chunks(text):
    """Splits the extracted text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    
    if not chunks:
        st.error("No valid text chunks were created from the PDF. Please check the file.")
        return []
    
    return chunks  # Return an empty list if no chunks were created


# Get embeddings for each chunk
def get_vector_store(chunks):
    """Embeds and stores the text chunks in a vector database (FAISS)."""
    if not chunks:
        log_message("warning", "Cannot create a vector store from empty chunks.")
        st.error("Cannot create a vector store from empty chunks.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        log_message("success", "Vector store created and saved successfully.")
        st.success("Vector store created and saved successfully!")

    except Exception as e:
        log_message("error", str(e))
        if "API key not valid" in str(e):
            st.error("Error: Invalid API Key. Please check your API key and try again.")
        else:
            st.error(f"An error occurred: {str(e)}")


# Chain-of-Thought prompting implementation
def get_conversational_chain():
    """Returns a chain that models role-based behavior for question-answering."""
    prompt_template = """
    You are an AI assistant designed to help users analyze data from PDF documents. You should act according to your role:

    - **Assistant's Role**: 
        1. You are knowledgeable and logical.
        2. Break down complex questions into smaller steps.
        3. Provide explanations before providing direct answers.

    - **User's Role**:
        1. Asks for clarification or specific information.
        2. May ask follow-up questions based on your answers.

    The interaction should follow a structured reasoning process. Here is how you should proceed:
    1. Read the user's question.
    2. Use the provided document context.
    3. Think logically and explain your reasoning.
    4. Provide a final answer based on your reasoning.

    Context:
    {context}

    Question:
    {question}

    Steps to Reason:
    1. Identify key information.
    2. Relate the question to the context.
    3. Use structured reasoning to answer.
    4. Articulate the reasoning process step-by-step.
    
    Final Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    context = "\n".join([doc.page_content for doc in docs])

    # Create a sequential chain to handle the query and response generation
    sequential_chain = SequentialChain(chains=[
        chain  # You can add more chains if needed
    ], input_variables=["input_documents", "context", "question"], output_variables=["output_text"])

    # Execute the sequential chain
    response = sequential_chain({
        "input_documents": docs,
        "context": context,
        "question": user_question
    })

    # Initialize colorama for coloring output
    init(autoreset=True)

    # Ensure the output_text contains "Final Answer:"
    output_text = response.get('output_text', '')
    if "Final Answer:" in output_text:
        reasoning_steps = output_text.split("Final Answer:")[0].strip()  # Get the reasoning part
        final_answer = output_text.split("Final Answer:")[1].strip()  # Get the final answer
    else:
        reasoning_steps = output_text.strip()  # If not present, use the whole output as reasoning
        final_answer = "No final answer provided."  # Default message for final answer

    # Print reasoning steps in color
    print(f"{Fore.BLUE}Reasoning Steps:{Style.RESET_ALL}")
    for line in reasoning_steps.split("\n"):
        if "1." in line:
            print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        elif "2." in line:
            print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        elif "3." in line:
            print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
        elif "4." in line:
            print(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")

    # Print the final answer in a different color
    print(f"{Fore.RED}Final Answer: {final_answer}{Style.RESET_ALL}")

    return final_answer


def save_user_info(name, phone, email):
    file_exists = os.path.isfile('user_info.csv')
    with open('user_info.csv', mode='a', newline='') as file:
        fieldnames = ['Name', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})

def main():
    # Set page configuration
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🖐", layout="wide")

    # Sidebar: Get API Key
    with st.sidebar:
        sidebar_bg_color = "#3336ff"  
        st.markdown("<h2 style='color: #4CAF50;'>API Configuration</h2>", unsafe_allow_html=True)

        with st.popover("🔐 Google API"):
            st.write("🔑 **Get Your API Key**: https://ai.google.dev/")
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                api_key = st.text_input("Enter your Google API Key", type="password", key="google_api_key")
                if st.button("Set API Key"):
                    if api_key:
                        set_google_api_key(api_key)
                        st.success("API Key set and saved to .env file!")
                    else:
                        st.error("Please enter a valid API Key.")
            else:
                genai.configure(api_key=api_key)
                st.success("API Key loaded from environment!")

            # Button to remove the API key
            if st.button("Remove API Key"):
                remove_google_api_key()

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<h2 style='color: #4CAF50;'>Upload Documents</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("", accept_multiple_files=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)

    # Main content area for displaying chat messages
    st.image("/Users/mymac/Lucky/Test8/Assets/img2.png", use_column_width='always')
    st.write(" ")
    st.markdown("<h3 style='color: #0c10e9;'>Ask a Question</h3>", unsafe_allow_html=True)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")

        # Check for specific user request to call them
        if "call me" in prompt.lower():
            st.session_state.collecting_info = True

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.markdown(f"**Assistant:** {response}")

    # Collect user information
    if "collecting_info" in st.session_state and st.session_state.collecting_info:
        st.subheader("Please provide your contact details:")
        with st.form(key="contact_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                save_user_info(name, phone, email)
                st.markdown(f"**Assistant:** Thank you, {name}. We will contact you at {phone} or {email}.")
                st.session_state.collecting_info = False

if __name__ == "__main__":
    main()

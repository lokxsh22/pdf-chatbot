# Chat With PDF

Chat With PDF is an interactive AI-powered application that allows users to upload PDF documents, process them, and ask questions about the content. This application utilizes Google’s Generative AI and LangChain to provide intelligent responses based on the content of the uploaded PDFs. It also offers various NLP techniques for enhanced conversational interaction, including Chain-of-Thought (CoT) prompting, role-based interaction, and structured reasoning.

## Features

- **PDF Text Extraction**: Upload multiple PDF documents and extract text content.
- **Text Chunking**: Large text is split into manageable chunks for better analysis.
- **Google Generative AI**: Embedding generation and conversational responses using Google’s API.
- **FAISS Integration**: Vector search capabilities with FAISS for fast, efficient retrieval.
- **Conversational AI**: Provides role-based and Chain-of-Thought (CoT) reasoning in responses.
- **Question Answering**: Users can ask questions based on the document's content.
- **User Information Collection**: A simple form to collect user information (name, phone, and email).
- **API Key Management**: Users can set or remove their Google API key via the UI.

## Techniques Implemented

The following AI prompting techniques and chaining methods are utilized in this project:

### Prompting Techniques
- **Direct Prompting**: The AI assistant provides direct responses to user questions.
- **Few-Shot Prompting**: Context is provided to the model using a few examples before responding.
- **Chain-of-Thought (CoT) Prompting**: The AI assistant explains its reasoning steps before providing the final answer.
- **Role-based Prompting**: The assistant behaves as a structured, logical entity that provides detailed reasoning.
- **Zero-shot Prompting**: The model generates answers without prior examples or instructions.
- **Interactive Prompting**: The AI assistant interacts in real-time with the user, responding to questions and inputs.

### Chaining Techniques
- **Sequential Chaining**: Multiple chains are run one after the other to create a coherent output.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Web-based UI framework for interactive user input and document upload.
- **PyPDF2**: Library to extract text from PDF files.
- **LangChain**: Framework for building NLP applications and managing prompting/chaining logic.
- **FAISS**: A library for efficient similarity search and vector storage.
- **Google Generative AI (Gemini)**: Google's API for conversational and embedding tasks.
- **Colorama**: Provides colored output in the terminal for easier debugging.
- **dotenv**: Manages environment variables like the Google API key and LangSmith configuration.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/lokxsh22/pdf-chatbot.git
    ```

2. Navigate to the project directory:

    ```bash
    cd pdf-chatbot
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the environment variables:

    Create a `.env` file in the root directory and add your Google API key and LangSmith configuration:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=your_langchain_api_key
    LANGCHAIN_PROJECT=your_langchain_project_name
    ```

## Environment Setup

Ensure that you have Python installed (preferably version 3.7 or higher). You can create a virtual environment to isolate your dependencies. Here’s how you can set it up:

1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

3. After activating the virtual environment, proceed with the installation of dependencies as mentioned above.

## Usage

1. **Run the application**:

    ```bash
    streamlit run main.py
    ```

2. **Upload PDFs**: Use the sidebar to upload your PDF documents.
3. **Ask Questions**: Type your questions in the chat interface and get answers based on the content of your PDFs.
4. **Clear History**: You can clear the chat history using the provided button.
5. **API Key Management**: Add or remove your Google API key from the sidebar.

## Key Components

- **`get_pdf_text()`**: Extracts and combines text from uploaded PDF files.
- **`get_text_chunks()`**: Splits long text into chunks for better embedding and analysis.
- **`get_vector_store()`**: Generates embeddings for text chunks and stores them using FAISS.
- **`get_conversational_chain()`**: Implements Chain-of-Thought (CoT) prompting for detailed reasoning.
- **`user_input()`**: Handles user questions and provides context-driven responses.
- **`save_user_info()`**: Saves user-provided details (name, phone, email) into a CSV file.

## Example

Here’s an example of how to interact with the bot:

1. **Upload a PDF document** that contains some information.
2. **Ask a question** about the document, such as "What are the key points on page 2?" or "Summarize the main content."
3. The bot will process the document, extract relevant information, reason step-by-step, and provide a detailed answer.

## Contact

For any questions or suggestions, feel free to reach out:

- **Name**: Lokeshwaran
- **Email**: lokxsh22@gmail.com

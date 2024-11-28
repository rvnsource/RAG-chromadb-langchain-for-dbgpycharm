# ### 1: Dependencies
# Langchain dependencies
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Updated imports from langchain-openai
from langchain.schema import Document  # Importing Document schema from Langchain
from langchain_chroma import Chroma
from dotenv import load_dotenv  # Importing dotenv to get API key from .env file

import os  # Importing os module for operating system functionalities
import shutil  # Importing shutil module for high-level file operations

# ### 2: Read PDF
# Directory to your pdf files:
DATA_PATH = r"data"


def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.

    Returns:
        List of Document objects: Loaded PDF documents represented as Langchain Document objects.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)  # Initialize PDF loader with specified directory
    return document_loader.load()  # Load PDF documents and return them as a list of Document objects

documents = load_documents()

# Add custom metadata to each document
for doc in documents:
    doc.metadata["author"] = "demo ravi author John Doe"  # Add author field
    doc.metadata["category"] = "demo ravi Research Paper"  # Add category field
print(documents)

# ### 3: Split into chunks of text
def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects containing text content to split.

    Returns:
        list[Document]: List of Document objects representing the split text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

chunks = split_text(documents)

# ### 4: Save to a RDB using Chroma
CHROMA_PATH = "chroma"

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    #db.persist() #This is no longer needed as Chroma automatically persists changes.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# ### 5: Create a Chroma Database
def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store


# Load environment variables from a .env file
load_dotenv()
# Generate the data store
generate_data_store()

# ### 6: Query vector database for relevant data
query_text = "Explain how the YOLO method works"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Use same embedding function as before
embedding_function = OpenAIEmbeddings()

# Prepare the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Search the DB.
results = db.similarity_search_with_relevance_scores(query_text, k=3)
if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")
else:
    from langchain.prompts import ChatPromptTemplate

    # Combine context from matching documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize OpenAI chat model
    model = ChatOpenAI()

    # Generate response text based on the prompt
    response_text = model.invoke(prompt)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Format and print response
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


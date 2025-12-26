import os
from dotenv import load_dotenv

from src.helper import load_pdf_file, get_text_chunks, download_hf_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

_ = load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

extracted_data = load_pdf_file('data/')
text_chunks = get_text_chunks(extracted_data)
embeddings = download_hf_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

if not pc.has_index(index_name):
 
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)



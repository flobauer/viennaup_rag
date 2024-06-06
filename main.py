# intro:
# explain RAG + LLM
# give credits to Dr Julija Bainiaksina for the initial code example this demo is built upon
# go into code

# import libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOC_PATH = "./ds.pdf"
CHROMA_PATH = "vdb" 

# ----- Data Indexing Process -----

# load your pdf doc
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

# split the doc into smaller chunks i.e. chunk_size=500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

# ----- Retrieval and Generation Process -----

# User question
query = 'Was ist die Kritik bezüglich Datenschutzes an OpenAI in diesem Dokument?'

# retrieve context - top 5 most relevant (closests) chunks to the query vector 
# (by default Langchain is using cosine distance metric)
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
# print(context_text)

# you can use a prompt template
PROMPT_TEMPLATE = """
    Beantworte die Frage nur auf der Grundlage des folgenden Kontexts:
    {context}
    Beantworte die Frage basierend auf dem obigen Kontext: {question}.
    Gib eine detaillierte Antwort.
    Rechtfertige deine Antworten nicht.
    Verwende keine Informationen, die nicht im Kontext erwähnt sind.
    Sag nicht "laut dem Kontext" oder "im Kontext erwähnt" oder ähnliches.
"""

# load retrieved context and user query in the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# call LLM model to generate the answer based on the given context and query
model = ChatOpenAI()
response_text = model.invoke(prompt)

# output
print(response_text.content)
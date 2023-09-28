from fastapi import FastAPI, UploadFile
from decouple import config
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import random
import numpy as np
from bs4 import BeautifulSoup
import re
import chromadb
from typing import Optional, List
from supabase import create_client, Client
from services import gpt
from ast import literal_eval

tags_metadata = [
    {
        "name": "documents",
        "description": "Operations with documents. Manage and search documents.",
    },
    {
        "name": "passages",
        "description": "Operations with passages. Manage and search passages within documents.",
    },
    # Add more tags as needed
]

app = FastAPI(
    title="Hebbia v0",
    description="An early version of the Hebbia AI app",
    openapi_tags=tags_metadata,
)


## Local Storage
# embedding_mappings = {}
# doc_metadata_mappings = {}

model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")
MAX_TOKEN_SIZE = 100

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="financial_passages")
supabase: Client = create_client(
    "https://zzkngsnmuuxvtkjelvhc.supabase.co",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp6a25nc25tdXV4dnRramVsdmhjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY5NTg1NzAyMywiZXhwIjoyMDExNDMzMDIzfQ.dZ7yD5cdd572ph2_nelH9ucwCq9s8UfDofu0T70Cf1M",
)


# @app.get("/")
# async def root():
#     result = gpt.get_company_ticker_from_input(
#         "How is Adobe and Salesforce doing these days?"
#     )
#     pattern = r"\[([^\]]*)\]"
#     matches = re.findall(pattern, result)
#     result = " ".join(matches)
#     companies = []
#     try:
#         companies = literal_eval(result)
#     except:
#         print("Couldn't find tickers in input")
#     return companies

#     result = gpt.get_ticker_from_filename("Tesla, Inc. _ 8-K (April 03, 2023).html")
#     return result


@app.post("/documents/", tags=["documents"])
async def upload_file(file: UploadFile):
    def get_token_length(sentence):
        return len(sentence.split(" "))

    def encode_chunks(chunks):
        embeddings = model.encode(chunks)
        return embeddings

    def split_sentence_chunks(sentence):
        chunks_arrs = []
        sentence_words = sentence.split(" ")
        for i in range(0, len(sentence_words), MAX_TOKEN_SIZE):
            chunks_arrs.append(sentence_words[i : i + MAX_TOKEN_SIZE])
        chunk_strings = [" ".join(chunk_arr) for chunk_arr in chunks_arrs]
        return chunk_strings

    def get_file_type(filename):
        if filename.endswith(".html"):
            return "html"
        elif filename.endswith(".txt"):
            return "txt"
        else:
            return "unknown"

    def get_source():
        sources = ["sharepoint", "drive", "dropbox", "file_upload", "email"]
        return random.choice(sources)

    # def get_company(filename):
    #     companies = [
    #         "amazon",
    #         "apple",
    #         "ford",
    #         "netflix",
    #         "nikola",
    #         "salesforce",
    #         "tesla",
    #         "walt disney",
    #     ]  # hardcoded, can change to LLM based approach later

    #     filename = filename.lower()
    #     for company in companies:
    #         if company in filename:
    #             return company
    #     return "unknown"

    # Upload document to Supabase first
    source = get_source()
    file_type = get_file_type(file.filename)
    company = ""
    metadata = {
        "filename": file.filename,
        "file_type": file_type,
        "description": "",
        "source": source,
        "company": company,
    }
    data, count = supabase.table("documents").insert({"metadata": metadata}).execute()
    document_id = data[1][0]["id"]

    chunks = []  # array of strings, each string is less than MAX_TOKEN_SIZE
    current_chunk = []  # array of sentences that will be put into a chunk
    current_chunk_length = 0

    try:
        print('Beginning to upload file: "' + file.filename + '"')
        sentences = []
        # Read the content
        contents = await file.read()

        # Separate logic when html
        if file.filename.endswith(".html"):
            soup = BeautifulSoup(contents, "html.parser")
            contents = soup.get_text()
            sentences = re.split(r"\n ", contents)
            title_tag = soup.find("title")
            title_tag_parts = title_tag.string.split("-")
            company = title_tag_parts[0].strip().upper()
        else:
            sentences = contents.decode("utf-8").split(". ")

        for sentence in sentences:
            # Preprocessing the sentence
            sentence = sentence.strip()
            sentence = sentence.replace("\n", "")
            sentence = re.sub(r"\s{2,}", " ", sentence)

            # If the sentence can fit in the current chunk, add it
            if current_chunk_length + get_token_length(sentence) < MAX_TOKEN_SIZE:
                current_chunk.append(sentence)
                current_chunk_length += get_token_length(sentence)
            else:
                # If the sentence cannot fit in the current chunk, add the current chunk to the chunks array
                # and create a new chunk with the sentence and the last sentence as an overlap
                if get_token_length(sentence) < MAX_TOKEN_SIZE:
                    chunks.append(". ".join(current_chunk) + ".")
                    # overlap
                    last_sentence = current_chunk[-1]
                    current_chunk = [last_sentence, sentence]
                    current_chunk_length = get_token_length(
                        last_sentence
                    ) + get_token_length(sentence)
                # If the sentence is too long to fit in a chunk, split the sentence into smaller chunks
                # and add the remainder to the current_chunk
                else:
                    chunks.append(". ".join(current_chunk) + ".")
                    sentence_chunks = split_sentence_chunks(sentence)
                    for i in range(0, len(sentence_chunks) - 1):
                        # don't add period because not an actual sentence
                        chunks.append(sentence_chunks[i])
                    current_chunk = [sentence_chunks[-1]]
                    current_chunk_length = get_token_length(sentence_chunks[-1])

        # Add the last remaining chunk
        chunks.append(". ".join(current_chunk))

        # Encode the chunks and store in local storage
        embeddings = encode_chunks(chunks)
        embeddings = [list(embedding) for embedding in embeddings]

        # TODO - use hash of chunk + filename + date_uploaded to allow uploads of same file
        ids = [str(hash(chunk)) for chunk in chunks]
        metadatas = [
            {
                "doc_id": document_id,
                "filename": file.filename,
                "file_type": file_type,
                "description": "",
                "source": source,
                "company": company,
            }
            for chunk in chunks
        ]

        collection.add(
            documents=chunks,
            # embeddings=embeddings, # Own embeddings not working, using ChromaDB embeddings
            metadatas=metadatas,
            ids=ids,
        )

        supabase.table("documents").update(
            {"status": "indexed", "passages": ids, "metadata": metadatas[0]}
        ).eq("id", document_id).execute()

        ## Local Storage
        # for i in range(0, len(embeddings)):
        #     embedding_key = tuple(embeddings[i])
        #     chunk = chunks[i]
        #     value = {
        #         "doc_id": doc_id,
        #         "passage": chunk,
        #     }
        #     embedding_mappings[embedding_key] = value

        # # Store the metadata
        # doc_metadata_mappings[doc_id] = {
        #     "filename": file.filename,
        #     "num_chunks": len(chunks),
        # }

        return {"chunks": chunks, "metadata": metadatas[0]}

    except Exception as e:
        print('Failed to upload file: "' + file.filename + "with error " + str(e) + '"')
        return {"error": str(e)}


@app.post("/documents/bulk", tags=["documents"])
async def ingest_bulk(files: list[UploadFile]):
    results = []
    for file in files:
        result = await upload_file(file)
        result = {
            "filename": file.filename,
            "chunks": result["chunks"],
            "metadata": result["metadata"],
        }
        results.append(result)
    return results


class SearchQuery(BaseModel):
    query: str
    companies: List[str] = []
    sources: List[str] = []
    file_types: List[str] = []

    class Config:
        extra = "forbid"


@app.post("/passages/search", tags=["passages"])
async def search(query: SearchQuery):
    filters = {}
    if len(query.companies) > 0:
        filters["company"] = {
            "$in": query.companies,
        }
    else:
        result = gpt.get_company_ticker_from_input(query.query)
        print("result: ", result)
        pattern = r"\[([^\]]*)\]"
        matches = re.findall(pattern, result)
        result = " ".join(matches)
        companies_parsed = []

        try:
            companies_parsed = literal_eval(result)
            if isinstance(companies_parsed, str):
                companies_parsed = [companies_parsed]
            filters["company"] = {
                "$in": companies_parsed,
            }
        except:
            print("Couldn't find tickers in input")
    if len(query.sources) > 0:
        filters["source"] = {
            "$in": query.sources,
        }
    if len(query.file_types) > 0:
        filters["file_type"] = {
            "$in": query.file_types,
        }

    result = collection.query(
        query_texts=[query.query],
        n_results=5,
        where=filters,
    )

    passages = [
        {
            "id": result["ids"][0][i],
            "passage": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        }
        for i in range(0, len(result["ids"][0]))
    ]
    return {"filters": filters, "hits": passages}


@app.get("/documents", tags=["documents"])
async def get_documents():
    data, count = supabase.table("documents").select("*").execute()
    return {"documents": data[1]}


@app.get("/documents/{document_id}", tags=["documents"])
async def get_document(document_id: str):
    data, count = (
        supabase.table("documents").select("*").eq("id", document_id).execute()
    )
    return data[1][0]


@app.patch("/documents/{document_id}", tags=["documents"])
async def update_document(document_id: str, metadata: dict):
    # Update Supabase SQL
    data, count = (
        supabase.table("documents")
        .update({"metadata": metadata})
        .eq("id", document_id)
        .execute()
    )

    document = data[1][0]
    # Update ChromaDB
    collection.update(
        ids=document["passages"],
        metadatas=[metadata] * len(document["passages"]),
    )

    return data[1][0]


# @app.post("/db/clear")
# async def clear_db():
#     chroma_client.reset()

## Local Storage approach
# results = []
# MAX_RESULTS_SIZE = 5
# query_embedding = model.encode([query.query])[0]
# for embedding_key in embedding_mappings.keys():
#     ## convert embedding_key tuple into np array
#     embedding = np.array(embedding_key)
#     ## calculate cosine similarity
#     cosine_similarity = np.dot(query_embedding, embedding) / (
#         np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
#     )

#     embedding_value = embedding_mappings[embedding_key]
#     results.append(
#         {
#             "confidence": round(float(cosine_similarity), 5),
#             "doc_id": embedding_value["doc_id"],
#             "passage": embedding_value["passage"],
#             "metadata": doc_metadata_mappings[embedding_value["doc_id"]],
#         }
#     )

# ## sort by cosine similarity in descending order
# results.sort(key=lambda x: x["confidence"], reverse=True)

# ## return the top 5 results
# results = results[:MAX_RESULTS_SIZE]
# return results

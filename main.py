from fastapi import FastAPI, UploadFile
from decouple import config
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import random
import numpy as np
from bs4 import BeautifulSoup
import re

app = FastAPI(title="Hebbia v0", description="An early version of the Hebbia AI app")

embedding_mappings = {}
doc_metadata_mappings = {}
model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.post("/upload")
async def upload_file(file: UploadFile):
    def get_word_length(sentence):
        return len(sentence.split(" "))

    def encode_chunks(chunks):
        embeddings = model.encode(chunks)
        return embeddings

    # Get random doc_id
    doc_id = random.randint(0, 1000000)

    MAX_TOKEN_SIZE = 100

    chunks = []
    current_chunk = []
    current_chunk_length = 0

    try:
        sentences = []
        # Read the content
        contents = await file.read()

        # Separate logic if html
        if file.filename.endswith(".html"):
            soup = BeautifulSoup(contents, "html.parser")
            contents = soup.get_text()
            contents = re.sub(r"\s{2,}", " ", contents)
            contents = re.sub(r"\n+", "\n", contents)
            sentences = contents.split(". ")
        else:
            sentences = contents.decode("utf-8").split(". ")

        # Iterate through sentences and create overlapping chunks
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = sentence.replace("\n", "")
            if current_chunk_length + get_word_length(sentence) < MAX_TOKEN_SIZE:
                current_chunk.append(sentence)
                current_chunk_length += get_word_length(sentence)
            else:
                chunks.append(". ".join(current_chunk) + ".")
                # overlap
                last_sentence = current_chunk[-1]
                current_chunk = [last_sentence, sentence]
                current_chunk_length = get_word_length(last_sentence) + get_word_length(
                    sentence
                )

        # Add the last chunk
        chunks.append(". ".join(current_chunk))

        # Encode the chunks and store in local storage
        embeddings = encode_chunks(chunks)
        for i in range(0, len(embeddings)):
            embedding_key = tuple(embeddings[i])
            chunk = chunks[i]
            value = {
                "doc_id": doc_id,
                "passage": chunk,
            }
            embedding_mappings[embedding_key] = value

        # Store the metadata
        doc_metadata_mappings[doc_id] = {
            "filename": file.filename,
            "num_chunks": len(chunks),
        }

        return {
            "chunks": chunks,
            "metadata": {
                "filename": file.filename,
            },
        }

    except Exception as e:
        return {"error": str(e)}


class SearchQuery(BaseModel):
    query: str


@app.get("/mappings")
async def mappings():
    print(embedding_mappings)


@app.post("/ingest/bulk")
async def ingest_bulk(files: list[UploadFile]):
    pass


@app.post("/search")
async def search(query: SearchQuery):
    results = []
    MAX_RESULTS_SIZE = 5
    query_embedding = model.encode([query.query])[0]
    for embedding_key in embedding_mappings.keys():
        ## convert embedding_key tuple into np array
        embedding = np.array(embedding_key)
        ## calculate cosine similarity
        cosine_similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )

        embedding_value = embedding_mappings[embedding_key]
        results.append(
            {
                "confidence": round(float(cosine_similarity), 5),
                "doc_id": embedding_value["doc_id"],
                "passage": embedding_value["passage"],
                "metadata": doc_metadata_mappings[embedding_value["doc_id"]],
            }
        )

    ## sort by cosine similarity in descending order
    results.sort(key=lambda x: x["confidence"], reverse=True)

    ## return the top 5 results
    results = results[:MAX_RESULTS_SIZE]
    return results

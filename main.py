from fastapi import FastAPI, UploadFile
from decouple import config
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import random

app = FastAPI(title="Hebbia v0", description="An early version of the Hebbia AI app")

embedding_mappings = {}
doc_metadata_mappings = {}


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.post("/upload")
async def upload_file(file: UploadFile):
    def get_word_length(sentence):
        return len(sentence.split(" "))

    def encode_chunks(chunks):
        model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")
        embeddings = model.encode(chunks)
        return embeddings

    # Get random doc_id
    doc_id = random.randint(0, 1000000)

    MAX_TOKEN_SIZE = 100

    chunks = []
    current_chunk = []
    current_chunk_length = 0

    try:
        # Read the content
        contents = await file.read()
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


@app.get("/metadata")
async def metadata():
    return doc_metadata_mappings


@app.post("/search")
async def search(query: SearchQuery):
    return query

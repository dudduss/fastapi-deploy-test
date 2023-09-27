from fastapi import FastAPI, UploadFile
from decouple import config
from pydantic import BaseModel

app = FastAPI(title="Hebbia v0", description="An early version of the Hebbia AI app")


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.post("/upload/")
async def upload_file(file: UploadFile):
    def get_word_length(sentence):
        return len(sentence.split(" "))

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


@app.post("/search")
async def search(query: SearchQuery):
    return query

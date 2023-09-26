from fastapi import FastAPI
from algoliasearch.search_client import SearchClient
from decouple import config
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


class SearchQuery(BaseModel):
    query: str


@app.post("/search")
async def search(query: SearchQuery):
    APPLICATION_ID = config("APPLICATION_ID")
    SEARCH_API_KEY = config("SEARCH_API_KEY")
    INDEX = config("INDEX")
    algoliaClient = SearchClient.create(APPLICATION_ID, SEARCH_API_KEY)
    index = algoliaClient.init_index(INDEX)
    objects = index.search(query.query)
    return objects

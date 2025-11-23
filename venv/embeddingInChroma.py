# This lab steps beyond traditional keyword-based methods through the use of vector databases and semantic search.

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
# from gensim.models.keyedvectors import KeyedVectors
#
# initialise model
model_path = "sentence-transformers/all-MiniLM-L6-v2"
encoder = SentenceTransformer(model_path)

# dataset
albums = [
    {"name": "Abbey Road",
     "description": "The last album recorded by the Beatles, featuring the famous medley on Side Two.",
     "artist": "The Beatles", "year": 1969},
    {"name": "Kind of Blue",
     "description": "A landmark jazz album by Miles Davis, known for its modal jazz style and improvisation.",
     "artist": "Miles Davis", "year": 1959},
    {"name": "Dark Side of the Moon",
     "description": "A progressive rock album by Pink Floyd, exploring themes of conflict, greed, time, and mental illness.",
     "artist": "Pink Floyd", "year": 1973},
    {"name": "Thriller",
     "description": "Michael Jackson's sixth studio album, breaking numerous records and featuring a fusion of pop, post-disco, funk, and rock.",
     "artist": "Michael Jackson", "year": 1982},
    {"name": "Back to Black",
     "description": "A soul and jazz album by Amy Winehouse, reflecting on her struggles with love and substance abuse.",
     "artist": "Amy Winehouse", "year": 2006},
    {"name": "Rumours",
     "description": "A classic rock album by Fleetwood Mac, noted for its harmonious vocals and introspective lyrics.",
     "artist": "Fleetwood Mac", "year": 1977},
    {"name": "A Love Supreme",
     "description": "A spiritual and expressive jazz suite by John Coltrane, considered one of the greatest jazz albums of all time.",
     "artist": "John Coltrane", "year": 1965},
    {"name": "Sgt. Pepper's Lonely Hearts Club Band",
     "description": "A groundbreaking album known for its innovative production and cover art.",
     "artist": "The Beatles", "year": 1967},
    {"name": "The Four Seasons",
     "description": "A set of four violin concertos by Antonio Vivaldi, each giving musical expression to a season of the year.",
     "artist": "Antonio Vivaldi", "year": 1725},
    {"name": "The Rise and Fall of Ziggy Stardust and the Spiders from Mars",
     "description": "A concept album by David Bowie, telling the story of a fictional rock star named Ziggy Stardust.",
     "artist": "David Bowie", "year": 1972},
    {"name": "Legend",
     "description": "A compilation album by Bob Marley and the Wailers, which became the best-selling reggae album of all time.",
     "artist": "Bob Marley", "year": 1984},
    {"name": "Pet Sounds",
     "description": "An influential album by The Beach Boys, known for its complex harmonies, arrangements, and pioneering recording techniques.",
     "artist": "The Beach Boys", "year": 1966},
    {"name": "Blue",
     "description": "A folk album by Joni Mitchell, featuring raw, emotional songs about love and loss.",
     "artist": "Joni Mitchell", "year": 1971},
]

# Vectorisation
embeddings = [encoder.encode(album['description']) for album in albums]

# Initialise Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma")


# create a collection in chromadb to store embeddings
collection_name = "music_albums"
collection = chroma_client.get_or_create_collection(name=collection_name)

# try:
#     chroma_client.delete_collection(name=collection_name)
# except ValueError as e:
#     pass
#
# collection = chroma_client.create_collection(name=collection_name)

for album, embedding in zip(albums, embeddings):
    try:
        collection.add(
            embeddings = [embedding],
            documents = [album['description']],
            metadatas=[{'name': album['name'], 'artist': album['artist'], 'year': album['year']}],
            ids = [album['name'].replace(" ", "_").lower()]
        )
        print(f"Successfully added {album['name']}")

    except Exception as e:
        print(f"Failed to add {album['name']} :  {e}")

# define a query and convert it to an embedding

query_text = "albums that has fusion"
query_embedding = encoder.encode(query_text).tolist()


# perform the query against the collection

try:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
    )
    print("Results:")
    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"\nResult {i+1} :")
        print(f"Album Name: {metadata['name']}")
        print(f"Artist: {metadata['artist']}")
        print(f"Year: {metadata['year']}")
        print(f"Description: {results['documents'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
except Exception as e:
    print(f"Error performing query with embeddings: {e}")




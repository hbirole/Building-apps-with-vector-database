# Semantic search using Pinecone vector DB Key points:
# - Import necessary libraries and modules.
# - Load the SentenceTransformer model for encoding text.
# - Check for CUDA availability and set the device accordingly.
# - Initialize Pinecone with the API key.
# - Create a serverless index in Pinecone with specified parameters.
# - Upload data in batches to the Pinecone index.   
# - Define a function to run queries against the Pinecone index and print results.
# - Execute a sample query to demonstrate functionality.
# - Have Fun! Reach out if you have any questions or need further assistance: https://himanibirole.com/

# Import necessary libraries and modules 
print("Importing modules...")
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
import os
import time
import torch
from tqdm.auto import tqdm
print("All modules imported.")

##-------------LOAD Dataset---------------------------------------------------------------------------

from datasets import load_dataset

print("Loading Quora dataset...")
dataset = load_dataset('quora', split='train[240000:290000]')
print("Loaded successfully!")

print("First 3 rows:")
print(dataset[:3])

questions = []

for record in dataset:
    questions.extend(record['questions']['text'])  # access the inner 'text' list

questions = list(set(questions))  # deduplicate

print('\n'.join(questions[:10]))
print('-' * 50)
print(f'Number of questions: {len(questions)}')

# #----------------------------------------------------------------------------------------------------
print("Checking for CUDA...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device != 'cuda':
    print('Sorry no cuda.')

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Model loaded.")

# ----------------------------------------------------------------------------------------------------

query = 'which city is the most populated in the world?'
print("Encoding query...")
xq = model.encode(query)
print("Encoded vector shape:", xq.shape)

#------------ initialize Pinecone and look for a serverless index ----------------------------

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")  # Replace with your Pinecone API key and environment
print("Pinecone initialized.")

index_name = "my-rag-index" # specify your index name
if index_name in pc.list_indexes().names():
    #Instantiate the index client
    index = pc.Index(index_name)
    print(f"✅ Connected to index: {index_name}")
    
    pc.delete_index(index_name)
    while index_name in pc.list_indexes().names():
        print("⏳ Waiting for index to be deleted...") 
        time.sleep(2)    
    print(f"✅ Index '{index_name}' deleted. Creating a new index.")
else:
    print(f"❌ Index '{index_name}' does not exist.Creating new index")
    
#-----Create a new Index-----------------------------------------------------------------------
    
    pc.create_index(
    name="my-rag-index", # specify your index name
    dimension=1536,  # specify the dimension of your embeddings
    metric="cosine",  # can be "cosine", "dotproduct", or "euclidean"
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # specify the cloud and region for your serverless index
    ),
    vector_type="dense"  # or "sparse"
)
# print("⏳ Waiting for index to be created...")
# while index_name not in pc.list_indexes().names():
#     time.sleep(2)
print("✅ my-rag-index created!")
index = pc.Index(index_name)  # Instantiate the index client")
index
print(f"✅ Index '{index_name}' connected successfully.")

#-----------#Upload data of batch size 200-------------------------------------------------------

batch_size=200
vector_limit=10000

questions = questions[:vector_limit]
print(f"Number of questions to upload: {len(questions)}")

import json

for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    xc = model.encode(questions[i:i_end])
    # create records list for upsert
    records = zip(ids, xc, metadatas)
    index.upsert(vectors=records)
    stats = index.describe_index_stats()
    print("Total vectors:", stats.total_vector_count)
    
index.describe_index_stats()

index = pc.Index("my-serverless-index")

#--------Lets Run queries------------------------------------------------------------------

print("Index ready. Running queries...")
def run_query(query):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
    
print("Running query...")
run_query(query)
print("Query completed.")

#-----------------------------------------------------------------------------------------------

run_query('which city has the best weather in the world?')
query = 'how do i make a tres-leches cake?'
run_query(query)

query = 'Should I spend in ethereum?'
run_query(query)
print("Query executed successfully.")

# ------------ping openai api key to check if it is valid--(not required)-------------------------

# from openai import OpenAI
# client = OpenAI(
#   api_key="Your-OpenAI-API-Key-Here"  # replace with your OpenAI API key
# )
# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   store=True,
#   messages=[
#     {"role": "user", "content": "write a haiku about ai"}
#   ]
# )
#print(completion.choices[0].message)
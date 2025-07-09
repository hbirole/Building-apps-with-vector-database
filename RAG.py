# Key Points:
# 1. Initialize Pinecone and create an index if it doesn't exist.
# 2. Load a dataset (e.g., from a CSV file). https://www.multimodal.dev/post/how-to-chunk-documents-for-rag
# 3. Prepare the data for embedding and upsert it into Pinecone.  
# 4. Connect to OpenAI's API for generating embeddings and completions.
# 5. Query the index with a user query and retrieve relevant documents.
# 6. Build a prompt using the retrieved documents and the user query.
# 7. Generate a response using OpenAI's API based on the prompt.
# 8. Have Fun! Reach out if you have any questions or need further assistance: https://himanibirole.com/
# Import necessary libraries

import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

import ast
import os
import time
import pandas as pd

# #  Initialize Pinecone, look for index ------------------------------------

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


# # # Download and unzip the dataset----------------------------------------------

import requests
import os
import pandas as pd
import zipfile

zip_path = "lesson2-wiki.csv.zip"
output_folder = "."  # or specify another directory

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

print("✅ '{zip_path}': Unzipped successfully!")

#-----------Load dataset-----------------------------------------------------------

print(os.listdir(output_folder)) 
max_articles_num = 500

df = pd.read_csv('wiki.csv', nrows=max_articles_num)
print(df.head())  
print(f"✅ Loaded {len(df)} articles from the dataset.")

#----Prepare the Embeddings and Upsert to Pinecone--------------------------------------

import sys
tqdm.pandas()
sys.stdout.flush()

from tqdm import tqdm

prepped = []

for i, row in tqdm(df.iterrows(), total=df.shape[0], dynamic_ncols=True):

    meta = ast.literal_eval(row['metadata'])
    prepped.append({'id':row['id'], 
                    'values':ast.literal_eval(row['values']), 
                    'metadata':meta})
    if len(prepped) >= 250:
        index.upsert(prepped)
        prepped = []
        
print(index.describe_index_stats())

print(df.shape)
print(df.head())

#----------------Connect to open AI-----------------------------------------------------------
client = OpenAI(
api_key="YOUR_OPENAI_API_KEY",  # Replace with your OpenAI API key
)
completion = client.chat.completions.create(
model="gpt-4o-mini",
store=True,
messages=[
    {"role": "user", "content": "write a haiku about ai"}
]
)
print(completion.choices[0].message)

#---------------------------------------------------------------------------------------------------

def get_embeddings(articles, model="text-embedding-ada-002"):
    return client.embeddings.create(input = articles, model=model)

#----Example query to test the RAG system-------------------------------------------------------------

query = "what is chichen itza?"
embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=5, include_metadata=True)
text = [r['metadata']['text'] for r in res['matches']]
print('\n'.join(text))

#--------Build the Prompt---------------------------------------------------------------------------

query = "write an article titled: what is the chichen itza?"
embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

contexts = [
    x['metadata']['text'] for x in res['matches']
]
prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
)

prompt_end = (
    f"\n\nQuestion: {query}\nAnswer:"
)

prompt = (
    prompt_start + "\n\n---\n\n".join(contexts) + 
    prompt_end
)

print(prompt)

#--------Generate the summary using OpenAI's API--------------------------------------

res = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0.2,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
print('-' * 80)
print(res.choices[0].text)

#----------End-----------------------------------------------------------------------------------
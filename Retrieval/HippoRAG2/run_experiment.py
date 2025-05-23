
# # Make sure to set your OPENAI_API_KEY in a .env file or export it in your shell before running this script.

from hipporag import HippoRAG

#Prepare dataset
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County."
]

save_dir = 'MetatagIndexing/HippoRAG2/HippoRAG_outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'facebook/opt-125m' # Any OpenAI model name
embedding_model_name = 'facebook/contriever'# Embedding model name (NV-Embed, GritLM or Contriever for now)

# Set HF_HOME to a local directory
import os
os.environ['HF_HOME'] = "./HF"
os.environ['OPENAI_API_KEY'] = "Empty"
#Startup a HippoRAG instance
print("Starting HippoRAG instance...")
hipporag = HippoRAG(
    save_dir=save_dir,
    llm_model_name=llm_model_name,
    embedding_model_name=embedding_model_name,
)

# Run indexing
hipporag.index(docs=docs)

# Run retrieval
retrieval_results = hipporag.retrieve(queries=["Who is the president of the United States?"], num_to_retrieve=2)

print(retrieval_results)
import gpt4all
import numpy as np

# Define the input text array
file_contents = [
    "file1.txt: My best friend is Alice. She loves to hike and read books. We often go on adventures together.",
    "file2.txt: John is a great colleague, always helping with project deadlines. He's very reliable.",
    "file3.txt: Emily is my cousin. We share a lot of childhood memories, especially from family gatherings.",
    "file4.txt: Alice recently adopted a new puppy. She's so excited about it.",
    "file5.txt: David is my neighbor. He's good at gardening, and he often shares his vegetables with us.",
    "file6.txt: Alice and I are planning a trip to the mountains next summer. It's going to be epic.",
    "file7.txt: Sarah is my old school teacher. She taught me a lot about history.",
    "file8.txt: My best friend, Alice, told me about a new coffee shop downtown. We should check it out.",
    "file9.txt: Robert is a distant relative. We only meet during major family events.",
    "file10.txt: Alice's favorite color is blue. She also enjoys painting in her free time.",
    "file10.txt: We had a nice time with Geoff Evans in the october 2024",
    "file10.txt: I visited my Mom in september 2024",
]


def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


print("Initializing Embed4All model...")
# Initialize the Embed4All model.
# This will download the model if it's not already present locally.
embedder = gpt4all.Embed4All()
print("Embed4All model initialized successfully.")

# Extract just the text content from the file_contents array.
# We remove the "fileX.txt: " prefix.
texts_to_embed = [content.split(": ", 1)[1] for content in file_contents]
original_file_names = [content.split(":", 1)[0] for content in file_contents]

print(f"\nGenerating embeddings for {len(texts_to_embed)} texts...")
# Generate embeddings for the list of texts
# The embed method returns a list of embeddings, where each embedding is a list of floats.
embeddings = embedder.embed(texts_to_embed)
print("Embeddings generated successfully.")

# Convert embeddings to numpy array for efficient calculations
embeddings_np = np.array(embeddings)

print(f"\nTotal embeddings generated: {len(embeddings_np)}")
if embeddings_np.size > 0:
    print(f"Dimension of each embedding: {embeddings_np.shape[1]}")
    print("\nFirst 3 embeddings (showing first 10 dimensions for brevity):")
    for i, emb in enumerate(embeddings_np[:3]):
        print(f"Embedding {i + 1}: {emb[:10]}...")  # Print first 10 dimensions
else:
    print("No embeddings were generated.")

print("\n--- Starting GPT4All Chat Session with Embedding Retrieval ---")
print("Initializing Gpt4All model for chat...")

# Initialize the Gpt4All model for chat.
# This will download a default model (e.g., 'ggml-gpt4all-j-v1.3-groovy.bin')
# if it's not already present locally. You can specify a model name if needed.
# Example: chat_model = gpt4all.Gpt4All("orca-mini-3b-gguf2.q4_0.gguf")
chat_model = gpt4all.GPT4All("qwen2.5-coder-7b-instruct-q4_0.gguf")
print("Gpt4All chat model initialized successfully. Type 'exit' to quit.")

# Start an interactive chat session
with chat_model.chat_session():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Exiting chat session.")
            break

        # 1. Embed the user's query
        print("Embedding user query...")
        query_embedding = embedder.embed([user_input])[0]
        query_embedding_np = np.array(query_embedding)

        # 2. Find the most similar text(s) from the pre-embedded file_contents
        similarities = []
        for i, text_embedding in enumerate(embeddings_np):
            sim = cosine_similarity(query_embedding_np, text_embedding)
            similarities.append((sim, texts_to_embed[i], original_file_names[i]))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Get the top 2 most relevant documents
        top_relevant_docs = similarities[:20]
        print(f"Top relevant documents for your query:")
        context_text = ""
        for sim, text, file_name in top_relevant_docs:
            print(f"  - {file_name} (Similarity: {sim:.4f}): {text}")
            context_text += f"{text}\n"

        # 3. Construct an augmented prompt
        # This is a simple RAG-like approach where relevant context is prepended
        # to the user's query before sending it to the LLM.
        augmented_prompt = f"You are an assistant responsible to manage my memories. Chat with me like a normal dialog and answer with a memories only if there is a question from me. Here is my memories you can use:\n\n{context_text}\nMessage: {user_input}\nAnswer:"

        print("\nGenerating response from model...")
        # 4. Generate a response from the model using the augmented prompt
        # Lower temperature for more focused responses given the context
        response = chat_model.generate(prompt=augmented_prompt, temp=0.2, max_tokens=200)
        print(f"Model: {response}")

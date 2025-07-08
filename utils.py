def chunk_text(text, n, overlap):
    """
    Chunks the given text into segments of n characters with overlap.

    Args:
    text (str): The text to be chunked.
    n (int): The number of characters in each chunk.
    overlap (int): The number of overlapping characters between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks
    
    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks

def create_embeddings(text, model="text-embedding-3-small", client=None):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings. Default is "text-embedding-3-small".

    Returns:
    dict: The response from the OpenAI API containing the embeddings.
    """
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # Return the response containing the embeddings

def _cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    Performs semantic search on the text chunks using the given query and embeddings.

    Args:
    query (str): The query for the semantic search.
    text_chunks (List[str]): A list of text chunks to search through.
    embeddings (List[dict]): A list of embeddings for the text chunks.
    k (int): The number of top relevant text chunks to return. Default is 5.

    Returns:
    List[str]: A list of the top k most relevant text chunks based on the query.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # Initialize a list to store similarity scores

    # Calculate similarity scores between the query embedding and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = _cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the indices of the top k most similar text chunks
    top_indices = [index for index, _ in similarity_scores[:k]]
    # Return the top k most relevant text chunks
    return [text_chunks[index] for index in top_indices]


def generate_response(system_prompt, user_message, model="gpt-4o-mini-2024-07-18"):
    """
    Generates a response from the AI model based on the system prompt and user message.

    Args:
    system_prompt (str): The system prompt to guide the AI's behavior.
    user_message (str): The user's message or query.
    model (str): The model to be used for generating the response. Default is "meta-llama/Llama-2-7B-chat-hf".

    Returns:
    dict: The response from the AI model.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

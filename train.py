import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from txtai.embeddings import Embeddings

def process_json_dataset(filepath):
    """
    Reads a JSON file, extracts data, and transforms it into a list of dictionaries.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary has "output" and "text_input" keys.
    """
    try:
      with open(filepath, 'r') as f:
          data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    index_data = []
    pair_counter = 0
    all_ids = set()
    for category, chunks in data.items():
        print(f"Processing category: {category}")
        for item in chunks:
            chunk_id_base = item['id']
            answer_chunk = item['text']
            chunk_metadata = item['metadata']

            for i, query in enumerate(item['questions']):
                pair_id = f"{chunk_id_base}_q{i}" # Generate a unique ID for the pair
                pair_counter += 1

                # Data object stores the question, the answer, and metadata
                data_object = {
                    "text": query, # This is what txtai will embed by default if 'text' key exists
                    "answer": answer_chunk,
                    "category": category,
                    "metadata": chunk_metadata
                }

                # Append tuple: (unique_pair_id, data_object_to_index, optional_tags)
                # We use the 'question' field for embedding similarity.
                index_data.append((pair_id, data_object, None)) # Pass metadata as tags

    print(f"Prepared {len(index_data)} items for indexing.")
    # print(index_data)
    return index_data

def indexing(filePath, save=False):

    # Create embeddings in dex with content enabled. The default behavior is to only store indexed vectors.
    embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

    # Map question to text and store content
    embeddings.index(index_data)

    if(save):
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        print(f"Saving QA index to {filePath}...")
        embeddings.save(filePath)
        print("QA Index saved.")

    try:
        # --- !!! ADD THIS CHECK !!! ---
        index_count_after = embeddings.count()
        print(f"[indexing function] Count immediately after .index() call: {index_count_after}")
        if index_count_after == 0 and len(index_data) > 0:
            print("[indexing function] CRITICAL: Index count is 0 but data was provided!")
        # --- END CHECK ---

        print("Indexing completed successfully in memory (according to try block).")

    except Exception as e:
        print(f"Error during embeddings.index(): {e}")
        return None # Return None if indexing failed

    return embeddings

def find_answer(user_query,limit=3):
    embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})
    index_path_qa = "./index_files"
    embeddings.load(index_path_qa) # Load the index from the specified path
    query_sql = f"SELECT text, answer, score FROM txtai WHERE similar('{user_query}') LIMIT {limit}"
    results = embeddings.search(query_sql)
    return results

def print_ans(search_results):
    if search_results:
        for result in search_results:
            print(result)
            print("-" * 10)
    else:
        print("  No similar questions found in the index.")

def index_data(version):
    data_set_version=version
    data_set_file_name = f"dataset_v{data_set_version}.json"
    dataset_filepath = f"./data/{data_set_file_name}"
    process_json_dataset(dataset_filepath)
    index_path_qa = "./index_files"
    embeddings = indexing(index_path_qa,save=True)
    return embeddings

ans = find_answer("Tell me about hologram chatbot")
print_ans(ans)
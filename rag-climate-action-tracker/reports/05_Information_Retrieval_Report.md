# Information_Retrieval_Report.md

## `climate_tracker/scripts/information_retrieval.py`


**Overall Purpose:**

The script `information_retrieval.py` is designed to process textual data related to different countries' climate policies. For each country, it extracts relevant text segments (chunks) from its associated documents/webpage sections. Then, for a predefined set of questions about climate policies, it finds the text chunk that is most semantically similar to each question and presents this chunk as the "answer" or "evidence." The output for each country is a Markdown file containing the questions and their corresponding best-matching text chunks, along with similarity scores and source URLs.

**How it Works:**

1.  **Setup and Initialization (Lines 1-18):**
    *   **Path Configuration:** It sets up the Python path to ensure modules from the parent project (`climate_tracker`) can be imported.
    *   **Environment Variables:** It loads environment variables from a `.env` file (likely containing the database URL and the path to an embedding model).
    *   **Logging:** It initializes a logger for recording the script's progress and any issues.
    *   **Imports:**
        *   `os`, `sys`, `Path`: For interacting with the file system and managing paths.
        *   `dotenv`: For loading environment variables.
        *   `nltk`: The Natural Language Toolkit, specifically used here for sentence tokenization, which is a prerequisite for text chunking.
        *   `sklearn.metrics.pairwise.cosine_similarity`: A function to calculate the cosine similarity between vector embeddings. This is the core mechanism for finding how "similar" a question is to a text chunk.
        *   `numpy`: For numerical operations, especially handling the vector embeddings.
        *   Project-specific modules:
            *   `get_logger`: Custom logging setup.
            *   `get_db_session`, `CountryModel`, `CountryPageSectionModel`: These are for interacting with a database. `CountryModel` likely represents a table (`countries_v2`) storing general country information (like name and `doc_id`). `CountryPageSectionModel` likely represents a table (`country_page_sections_v2`) storing text content from different sections of a country's climate policy page/document, linked to a `CountryModel` via `country_doc_id`.
            *   `BAAIEmbedder`: A custom class used to convert text (questions and chunks) into numerical vector representations (embeddings). This uses a pre-trained model (specified by `BGE_MODEL_PATH` environment variable, defaulting to "BAAI/bge-m3").

2.  **Predefined Questions (Lines 20-25):**
    *   `PREDEFINED_QUESTIONS`: A list of five specific questions that the script will try to find answers for in each country's documents. These questions cover topics like net-zero targets, sectoral emission targets, energy efficiency, net-zero electricity, and carbon pricing.

3.  **`chunk_text` Function (Lines 28-44):**
    *   **Purpose:** To break down a large piece of text into smaller, overlapping chunks of sentences. This is done because embedding models often have input length limitations, and processing smaller chunks can yield more focused similarity matches.
    *   **Process:**
        *   It first checks if the NLTK `punkt` tokenizer (for sentence splitting) is available and downloads it if not.
        *   It tokenizes the input `text` into individual sentences using `nltk.sent_tokenize()`.
        *   It then groups these sentences into `chunks`. The `chunk_size` parameter (default is 3 sentences) determines how many sentences are in each chunk, and `overlap` (default is 1 sentence) determines how many sentences are shared between consecutive chunks. This overlap helps ensure that a piece of relevant information isn't split awkwardly across two non-overlapping chunks.
        *   It returns a list of these text chunks, filtering out any empty ones.

4.  **`retrieve_and_format_answers` Function (Lines 46-190):**
    *   This is the main function that orchestrates the entire information retrieval process.
    *   **Initialization (Lines 47-57):**
        *   Sets up an `output_dir_name` (defaulting to "retrieved\_country\_reports\_v2\_chunked") and creates this directory if it doesn't exist.
        *   Initializes the `BAAIEmbedder` using the model path from the environment variable.
        *   Establishes a database session using the `DATABASE_URL` from the environment variable.
    *   **Fetch Countries (Lines 59-61):**
        *   Queries the database to get all entries from the `CountryModel` (table `countries_v2`).
    *   **Process Each Country (Lines 63-185):**
        *   The script iterates through each `country_data` object fetched from the database.
        *   **Country-Specific Setup (Lines 64-67):**
            *   Extracts `country_doc_id` and `country_name`.
            *   Starts building a Markdown string (`country_markdown_parts`) for the country's report.
        *   **Fetch Country Sections (Lines 69-73):**
            *   Queries the `CountryPageSectionModel` (table `country_page_sections_v2`) to find all text sections associated with the current `country_doc_id`.
        *   **Handle No Sections (Lines 75-80):**
            *   If no sections are found for a country, it logs a warning, adds a note to the Markdown report, and writes a minimal report file for that country, then skips to the next country.
        *   **Chunking All Sections for the Country (Lines 82-94):**
            *   It initializes lists to store all chunks from all sections of the current country (`all_country_chunks`), their corresponding source URLs (`all_country_chunk_source_urls`), and source section titles (`all_country_chunk_source_section_titles`).
            *   It iterates through each `section` fetched for the country.
            *   If a section has no `text_content`, it's skipped.
            *   It calls the `chunk_text` function to split the `section.text_content` into chunks.
            *   For each generated `chunk`, it appends the chunk, its source URL (using `section.section_url` or falling back to `country_data.country_url`), and the `section.section_title` to their respective lists.
        *   **Handle No Chunks (Lines 96-102):**
            *   If, after processing all sections, no valid text chunks were generated (e.g., all sections were empty), it logs a warning, adds a note to the Markdown report, writes a minimal report, and skips to the next country.
        *   **Embed Chunks (Lines 104-131):**
            *   Logs that it's starting to embed the chunks for the current country.
            *   `embedder.encode_batch(all_country_chunks)`: This is where the BAAI embedding model is used to convert the list of text chunks into a list of numerical vector embeddings. The result (`raw_embeddings_for_debug`) is expected to be a NumPy array where each row is the embedding for a chunk.
            *   **Crucial Error Handling and Validation (Lines 109-131):**
                *   It checks if the `raw_embeddings_for_debug` is a non-empty NumPy `ndarray`.
                *   If it is, it assigns it to `country_chunk_embeddings_np`.
                *   It specifically handles the case where there might be only one chunk, which could result in a 1D NumPy array; if so, it reshapes it into a 2D array (1 row, N columns).
                *   If, after these checks, `country_chunk_embeddings_np` is `None` or empty (meaning embedding failed or produced an unexpected result), it logs a detailed error (including the type of object returned by the embedder), adds an error message to the Markdown report, writes a minimal report, and skips to the next country.
                *   A `try-except` block also catches any other exceptions during this embedding process, logs the error, and handles it similarly by writing a minimal report and skipping.
        *   **Process Each Predefined Question (Lines 133-177):**
            *   It iterates through each `question` in the `PREDEFINED_QUESTIONS` list.
            *   Appends the question to the Markdown report.
            *   **Embed Question (Lines 136-141):**
                *   `embedder.encode_batch([question])[0]`: The current question is also embedded into a numerical vector using the same embedder. It's wrapped in a list because `encode_batch` expects a list, and `[0]` extracts the single embedding.
                *   If an error occurs during question embedding, it's logged, an error message is added to the report, and the script continues to the next question.
            *   **Calculate Similarities (Lines 143-149):**
                *   `cosine_similarity(question_embedding.reshape(1, -1), country_chunk_embeddings_np)`: Calculates the cosine similarity between the question's embedding and *all* the chunk embeddings for the current country. The `reshape(1, -1)` ensures the question embedding is a 2D array, as required by `cosine_similarity`.
                *   If `similarities.size` is 0 (e.g., if `country_chunk_embeddings_np` was empty despite earlier checks, though unlikely given the robust error handling), it logs a warning and adds a note to the report.
            *   **Find Best Match (Lines 151-155):**
                *   `np.argmax(similarities)`: Finds the index of the chunk that has the highest similarity score with the question.
                *   It then retrieves the `best_score`, the `best_matching_chunk_text`, its `best_matching_chunk_source_url`, and the `source_section_title_for_log`.
            *   **Format and Append Answer to Report (Lines 159-165):**
                *   Adds the "Answer/Evidence" heading along with the similarity score to the Markdown.
                *   The `best_matching_chunk_text` is formatted: backticks (`)` are escaped (to `\\``) and newlines are replaced with spaces, then it's added as a blockquote (`> ...`).
                *   The `best_matching_chunk_source_url` is added as a clickable link.
        *   **Write Country Report (Lines 168-173):**
            *   After all questions are processed for a country, the complete `country_markdown_parts` list is joined into a single string.
            *   A filename is created (e.g., `usa_report.md`) based on the `country_doc_id`.
            *   The Markdown content is written to this file in the specified `output_path`.
    *   **Completion and Cleanup (Lines 187-190):**
        *   Logs that the entire process is complete and where the reports are saved.
        *   Closes the database session if it was opened.

5.  **Script Execution (Lines 192-193):**
    *   `if __name__ == "__main__":`: This standard Python construct ensures that the `retrieve_and_format_answers()` function is called only when the script is executed directly (not when it's imported as a module into another script).

In essence, this script automates the process of finding the most relevant pieces of information from a collection of country-specific climate documents in response to a fixed set of questions, using semantic similarity based on text embeddings. It's robust in its handling of missing data and potential errors during the embedding process.


---

**Changes for Fallback Mechanism:**

To improve robustness in the information retrieval process, we introduced a chunk-level fallback mechanism when generating embeddings on the fly using BAAI/bge-m3.


Previously, if any individual text chunk failed during embedding (e.g., due to encoding issues, excessive length, or unexpected input), the entire country-level question-answer generation would fail — resulting in skipped Markdown reports.

How It Works
* Each chunk is embedded individually in a try/except block.
* If a chunk fails to embed:
    * It is skipped, and a warning is logged.
    * The process continues embedding the rest of the chunks.
* Only chunks with successfully generated embeddings are retained for:
    * Similarity calculation with predefined questions
    * Markdown report generation


Key Improvements
* Graceful degradation: Even if some chunks fail, the country report can still be generated using the remaining valid ones.
* More complete coverage: Countries that previously had one bad chunk (causing total failure) now produce usable reports.
* No reprocessing needed: Past runs can remain untouched, but future reruns automatically benefit from the fallback.


Example
Before:
A single bad chunk caused gabon_report.md to be skipped entirely.
After:
That chunk is skipped, and the remaining chunks are used to answer all predefined questions — producing a complete report.


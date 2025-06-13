#`climate_tracker/climate_tracker/scripts/generate_embeddings.py`

## Improved Fallback Mechanism 

To enhance robustness and flexibility in our embedding pipeline, we updated the generate_embeddings.py script with a more resilient fallback mechanism and optional filtering support.

## Creating a Fallback Mechanism for Generating the Embeddings
Initially: if one chunk in the batch failed, the whole batch was skipped — causing valid sections to be lost.

To fix this, we now embed each section one by one, using the following pattern:

```python 

embeddings = []
for i, text in enumerate(texts):
    try:
        vec = embedder.encode_batch([text])[0]
        embeddings.append(vec)
    except Exception as e:
        logger.warning(f"⚠ Failed to embed section index {i}: {e}")
        embeddings.append(None)
```

This allows us to:

* Safely embed each section independently
* Skip any broken section
* Log clearly what failed
* And commit all valid embeddings to the database


## Handling Empty or Unusable Sections Gracefully
In cases where none of the sections in a batch contain valid text_content, the script used to loop endlessly — repeatedly trying the same unusable rows.

We added an early exit condition when a specific country is being embedded (via --only-country) and no valid text exists:

```python 
logger.info("No valid texts to embed in the current batch. Checking if more sections are pending...")

# NEW SAFETY CHECK — if nothing is valid for this country, exit
if only_country:
    logger.info(f"No valid sections with text found for country: {only_country}. Exiting early.")
    break

# For general case, continue looping in case next batch has valid sections
if not session.query(CountryPageSectionModel).filter(and_(*base_filter)).first():
    logger.info("No more sections found to process at all.")
    break
continue
```

This ensures that the script does not get stuck in a loop when working with countries like Gabon that may have missing or empty data.

## Key Improvements Recap

### 1. Per-section Embedding Fallback

One bad section no longer blocks the batch.
All valid sections are embedded and saved.


### 2. Command-line Filtering with --only-country

Run embedding for a specific country only:
```python 
python generate_embeddings.py --only-country gabon
```

### 3. Early Exit for Countries with No Valid Text

Prevents infinite looping in edge cases.
Helps debug missing data at the source (e.g., scraping).


### 4. Safe, Idempotent Re-runs

Sections with existing embeddings are skipped.
The script is safe to run multiple times.


## Note on Gabon
While we tested the fallback using --only-country gabon, it did not result in any embeddings.

We found that Gabon’s text_content fields were either empty or not scraped properly. Thus, the fallback mechanism worked as intended — but there was no usable input to embed.
### THIS SCRIPT IS NOW DEPRECATED, IT HAS BEEN REPLACED BY 3_chunk.py and 3.5_embed.py

import sys
from pathlib import Path
import traceback
import logging
import asyncio
import argparse
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import os 
from sqlalchemy import create_engine, text
from datetime import datetime, date
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import uuid
from uuid import UUID

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.extract_document import extract_text_from_pdf
from group4py.src.chunk_embed import DocChunk as DocChunker, Embedding
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.schema import DocChunk
from group4py.src.database import Connection, NDCDocumentORM as Document, DocChunkORM
from group4py.src.constants.settings import FILE_PROCESSING_CONCURRENCY
from group4py.src.hop_rag import HopRAGGraphProcessor
from group4py.src.schema import DatabaseConfig

logger = logging.getLogger(__name__)

def get_file_paths():
    """
    Get the file paths of PDF files in the data/pdfs folder.
    """
    pdf_directory = project_root / "data" / "pdfs"
    if not pdf_directory.exists():
        logger.warning(f"[3_PROCESS] PDF directory not found: {pdf_directory}")
        return []
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    logger.info(f"[3_PROCESS] Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    # Convert Path objects to strings
    return [str(file_path) for file_path in pdf_files]

async def process_file_one(file_path: str, force_reprocess: bool = False):
    """
    Process a file and return a list of chunks and embeddings.
    
    Args:
        file_path: Path to the PDF file
        force_reprocess: If True, reprocess the file even if it has been processed before
    """
    try:
        # Extract file name to be used as document ID
        file_name = Path(file_path).stem
        logger.info(f"[3_PROCESS] Processing file {file_path}")
        
        # Check if document has already been processed
        from group4py.src.schema import DatabaseConfig
        config = DatabaseConfig.from_env()
        connection = Connection(config)
        connection.connect()
        doc_id = file_name
        
        # Use the new check_document_processed function
        is_processed, document = connection.check_document_processed(doc_id)
        
        if is_processed and not force_reprocess:
            logger.info(f"[3_PROCESS] Document {doc_id} has already been processed. Skipping.")
            return None
        
        # Convert string doc_id to UUID using deterministic UUID5 (do this once at the start)
        doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        
        # Handle force reprocessing and document creation in a coordinated way
        if force_reprocess and document:
            logger.info(f"[3_PROCESS] Force reprocessing document {doc_id}")
            session = connection.get_session()
            try:
                # First, delete any logical relationships that reference this document's chunks
                from group4py.src.database import LogicalRelationshipORM
                from sqlalchemy import select
                
                # Get all chunk IDs for this document using explicit select
                chunk_ids_query = select(DocChunkORM.id).filter(DocChunkORM.doc_id == doc_uuid)
                
                # Delete relationships where either source or target chunk belongs to this document
                deleted_relationships = session.query(LogicalRelationshipORM).filter(
                    (LogicalRelationshipORM.source_chunk_id.in_(chunk_ids_query)) |
                    (LogicalRelationshipORM.target_chunk_id.in_(chunk_ids_query))
                ).delete(synchronize_session=False)
                
                if deleted_relationships > 0:
                    logger.info(f"[3_PROCESS] Deleted {deleted_relationships} relationships for document {doc_id}")
                
                # Now delete the chunks
                deleted_chunks = session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_uuid).delete()
                logger.info(f"[3_PROCESS] Removed {deleted_chunks} existing chunks for document {doc_id}")
                
                session.commit()
                
            except Exception as e:
                logger.error(f"[3_PROCESS] Error removing existing chunks and relationships: {e}")
                logger.error(f"[3_PROCESS] Traceback: {traceback.format_exc()}")
                session.rollback()
            finally:
                session.close()
        
        # Ensure document exists in database
        if not document:
            logger.info(f"[3_PROCESS] Document {doc_id} not found in database. Creating record...")
            session = connection.get_session()
            
            try:
                # Extract document metadata from filename
                parts = doc_id.split('_')
                country = parts[0].replace('_', ' ').title() if len(parts) > 0 else ''
                language = parts[1] if len(parts) > 1 else ''
                
                # Try to parse date if available (format: YYYYMMDD)
                submission_date = None
                if len(parts) > 2 and len(parts[2]) >= 8:
                    date_str = parts[2][:8]  # Extract YYYYMMDD portion
                    try:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        submission_date = date(year, month, day)
                    except (ValueError, IndexError):
                        logger.warning(f"[3_PROCESS] Could not parse date from filename: {parts[2]}")
                
                # Create document record
                new_document = Document(
                    doc_id=doc_uuid,
                    country=country,
                    title=f"{country} NDC",
                    url=f"file:///{file_path.replace(os.sep, '/')}",  # Convert local path to file:// URL
                    language=language,
                    submission_date=submission_date,
                    file_path=file_path,
                    file_size=Path(file_path).stat().st_size / (1024 * 1024),  # Size in MB
                    scraped_at=datetime.now()  # Use current timestamp for local files
                )
                
                session.add(new_document)
                session.commit()
                logger.info(f"[3_PROCESS] Created document record for {doc_id}")
                
                # Store document attributes before closing session to avoid DetachedInstanceError
                document_country = new_document.country
                document = new_document
                
            except Exception as e:
                logger.error(f"[3_PROCESS] Error creating document record: {e}")
                session.rollback()
                document_country = 'Unknown'
                # If we can't create the document, we can't proceed
                return None
            finally:
                session.close()
        else:
            # Store document attributes before any potential session changes
            try:
                # Create a new session to safely access the document attributes
                temp_session = connection.get_session()
                # Refresh the document object in the new session
                document = temp_session.merge(document)
                document_country = document.country if document.country else 'Unknown'
                temp_session.close()
            except Exception as e:
                logger.warning(f"[3_PROCESS] Could not access document country: {e}")
                document_country = 'Unknown'
        
        # If force_reprocess is True and document exists, we need to delete existing chunks and relationships
        if force_reprocess and document:
            logger.info(f"[3_PROCESS] Force reprocessing document {doc_id}")
            session = connection.get_session()
            try:
                # First, delete any logical relationships that reference this document's chunks
                from group4py.src.database import LogicalRelationshipORM
                from sqlalchemy import select
                
                # Get all chunk IDs for this document using explicit select
                chunk_ids_query = select(DocChunkORM.id).filter(DocChunkORM.doc_id == doc_uuid)
                
                # Delete relationships where either source or target chunk belongs to this document
                deleted_relationships = session.query(LogicalRelationshipORM).filter(
                    (LogicalRelationshipORM.source_chunk_id.in_(chunk_ids_query)) |
                    (LogicalRelationshipORM.target_chunk_id.in_(chunk_ids_query))
                ).delete(synchronize_session=False)
                
                if deleted_relationships > 0:
                    logger.info(f"[3_PROCESS] Deleted {deleted_relationships} relationships for document {doc_id}")
                
                # Now delete the chunks
                deleted_chunks = session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_uuid).delete()
                logger.info(f"[3_PROCESS] Removed {deleted_chunks} existing chunks for document {doc_id}")
                
                session.commit()
                
            except Exception as e:
                logger.error(f"[3_PROCESS] Error removing existing chunks and relationships: {e}")
                logger.error(f"[3_PROCESS] Traceback: {traceback.format_exc()}")
                session.rollback()
            finally:
                session.close()
        
        # 1. Extract text from PDF with multiple strategies
        extracted_elements = None
        extraction_strategies = ['fast', 'auto', 'ocr_only']  # Use valid strategies for unstructured partition_pdf
        
        for strategy in extraction_strategies:
            try:
                logger.info(f"[3_PROCESS] Attempting text extraction with strategy: {strategy}")
                raw_extracted_elements = extract_text_from_pdf(file_path, strategy=strategy)
                
                # Check if we got meaningful content from the extraction function
                if raw_extracted_elements and len(raw_extracted_elements) > 0:
                    logger.info(f"[3_PROCESS] Strategy {strategy} extracted {len(raw_extracted_elements)} elements")
                    
                    # Handle different element structures - some may be dicts, others may be objects
                    text_content = []
                    for elem in raw_extracted_elements:
                        try:
                            # Try dict-like access first
                            if hasattr(elem, 'get'):
                                text = elem.get('text', '').strip()
                            # Try object attribute access
                            elif hasattr(elem, 'text'):
                                text = getattr(elem, 'text', '').strip()
                            # Try direct string conversion
                            elif isinstance(elem, str):
                                text = elem.strip()
                            # Try converting object to dict
                            elif hasattr(elem, '__dict__'):
                                elem_dict = elem.__dict__
                                text = elem_dict.get('text', '').strip()
                            else:
                                text = str(elem).strip()
                            
                            if text:
                                # Create standardized element structure
                                standardized_elem = {
                                    'text': text,
                                    'metadata': {}
                                }
                                
                                # Try to extract metadata if available
                                try:
                                    if hasattr(elem, 'get'):
                                        standardized_elem['metadata'] = elem.get('metadata', {})
                                    elif hasattr(elem, 'metadata'):
                                        metadata = getattr(elem, 'metadata', {})
                                        if hasattr(metadata, '__dict__'):
                                            standardized_elem['metadata'] = metadata.__dict__
                                        else:
                                            standardized_elem['metadata'] = metadata if isinstance(metadata, dict) else {}
                                    elif hasattr(elem, '__dict__'):
                                        # Extract relevant metadata from object attributes
                                        elem_dict = elem.__dict__
                                        metadata = {}
                                        for key, value in elem_dict.items():
                                            if key != 'text' and not key.startswith('_'):
                                                metadata[key] = value
                                        standardized_elem['metadata'] = metadata
                                except Exception as meta_error:
                                    logger.warning(f"[3_PROCESS] Could not extract metadata from element: {meta_error}")
                                    standardized_elem['metadata'] = {}
                                
                                text_content.append(standardized_elem)
                        except Exception as elem_error:
                            logger.warning(f"[3_PROCESS] Error processing element with strategy {strategy}: {elem_error}")
                            continue
                    
                    if text_content:
                        logger.info(f"[3_PROCESS] Successfully extracted {len(text_content)} text elements using {strategy} strategy")
                        extracted_elements = text_content
                        break  # Success! Stop trying other strategies
                    else:
                        logger.warning(f"[3_PROCESS] Strategy {strategy} returned elements but no actual text content")
                else:
                    logger.warning(f"[3_PROCESS] Strategy {strategy} returned no elements")
                    
            except Exception as e:
                logger.warning(f"[3_PROCESS] Strategy {strategy} failed: {e}")
                # Include traceback for debugging
                logger.debug(f"[3_PROCESS] Strategy {strategy} traceback: {traceback.format_exc()}")
                extracted_elements = None
                continue
        
        # If all extraction strategies failed, try to create a minimal fallback
        if not extracted_elements:
            logger.error(f"[3_PROCESS] All text extraction strategies failed for {file_path}")
            
            # Create a minimal fallback chunk with filename information
            fallback_text = f"Document: {Path(file_path).name}\nCountry: {document_country}\nNote: Text extraction failed - document may be image-based or corrupted."
            
            extracted_elements = [{
                'text': fallback_text,
                'metadata': {
                    'page_number': 1,
                    'paragraph_number': 1,
                    'extraction_method': 'fallback',
                    'extraction_status': 'failed'
                }
            }]
            
            logger.warning(f"[3_PROCESS] Using fallback content for {file_path}")
        
        logger.info(f"[3_PROCESS] Final extracted elements count: {len(extracted_elements)}")
        
        # 2. Create chunks from the extracted text
        doc_chunker = DocChunker()
        chunks = doc_chunker.chunk_document_by_sentences(extracted_elements)
        logger.info(f"[3_PROCESS] Created {len(chunks)} chunks from {file_path}")
        
        # 3. Clean chunks
        try:
            cleaned_chunks = doc_chunker.cleaning_function(chunks)
            logger.info(f"[3_PROCESS] Finished cleaning chunks from {file_path}, got {len(cleaned_chunks)} cleaned chunks")
        except Exception as e:
            logger.error(f"[3_PROCESS] Error during chunk cleaning: {e}")
            # Fall back to using the original chunks if cleaning fails
            logger.info(f"[3_PROCESS] Falling back to original chunks")
            cleaned_chunks = chunks
        
        # Validate cleaned chunks before proceeding
        if not cleaned_chunks or len(cleaned_chunks) == 0:
            logger.error(f"[3_PROCESS] No valid chunks after cleaning for {file_path}")
            
            # Create a minimal chunk if we have none
            if extracted_elements:
                minimal_chunk = {
                    'text': extracted_elements[0].get('text', f"Minimal content for {Path(file_path).name}"),
                    'metadata': {
                        'chunk_index': 0,
                        'extraction_method': 'minimal_fallback'
                    }
                }                
                cleaned_chunks = [minimal_chunk]
                logger.warning(f"[3_PROCESS] Created minimal fallback chunk for {file_path}")
            else:
                return None
                  # 4. Create DocChunk objects before uploading
        db_chunks = []
        
        for i, chunk in enumerate(cleaned_chunks):
            # Skip empty chunks
            if not chunk.get('text', '').strip():
                logger.warning(f"[3_PROCESS] Skipping empty chunk {i} from {file_path}")
                continue
                
            # Create a DocChunkORM instance
            chunk_model = DocChunkORM(
                id=uuid.uuid4(),  # Use UUID object instead of string
                doc_id=doc_uuid,  # Use the UUID we created at the start
                content=chunk.get('text', ''),
                chunk_index=i,
                paragraph=chunk.get('metadata', {}).get('paragraph_number'),
                language=chunk.get('metadata', {}).get('language'),
                chunk_data=chunk.get('metadata', {})
            )
            
            db_chunks.append(chunk_model)
        
        # Verify we have chunks to upload
        if not db_chunks:
            logger.error(f"[3_PROCESS] No valid chunks to upload for {file_path}")
            return None
        
        # 5. Upload chunks to the database, specifically to doc_chunks table
        logger.info(f"[3_PROCESS] Attempting to upload {len(db_chunks)} chunks to database for {file_path}")
        upload_success = connection.upload(db_chunks, table='doc_chunks')
        if not upload_success:
            logger.error(f"[3_PROCESS] Failed to upload chunks to database for {file_path}")
            return None
            
        logger.info(f"[3_PROCESS] Uploaded {len(db_chunks)} chunks to doc_chunks table")

        # 6. Update documents table that the document has been processed
        connection.update_processed(Document, doc_id, chunks=cleaned_chunks, table='documents')
        logger.info(f"[3_PROCESS] Updated processed status in documents table for {doc_id}")
        
        # 7. Create transformer embeddings
        try:
            embedder = Embedding()
            # Explicitly load the models before embedding
            embedder.load_models()
            
            # Check if embedding models were loaded successfully
            if embedder.models_loaded:
                logger.info(f"[3_PROCESS] Embedding models loaded successfully, generating embeddings for {len(cleaned_chunks)} chunks")
                transformer_embedded_chunks = embedder.embed_many(cleaned_chunks)
                logger.info(f"[3_PROCESS] Created transformer embeddings for {len(transformer_embedded_chunks)} chunks")
            else:
                logger.error(f"[3_PROCESS] Embedding models failed to load, falling back to chunks without embeddings")
                transformer_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        except Exception as e:
            logger.error(f"[3_PROCESS] Error creating transformer embeddings: {e}")
            transformer_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        
        # 8. Create word2vec embeddings
        try:
            # Check if we have valid transformer embeddings before attempting word2vec
            if embedder.models_loaded:
                embedder_word2vec = Embedding()
                word2vec_embedded_chunks = embedder_word2vec.word2vec_embedding(cleaned_chunks)
                logger.info(f"[3_PROCESS] Created word2vec embeddings for {len(word2vec_embedded_chunks)} chunks")
            else:
                logger.warning(f"[3_PROCESS] Skipping word2vec embeddings since transformer models failed to load")
                word2vec_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        except Exception as e:
            logger.error(f"[3_PROCESS] Error creating word2vec embeddings: {e}")
            word2vec_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        
        # 9. Process the embeddings and update the database
        session = connection.get_session()
        try:           
            # Retrieve the existing chunks from the database to get their IDs
            db_chunk_query = session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_uuid).all()
            db_chunk_dict = {chunk.chunk_index: chunk for chunk in db_chunk_query}
            
            embeddings_updated = 0
            # Update each chunk with its embeddings
            for i, chunk in enumerate(cleaned_chunks):
                # Skip chunks that don't exist in the database
                if i not in db_chunk_dict:
                    logger.warning(f"[3_PROCESS] Chunk with index {i} for document {doc_id} not found in database")
                    continue
                    
                # Get the transformer embedding for this chunk
                transformer_embedding = None
                if i < len(transformer_embedded_chunks) and isinstance(transformer_embedded_chunks[i], dict):
                    transformer_embedding = transformer_embedded_chunks[i].get('embedding', [])
                    
                # Get the word2vec embedding for this chunk
                word2vec_embedding = None
                if i < len(word2vec_embedded_chunks) and isinstance(word2vec_embedded_chunks[i], dict):
                    word2vec_embedding = word2vec_embedded_chunks[i].get('w2v_embedding', [])
                
                # Get the corresponding database object
                db_chunk = db_chunk_dict[i]
                
                # Track if we updated anything
                chunk_updated = False
                
                # Update embeddings if they exist and are valid
                if transformer_embedding and isinstance(transformer_embedding, list) and len(transformer_embedding) > 0:
                    # Ensure all elements are floats
                    transformer_embedding = [float(val) if not isinstance(val, float) else val for val in transformer_embedding]
                    db_chunk.transformer_embedding = transformer_embedding
                    chunk_updated = True
                    logger.debug(f"[3_PROCESS] Updated transformer embedding for chunk {i} (length: {len(transformer_embedding)})")
                
                if word2vec_embedding and isinstance(word2vec_embedding, list) and len(word2vec_embedding) > 0:
                    # Ensure all elements are floats
                    word2vec_embedding = [float(val) if not isinstance(val, float) else val for val in word2vec_embedding]
                    db_chunk.word2vec_embedding = word2vec_embedding
                    chunk_updated = True
                    logger.debug(f"[3_PROCESS] Updated word2vec embedding for chunk {i} (length: {len(word2vec_embedding)})")
                
                if chunk_updated:
                    embeddings_updated += 1
        
            # Commit the embedding updates
            session.commit()
            logger.info(f"[3_PROCESS] Successfully updated embeddings for {embeddings_updated} chunks in document {doc_id}")
            
            # 10. Run HopRAG processing after embeddings are complete
            try:
                logger.info(f"[3_PROCESS] Running HopRAG processing for {doc_id}")
                config = DatabaseConfig.from_env()
                processor = HopRAGGraphProcessor(config)
                await processor.initialize()
                
                # Generate embeddings if needed
                logger.info(f"[3_PROCESS] Processing embeddings in batch for {doc_id}")
                await processor.process_embeddings_batch(batch_size=100)
                
                # Build relationships for this document's chunks with enhanced logging
                logger.info(f"[3_PROCESS] Building logical relationships for {doc_id}")
                
                # Get the current relationship count before building using Connection class
                rel_count_before = processor.db_connection.count_relationships()
                logger.info(f"[3_PROCESS] Current relationship count: {rel_count_before}")
                
                # Build relationships with more detailed parameters
                # Add filtering to focus only on chunks from the current document
                logger.info(f"[3_PROCESS] Building relationships specifically for document: {doc_id}")
                doc_chunk_count = await processor.get_doc_chunk_count(doc_id)
                logger.info(f"[3_PROCESS] Document {doc_id} has {doc_chunk_count} chunks to process for relationships")
                
                # Ensure we're focused on this document's chunks
                await processor.build_relationships_sparse(
                    max_neighbors=30, 
                    min_confidence=0.55,
                    doc_id=doc_id,  # Add document ID filter to focus on current document
                    force_commit=True  # Ensure relationships are committed to database
                )
                
                # Check how many relationships were added using Connection class
                try:
                    rel_count_after = processor.db_connection.count_relationships()
                    new_rels = rel_count_after - rel_count_before
                    logger.info(f"[3_PROCESS] Added {new_rels} new relationships. Total now: {rel_count_after}")
                    
                    # Get relationship counts by type for this document
                    relationship_counts = processor.db_connection.count_relationships(by_type=True)
                    logger.info(f"[3_PROCESS] Relationship counts by type: {relationship_counts}")
                    
                except Exception as e:
                    logger.warning(f"[3_PROCESS] Could not get updated relationship count: {e}")
                
                # Clean up
                await processor.close()
                logger.info(f"[3_PROCESS] HopRAG processing completed for {doc_id}")
                
                # Final commit to ensure all changes are persisted
                session.commit()
                logger.info(f"[3_PROCESS] Successfully committed all changes for document {doc_id}")
                
            except Exception as e:
                session.rollback()  # Roll back any failed HopRAG changes
                logger.error(f"[3_PROCESS] HopRAG processing failed for {doc_id}: {str(e)}")
                # Don't fail the entire process if HopRAG fails

        except Exception as e:
            session.rollback()
            logger.error(f"[3_PROCESS] Error updating embeddings in database: {e}")
            logger.error(f"[3_PROCESS] Traceback: {traceback.format_exc()}")
        finally:
            session.close()
        
        logger.info(f"[3_PROCESS] File {file_path} processed successfully")
        
        return db_chunks
    
    except Exception as e:
        logger.error(f"[3_PROCESS] Error processing file {file_path}: {e}")
        logger.error(f"[3_PROCESS] Traceback: {traceback.format_exc()}")
        return None

async def process_file_many(file_path): 
    semaphore = asyncio.Semaphore(FILE_PROCESSING_CONCURRENCY)
    async with semaphore:
        return await process_file_one(file_path)

@Logger.log(log_file=project_root / "logs/process.log", log_level="INFO")
async def run_script(force_reprocess: bool = False):
    """
    WARNING: THIS SCRIPT IS NOW DEPRECATED, IT HAS BEEN REPLACED BY 3_chunk.py and 3.5_embed.py.
    
    YOU SHOULD NOT REFERENCE OR RUN THIS SCRIPT DIRECTLY.
    """
    try:
        logger.warning(f"\n\n[3_PROCESS] Running script with force_reprocess={force_reprocess}...")
        file_paths = get_file_paths()
        
        # Show the progress bar for all file processing
        with tqdm(total=len(file_paths), desc="Processing PDF files", unit="file") as pbar:
            # Create a wrapped version of process_file_many that updates the progress bar
            async def process_with_progress(file_path):
                try:
                    # Pass the force_reprocess parameter to process_file_one
                    result = await process_file_one(file_path, force_reprocess=force_reprocess)
                    # Update progress bar after processing each file
                    pbar.update(1)
                    # Add the filename to the progress bar description when it completes
                    pbar.set_postfix_str(f"Last: {Path(file_path).name}")
                    return result
                except Exception as e:
                    # Still update progress even on error
                    pbar.update(1)
                    # Show error in progress bar
                    pbar.set_postfix_str(f"Error with {Path(file_path).name}")
                    logger.error(f"[3_PROCESS] Error in process_with_progress: {e}")
                    return None
                    
            # Process files with progress tracking
            tasks = [process_with_progress(file_path) for file_path in file_paths]
            chunk_results = await asyncio.gather(*tasks)
        
        # Filter out None results and empty lists
        valid_chunks = [chunks for chunks in chunk_results if chunks is not None and len(chunks) > 0]
        
        if not valid_chunks:
            logger.warning(f"[3_PROCESS] No valid chunks to upload to database")
        else:
            total_chunks = sum(len(chunks) for chunks in valid_chunks)
            logger.warning(f"[3_PROCESS] All files processed successfully. {total_chunks} chunks processed")
            logger.warning(f"[3_PROCESS] Script completed successfully. Exiting.")
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 3_process.py: {e}\n\nTraceback: {traceback.format_exc()}\n\n\n\n")
        raise e

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process PDF files and add them to the database")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of all documents, even if already processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the script with force_reprocess parameter
    asyncio.run(run_script(force_reprocess=args.force))
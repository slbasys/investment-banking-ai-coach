# src/data_processing.py
"""
Data processing module for the Investment Banking AI Coach.

This module handles loading, processing, and embedding documents for the RAG system.
Functions include loading PDFs, text splitting, metadata enrichment, and vector storage.
"""

import os
import logging
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables for paths
def get_project_paths():
    """Get project directory paths from environment variables"""
    directory_project = os.environ.get('DIR_PROJECT')
    directory_rag_indexes = os.environ.get('DIR_RAG_INDEXES')
    
    if not directory_project:
        raise ValueError("Environment variable DIR_PROJECT not set")
    
    # Define paths for various content types
    interview_guides_path = os.path.join(directory_project, 'Interview_Guides')
    technical_categories_path = os.path.join(directory_project, 'Technical_Categories')
    trends_path = os.path.join(directory_project, 'Trends')
    vectorstore_path = os.path.join(directory_project, 'vectorstore')
    
    return {
        "project": directory_project,
        "rag_indexes": directory_rag_indexes,
        "interview_guides": interview_guides_path,
        "technical_categories": technical_categories_path,
        "trends": trends_path,
        "vectorstore": vectorstore_path
    }

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load all PDF documents from a directory.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading documents from {directory_path}...")
    try:
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",  # Load all PDFs recursively
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def create_text_splitter(
    chunk_size: int = 2000,
    chunk_overlap: int = 300
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for document chunking.
    
    Args:
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n### ", "\n## ", "\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

def process_documents_with_metadata(
    documents: List[Document], 
    category: str,
    text_splitter: Optional[RecursiveCharacterTextSplitter] = None
) -> List[Document]:
    """
    Process, split and add metadata to documents.
    
    Args:
        documents: List of documents to process
        category: Category to assign to the documents (e.g., "technical", "interview_guide")
        text_splitter: Optional preconfigured text splitter
        
    Returns:
        List of processed and split documents with metadata
    """
    logger.info(f"Processing {len(documents)} {category} documents...")
    
    # Use provided text splitter or create a new one
    if text_splitter is None:
        text_splitter = create_text_splitter()
    
    # Split documents into chunks
    splits = text_splitter.split_documents(documents)
    
    # Add metadata
    for split in splits:
        # Preserve original source file name
        source_file = split.metadata.get("source", "Unknown")
        
        # Add normalized category metadata (lowercase for consistency)
        split.metadata["category"] = category.lower()
        
        # Include original source file name in metadata
        split.metadata["source_file"] = os.path.basename(source_file)
        
        # Detect additional question formats (beyond "question:")
        content_lower = split.page_content.lower()
        if any(keyword in content_lower for keyword in ["question:", "quiz:", "faq:", "example question:"]):
            split.metadata["type"] = "qa_pair"
        else:
            split.metadata["type"] = "information"
    
    logger.info(f"Created {len(splits)} chunks from {category} documents with detailed metadata")
    return splits

def validate_document_loading(
    interview_chunks: List[Document],
    technical_chunks: List[Document],
    trend_chunks: List[Document]
) -> bool:
    """
    Validate that documents were successfully loaded and processed.
    
    Args:
        interview_chunks: Documents from interview guides
        technical_chunks: Documents from technical categories
        trend_chunks: Documents from trends
        
    Returns:
        Boolean indicating successful validation
    """
    total_chunks = len(interview_chunks) + len(technical_chunks) + len(trend_chunks)
    
    if total_chunks == 0:
        logger.warning("WARNING: No documents were processed. Check your directory paths.")
        return False
    else:
        logger.info(f"Successfully processed {total_chunks} document chunks.")
        logger.info(f"- {len(interview_chunks)} from interview guides")
        logger.info(f"- {len(technical_chunks)} from technical categories")
        logger.info(f"- {len(trend_chunks)} from trends")
        
        # Check for category distribution
        category_counts = {}
        for chunk in interview_chunks + technical_chunks + trend_chunks:
            cat = chunk.metadata.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info("Category distribution:")
        for cat, count in category_counts.items():
            logger.info(f"- {cat}: {count} chunks")
        
        return True

def create_or_load_vectorstore(
    embedding_function,
    documents: Optional[List[Document]] = None,
    persist_directory: Optional[str] = None,
    force_recreate: bool = False
) -> Chroma:
    """
    Create a new vector store or load an existing one.
    
    Args:
        embedding_function: Function to use for embeddings
        documents: Optional list of documents to embed
        persist_directory: Directory to store the vector database
        force_recreate: Whether to force recreation of the database
        
    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        paths = get_project_paths()
        persist_directory = paths["vectorstore"]
    
    # Check if vector store already exists
    if os.path.exists(persist_directory) and not force_recreate:
        logger.info("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        logger.info(f"Loaded vector store with {vector_store._collection.count()} documents")
        return vector_store
    
    # Create new vector store if it doesn't exist or recreation is forced
    logger.info("Creating new vector store...")
    if documents:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vector_store.persist()
        logger.info(f"Created vector store with {len(documents)} documents")
        return vector_store
    else:
        logger.warning("No documents provided for creating vector store")
        return None

def preview_chunks(chunks: List[Document], n: int = 3) -> None:
    """
    Preview a few chunks to check content quality.
    
    Args:
        chunks: List of document chunks to preview
        n: Number of chunks to display
    """
    logger.info("=== Document Chunks Preview ===")
    for i, chunk in enumerate(chunks[:n]):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        logger.info(f"Category: {chunk.metadata.get('category', 'Unknown')}")
        logger.info(f"Type: {chunk.metadata.get('type', 'Unknown')}")
        logger.info(f"Content: {chunk.page_content[:200]}...")

def load_and_process_all_documents():
    """
    Load, process, and create vector store for all document types.
    
    Returns:
        Tuple containing (vector_store, all_chunks)
    """
    # Get paths
    paths = get_project_paths()
    
    # Initialize embedding function
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load documents
    interview_guides = load_documents_from_directory(paths["interview_guides"])
    technical_docs = load_documents_from_directory(paths["technical_categories"])
    trend_docs = load_documents_from_directory(paths["trends"])
    
    # Create text splitter
    text_splitter = create_text_splitter()
    
    # Process documents
    interview_chunks = process_documents_with_metadata(interview_guides, "interview_guide", text_splitter)
    technical_chunks = process_documents_with_metadata(technical_docs, "technical", text_splitter)
    trend_chunks = process_documents_with_metadata(trend_docs, "trend", text_splitter)
    
    # Validate processing
    is_valid = validate_document_loading(interview_chunks, technical_chunks, trend_chunks)
    if not is_valid:
        raise ValueError("Document loading failed. Please check your file paths and structure.")
    
    # Combine all chunks
    all_chunks = interview_chunks + technical_chunks + trend_chunks
    logger.info(f"Total chunks prepared for embedding: {len(all_chunks)}")
    
    # Create vector store
    vector_store = create_or_load_vectorstore(
        embedding_function=embedding_function,
        documents=all_chunks,
        persist_directory=paths["vectorstore"],
        force_recreate=False
    )
    
    return vector_store, all_chunks

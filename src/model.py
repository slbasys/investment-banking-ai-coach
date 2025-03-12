# src/model.py
"""
Model module for the Investment Banking AI Coach.

This module contains the core RAG implementation and response generation logic.
It includes mode detection, system prompts, and the agent state schema.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define different system prompts
CORE_PROFESSIONAL_PROMPT = """
You are Finance Bro, an expert investment banking interview coach with extensive experience in preparing candidates for roles at bulge bracket and elite boutique banks. Your knowledge spans technical topics including valuation methodologies, M&A analysis, LBO modeling, accounting concepts, and industry-specific questions.

For company-specific questions, you can provide historical stock data and company information, but not real-time stock prices. When discussing stock prices, make it clear that you're providing recent historical data, not current market prices.

You provide accurate, technically precise information while maintaining a supportive coaching approach. Your guidance is based on real interview experiences and industry best practices.
"""

JUNIOR_CANDIDATE_ADDON = """
Recognize that you're working with an entry-level or undergraduate candidate. Break down complex concepts into digestible components, avoid assuming prior knowledge, provide fundamental examples, and explain industry terminology. Focus on basic valuation concepts, financial statement structure, and entry-level technical questions. Use an encouraging, educational tone.
"""

EXPERIENCED_CANDIDATE_ADDON = """
Recognize that you're working with an experienced candidate (MBA, lateral hire, or someone with prior finance experience). Elevate your technical depth with advanced modeling techniques, nuanced transaction analyses, and complex valuation methodologies. Reference specific deal types and industry-specific considerations. Expect higher technical proficiency and use a more collegial tone.
"""

BRO_MODE_PROMPT = """
ACTIVATE BRO MODE: You are now in "Bro mode" - still an expert investment banking coach but with the personality of a stereotypical Wall Street finance bro. Use casual language, industry jargon, and references to finance culture (Patagonia vests, client dinners, deal all-nighters). Include phrases like "crushing it," "absolute beast mode," and references to "the Street." Keep your technical information 100% accurate but deliver it with finance bro flair.
"""

def determine_mode_by_keywords(query: str):
    """
    Determine the mode based on keywords in the query.
    
    Args:
        query: The user's query string
        
    Returns:
        Tuple of (experience_level, bro_mode)
    """
    # Check for experience level indicators
    junior_keywords = ["beginner", "new to", "entry level", "undergraduate", "basics", "simple"]
    experienced_keywords = ["advanced", "complex", "MBA", "experienced", "senior", "technical"]
    
    # Check for bro mode indicators
    bro_keywords = ["bro", "bro mode", "finance bro", "wall street bro", "crush it"]
    
    # Determine experience level
    experience_level = "standard"
    if any(keyword in query.lower() for keyword in junior_keywords):
        experience_level = "junior"
    elif any(keyword in query.lower() for keyword in experienced_keywords):
        experience_level = "experienced"
    
    # Determine if bro mode is active
    bro_mode = any(keyword in query.lower() for keyword in bro_keywords)
    
    return experience_level, bro_mode

def detect_company_ticker(query: str) -> str:
    """
    Detect company mentions in a query and map to ticker symbols.
    
    Args:
        query: The user's query string
        
    Returns:
        Company ticker symbol if detected, empty string otherwise
    """
    company_map = {
        # Major tech companies
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", 
        "amazon": "AMZN", "meta": "META", "facebook": "META",
        "tesla": "TSLA", "netflix": "NFLX", "nvidia": "NVDA",
        
        # Financial companies
        "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
        "bank of america": "BAC", "wells fargo": "WFC",
        
        # Add more companies as needed
    }
    
    # Check for company name mentions
    for company, ticker in company_map.items():
        if company.lower() in query.lower():
            return ticker
    
    # Check for direct ticker mentions
    words = query.split()
    for word in words:
        # Check if word is an uppercase 1-5 letter word (likely a ticker)
        if word.isupper() and 1 <= len(word) <= 5 and word in [t.upper() for t in company_map.values()]:
            return word
    
    return ""

def construct_system_prompt(experience_level="standard", bro_mode=False):
    """
    Construct the system prompt based on experience level and bro mode.
    
    Args:
        experience_level: Level of experience ("junior", "standard", "experienced")
        bro_mode: Whether to activate bro mode
        
    Returns:
        Constructed system prompt string
    """
    prompt = CORE_PROFESSIONAL_PROMPT
    
    if experience_level == "junior":
        prompt += "\n\n" + JUNIOR_CANDIDATE_ADDON
    elif experience_level == "experienced":
        prompt += "\n\n" + EXPERIENCED_CANDIDATE_ADDON
    
    if bro_mode:
        prompt += "\n\n" + BRO_MODE_PROMPT
    
    return prompt

# Define agent state schema
class FinanceCoachState(AgentState):
    """State for the Finance Interview Coach agent"""
    question_category: str = ""  # Track category of questions (valuation, accounting, etc.)
    company_ticker: str = ""     # For company-specific questions
    retrieved_documents: List[Document] = []  # Store retrieved documents
    n_contexts: int = 5  # Number of context documents to retrieve

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model("gpt-4o-mini", temperature=0, model_provider="openai")

# Load vector store
def get_vector_store():
    """
    Initialize and return the vector store for document retrieval.
    
    Returns:
        Initialized Chroma vector store
    """
    # Path to the main vector store
    vectorstore_path = os.path.join(os.environ.get('DIR_PROJECT', '.'), 'vectorstore')
    
    # Load the vector store
    try:
        vector_store = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings
        )
        logger.info(f"Loaded vector store with {vector_store._collection.count()} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise

# Get vector store instance
vector_store = get_vector_store()

@tool
def retrieve_finance_content(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve relevant finance interview content based on query.
    
    Args:
        query: The user's query
        state: The agent state
        
    Returns:
        Formatted context string with relevant documents
    """
    try:
        # Default to 5 documents if n_contexts not specified
        n_contexts = state.get("n_contexts", 5)
        
        # Get company ticker from state or detect it from query
        company_ticker = state.get("company_ticker", "")
        if not company_ticker:
            company_ticker = detect_company_ticker(query)
            if company_ticker:
                logger.info(f"Detected company ticker: {company_ticker}")
                # Update the state with the detected ticker
                state["company_ticker"] = company_ticker

        # Build a filter that matches your metadata structure
        filter_query = None
        if state.get("question_category"):
            # Use compound filter with $and and $eq for more reliable matching
            filter_query = {"$and": [{"category": {"$eq": state.get("question_category")}}]}

        # First, search general finance interview content
        contexts = vector_store.similarity_search(
            query,
            k=n_contexts,
            filter=filter_query
        )

        # If no results found with filter, try without filter
        if not contexts and filter_query:
            logger.info(f"No results found with filter. Trying without filter...")
            contexts = vector_store.similarity_search(query, k=n_contexts)
        
        # If company ticker is available, try to get company-specific information
        company_docs = []
        if company_ticker:
            try:
                # Check if company-specific RAG index exists
                rag_path = os.environ.get('DIR_RAG_INDEXES')
                company_index_path = os.path.join(rag_path, company_ticker)
                
                if os.path.exists(company_index_path):
                    logger.info(f"Found company-specific RAG index for {company_ticker}")
                    # Load the company-specific vector store
                    company_vector_store = Chroma(
                        collection_name="sec_filings",
                        embedding_function=embeddings,
                        persist_directory=company_index_path
                    )
                    
                    # Search in company-specific documents
                    company_docs = company_vector_store.similarity_search(
                        query,
                        k=2  # Limit to top 2 company-specific docs
                    )
                    
                    if company_docs:
                        logger.info(f"Retrieved {len(company_docs)} company-specific documents for {company_ticker}")
                        # Add metadata to company documents
                        for doc in company_docs:
                            doc.metadata["source"] = f"{company_ticker} SEC Filing"
                            doc.metadata["category"] = "company_specific"
            except Exception as company_error:
                logger.error(f"Error retrieving company-specific data: {company_error}")

        # Combine company-specific documents with general documents
        if company_docs:
            # Insert company docs at the beginning for higher relevance
            contexts = company_docs + contexts

        # Format the context string
        context_str = ""
        for i, doc in enumerate(contexts):
            source = doc.metadata.get('source', 'Unknown')
            category = doc.metadata.get('category', 'General')
            context_str += f"# CONTEXT DOCUMENT {i}\nSource: {source}\nCategory: {category}\n{doc.page_content}\n\n"

        # Update state with retrieved documents
        state["retrieved_documents"] = contexts

        return context_str if contexts else "No relevant documents found."

    except Exception as e:
        logger.error(f"Error in retrieve_finance_content: {e}")
        return f"Error retrieving information: {str(e)}"

# Initialize agent with tools
finance_coach_agent = None

def initialize_agent(additional_tools=None):
    """
    Initialize and return the finance coach agent.
    
    Args:
        additional_tools: Optional list of additional tools to add
        
    Returns:
        Initialized agent
    """
    global finance_coach_agent
    
    # Import tools here to avoid circular imports
    from src.tools import get_stock_data, get_company_details
    
    # Register tools with the agent
    tools = [retrieve_finance_content, get_stock_data, get_company_details]
    
    # Add any additional tools
    if additional_tools:
        tools.extend(additional_tools)
    
    # Create the agent with tools and state schema
    finance_coach_agent = create_react_agent(
        llm,
        tools,
        state_schema=FinanceCoachState
    )
    
    return finance_coach_agent

def process_finance_query(query: str, category: str = "", company_ticker: str = "", 
                         experience_level: str = None, bro_mode: bool = None) -> str:
    """
    Process a finance interview question through the RAG pipeline.
    
    Args:
        query: The interview question or prompt
        category: Optional category (e.g., "technical", "behavioral")
        company_ticker: Optional company ticker for company-specific questions
        experience_level: Explicitly set experience level ("junior", "standard", "experienced")
        bro_mode: Explicitly enable/disable bro mode
        
    Returns:
        String containing the AI response
    """
    global finance_coach_agent
    
    # Initialize agent if not already done
    if finance_coach_agent is None:
        initialize_agent()
    
    # Determine mode based on keywords in the query if not explicitly provided
    if experience_level is None or bro_mode is None:
        auto_exp_level, auto_bro_mode = determine_mode_by_keywords(query)
        experience_level = experience_level or auto_exp_level
        bro_mode = bro_mode if bro_mode is not None else auto_bro_mode
    
    # Construct the system prompt based on detected modes
    system_prompt = construct_system_prompt(experience_level, bro_mode)
    
    # If no company ticker provided, detect it from the query
    if not company_ticker:
        company_ticker = detect_company_ticker(query)
    
    # Log the active modes for debugging
    logger.info(f"Invoking agent with:")
    logger.info(f"- Experience level: {experience_level}")
    logger.info(f"- Bro mode: {'ON' if bro_mode else 'OFF'}")
    logger.info(f"- Category: {category}")
    logger.info(f"- Company ticker: {company_ticker if company_ticker else 'None'}")
    
    # Create initial state with dynamic system prompt
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ],
        "question_category": category,
        "company_ticker": company_ticker,
        "retrieved_documents": [],
        "n_contexts": 5
    }
    
    # Invoke the agent
    result = finance_coach_agent.invoke(initial_state)
    
    # Return the final response
    return result["messages"][-1].content

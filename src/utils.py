"""
Utility functions for the Investment Banking AI Coach.

This module provides helper functions used across the project,
including text formatting, LaTeX conversion, and utility functions
for working with financial data.
"""

import logging
import os
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_latex_content(content: str) -> str:
    """
    Format LaTeX content for better readability.
    
    Args:
        content: String containing LaTeX notation
        
    Returns:
        Formatted string with more readable math notation
    """
    # Replace block math with box formatting
    content = content.replace("\\[", "\n┌─ FORMULA ─────────────────────┐\n│ ")
    content = content.replace("\\]", " │\n└──────────────────────────────┘\n")
    
    # Replace inline math with bracket formatting
    content = content.replace("\\(", "「")
    content = content.replace("\\)", "」")
    
    # Replace common LaTeX symbols
    content = content.replace("\\beta", "β")
    content = content.replace("\\alpha", "α")
    content = content.replace("\\delta", "δ")
    content = content.replace("\\gamma", "γ")
    content = content.replace("\\lambda", "λ")
    content = content.replace("\\sigma", "σ")
    content = content.replace("\\pi", "π")
    content = content.replace("\\theta", "θ")
    content = content.replace("\\text{", "")
    content = content.replace("\\left(", "(")
    content = content.replace("\\right)", ")")
    content = content.replace("\\times", "×")
    content = content.replace("\\frac{", "(")
    content = content.replace("}", ")")
    
    return content

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

def determine_mode_by_keywords(query: str) -> tuple:
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

def extract_question_category(query: str) -> str:
    """
    Extract the likely category of a finance interview question.
    
    Args:
        query: The user's query string
        
    Returns:
        Category string (technical, behavioral, etc.)
    """
    # Define keyword mappings for different categories
    category_keywords = {
        "technical": [
            "valuation", "dcf", "wacc", "capm", "lbo", "m&a", "merger", "acquisition",
            "financial statement", "balance sheet", "income statement", "cash flow", 
            "accounting", "ratio", "ebitda", "pe ratio", "irr", "npv"
        ],
        "behavioral": [
            "tell me about yourself", "why investment banking", "strengths", "weaknesses",
            "leadership", "teamwork", "challenge", "conflict", "achievement", "fail",
            "why our bank", "culture fit", "work ethic", "motivate"
        ],
        "market": [
            "market", "industry", "trend", "economy", "recent deal", "transaction",
            "news", "sector", "ipo", "stock market", "interest rate", "economic"
        ]
    }
    
    # Check for category matches
    for category, keywords in category_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            return category
    
    # Default to empty string if no clear category
    return ""

def format_financial_data(data: Dict) -> str:
    """
    Format financial data into a readable string.
    
    Args:
        data: Dictionary containing financial data
        
    Returns:
        Formatted string with financial information
    """
    if "error" in data:
        return f"Error retrieving financial data: {data['error']}"
    
    # Format stock price data
    if "close_price" in data:
        result = [
            f"Financial Data for {data.get('ticker', 'Unknown')}:",
            f"Date: {data.get('timestamp', 'N/A')}",
            f"Close Price: ${data.get('close_price', 'N/A')}",
            f"Open Price: ${data.get('open_price', 'N/A')}",
            f"High: ${data.get('high_price', 'N/A')}",
            f"Low: ${data.get('low_price', 'N/A')}",
            f"Volume: {data.get('volume', 'N/A'):,}"
        ]
        return "\n".join(result)
    
    # Format company details
    elif "name" in data:
        result = [
            f"Company Information for {data.get('name', data.get('ticker', 'Unknown'))}:",
            f"Ticker: {data.get('ticker', 'N/A')}",
            f"Industry: {data.get('industry', 'N/A')}",
            f"Market Cap: ${data.get('market_cap', 'N/A'):,}" if data.get('market_cap') else "Market Cap: N/A",
            f"Exchange: {data.get('primary_exchange', 'N/A')}",
            f"Website: {data.get('homepage_url', 'N/A')}",
            f"\nDescription: {data.get('description', 'N/A')}"
        ]
        return "\n".join(result)
    
    # Generic formatting
    else:
        return "\n".join([f"{k}: {v}" for k, v in data.items() if k != "error"])

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length and add ellipsis.
    
    Args:
        text: String to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated string with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."

def create_context_string(documents: List, include_metadata: bool = True) -> str:
    """
    Create a formatted context string from a list of documents.
    
    Args:
        documents: List of Document objects
        include_metadata: Whether to include metadata in the output
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    context_str = ""
    for i, doc in enumerate(documents):
        context_str += f"# CONTEXT DOCUMENT {i+1}\n"
        
        if include_metadata and hasattr(doc, 'metadata'):
            for key, value in doc.metadata.items():
                if key in ['source', 'category', 'type']:
                    context_str += f"{key.capitalize()}: {value}\n"
        
        context_str += f"{doc.page_content}\n\n"
        
    return context_str

def get_project_paths():
    """
    Get project directory paths from environment variables.
    
    Returns:
        Dictionary containing project paths
    """
    directory_project = os.environ.get('DIR_PROJECT')
    directory_rag_indexes = os.environ.get('DIR_RAG_INDEXES')
    
    if not directory_project:
        logger.warning("Environment variable DIR_PROJECT not set")
        directory_project = '.'
    
    if not directory_rag_indexes:
        logger.warning("Environment variable DIR_RAG_INDEXES not set")
        directory_rag_indexes = os.path.join(directory_project, 'Rag_Index_Data')
    
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

def is_finance_question(query: str) -> bool:
    """
    Determine if a query is related to finance or investment banking.
    
    Args:
        query: The user's query string
        
    Returns:
        Boolean indicating if the query is finance-related
    """
    finance_keywords = [
        "investment", "banking", "finance", "stock", "market", "valuation",
        "dcf", "wacc", "m&a", "merger", "acquisition", "private equity", "hedge fund",
        "financial", "statement", "balance sheet", "income", "cash flow", "accounting",
        "ratio", "capital", "leverage", "debt", "equity", "asset", "liability",
        "revenue", "earnings", "profit", "loss", "dividend", "yield", "interest",
        "rate", "bond", "security", "portfolio", "risk", "return", "investment bank",
        "bulge bracket", "boutique", "advisory", "deal", "transaction", "ipo"
    ]
    
    return any(keyword in query.lower() for keyword in finance_keywords)

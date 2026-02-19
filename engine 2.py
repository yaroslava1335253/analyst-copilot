# engine.py
"""
Analyst Co-Pilot Engine
=======================
This module handles:
1. Financial data fetching (yfinance)
2. SEC 10-K document retrieval (sec-edgar-downloader)
3. Document chunking and vector storage (ChromaDB)
4. Citation-aware LLM analysis (OpenAI)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd
from openai import OpenAI

# SEC Edgar imports
from sec_edgar_downloader import Downloader

# LangChain imports for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ChromaDB for vector storage
import chromadb
from chromadb.config import Settings

# --- Configuration ---
SEC_DOWNLOAD_DIR = Path("./sec_filings")
CHROMA_PERSIST_DIR = Path("./chroma_db")

# --- Core Functions for Financial Analysis ---

def get_financials(ticker_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the income statement and balance sheet for a given stock ticker.
    Returns two Pandas DataFrames.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        income_statement = stock.income_stmt
        balance_sheet = stock.balance_sheet
        return income_statement, balance_sheet
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- SEC 10-K Fetching ---

def fetch_10k_filing(ticker: str, year: Optional[int] = None, email: str = "your-email@example.com") -> dict:
    """
    Downloads the 10-K filing for a given ticker from SEC EDGAR.
    
    Args:
        ticker: Stock ticker symbol (e.g., "MSFT")
        year: Specific fiscal year (optional, defaults to most recent)
        email: Email for SEC EDGAR identification (required by SEC)
    
    Returns:
        dict with keys: 'success', 'filepath', 'text', 'error'
    """
    SEC_DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    try:
        # Initialize downloader with required email
        dl = Downloader(company_name="AnalystCopilot", email_address=email, download_folder=str(SEC_DOWNLOAD_DIR))
        
        # Download the 10-K filings (gets most recent by default)
        if year:
            dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year}-12-31")
        else:
            dl.get("10-K", ticker, limit=1)  # Get most recent
        
        # Find the downloaded file
        ticker_dir = SEC_DOWNLOAD_DIR / "sec-edgar-filings" / ticker.upper() / "10-K"
        
        if not ticker_dir.exists():
            return {"success": False, "error": f"No 10-K found for {ticker}", "filepath": None, "text": None}
        
        # Get the most recent filing directory
        filing_dirs = sorted(ticker_dir.iterdir(), reverse=True)
        if not filing_dirs:
            return {"success": False, "error": f"No 10-K filings downloaded for {ticker}", "filepath": None, "text": None}
        
        # Read the full submission text file
        latest_dir = filing_dirs[0]
        text_file = latest_dir / "full-submission.txt"
        
        if text_file.exists():
            text_content = text_file.read_text(encoding='utf-8', errors='ignore')
            # Clean up the text (remove HTML tags, extra whitespace)
            text_content = _clean_filing_text(text_content)
            return {
                "success": True,
                "filepath": str(text_file),
                "text": text_content,
                "filing_date": latest_dir.name,
                "error": None
            }
        else:
            return {"success": False, "error": "Filing downloaded but text file not found", "filepath": None, "text": None}
            
    except Exception as e:
        return {"success": False, "error": str(e), "filepath": None, "text": None}


def _clean_filing_text(text: str) -> str:
    """
    Cleans SEC filing text by removing HTML tags and normalizing whitespace.
    """
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove common SEC header noise
    text = re.sub(r'CONFORMED SUBMISSION TYPE.*?FILER:', '', text, flags=re.DOTALL)
    return text.strip()


# --- Vector Memory Layer (ChromaDB) ---

def get_chroma_client():
    """
    Returns a persistent ChromaDB client.
    """
    CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def build_document_index(text: str, ticker: str, filing_date: str) -> str:
    """
    Chunks the 10-K text and stores it in ChromaDB with metadata.
    
    Args:
        text: The full 10-K filing text
        ticker: Stock ticker for identification
        filing_date: Filing date for metadata
    
    Returns:
        collection_name: Name of the ChromaDB collection created
    """
    # Create a unique collection name based on ticker and filing
    collection_name = f"{ticker.lower()}_{filing_date}".replace("-", "_").replace("/", "_")
    
    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger chunks = fewer embeddings = faster
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the text into chunks (limit to first 500 chunks for speed)
    chunks = text_splitter.split_text(text)[:500]
    
    print(f"[Engine] Indexing {len(chunks)} chunks for {ticker}...")
    
    # Get ChromaDB client and create/get collection
    client = get_chroma_client()
    
    # Delete existing collection if it exists (for fresh indexing)
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"ticker": ticker, "filing_date": filing_date}
    )
    
    # Add chunks with metadata in batches for better performance
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        batch_metadatas = [{"chunk_index": j, "ticker": ticker, "source": f"10-K ({filing_date})"} for j in range(i, i+len(batch_chunks))]
        
        collection.add(
            documents=batch_chunks,
            ids=batch_ids,
            metadatas=batch_metadatas
        )
        print(f"[Engine] Indexed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    print(f"[Engine] Indexing complete for {ticker}")
    return collection_name


def query_with_sources(question: str, collection_name: str, n_results: int = 5) -> list[dict]:
    """
    Queries the vector store and returns relevant chunks with source metadata.
    
    Args:
        question: The user's question or analysis request
        collection_name: Name of the ChromaDB collection to query
        n_results: Number of relevant chunks to retrieve
    
    Returns:
        List of dicts with keys: 'text', 'source', 'chunk_index', 'relevance_score'
    """
    client = get_chroma_client()
    
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        return [{"error": f"Collection not found: {e}"}]
    
    # Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    # Format results with source information
    formatted_results = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        formatted_results.append({
            "text": doc,
            "source": metadata.get("source", "Unknown"),
            "chunk_index": metadata.get("chunk_index", i),
            "relevance_score": 1 - distance  # Convert distance to similarity
        })
    
    return formatted_results


# --- Citation-Aware LLM Analysis ---

def run_adversarial_analysis(
    analysis_type: str,
    context_chunks: list[dict],
    financials_data: Optional[pd.DataFrame] = None,
    company_name: str = "the company"
) -> dict:
    """
    Runs citation-aware analysis using adversarial personas.
    
    Args:
        analysis_type: "skeptic" or "believer"
        context_chunks: Retrieved chunks from the 10-K
        financials_data: Optional financial data DataFrame
        company_name: Name of the company being analyzed
    
    Returns:
        dict with keys: 'analysis', 'claims' (list of claim objects with citations)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build context string from retrieved chunks
    context_str = "\n\n---\n\n".join([
        f"[Source: {chunk['source']}, Chunk #{chunk['chunk_index']}]\n{chunk['text']}"
        for chunk in context_chunks if 'error' not in chunk
    ])
    
    # Add financial data if provided
    if financials_data is not None and not financials_data.empty:
        context_str += f"\n\n---\n\n[Source: Financial Statements]\n{financials_data.to_string()}"
    
    # Define adversarial personas
    personas = {
        "skeptic": {
            "system": """You are a SKEPTICAL CREDIT ANALYST. Your job is to find problems, risks, and red flags.
You must be critical and challenge the company's narrative. Focus on:
- Declining metrics or concerning trends
- Risk factors and potential downsides
- Aggressive accounting or questionable assumptions
- Competitive threats and market risks

CRITICAL: You MUST cite your sources. For EVERY claim you make, include the exact quote from the source material that supports it.""",
            
            "user_template": f"""Analyze {company_name} from a SKEPTICAL perspective. 

Based on the following 10-K excerpts and data, identify the TOP 3 RISKS or RED FLAGS.

For each risk, you MUST provide:
1. The risk/concern (be specific)
2. The EXACT quote from the source that supports this concern
3. Why this matters to an investor

SOURCE MATERIAL:
{context_str}

Respond in this JSON format:
{{
    "summary": "One paragraph executive summary of key concerns",
    "claims": [
        {{
            "claim": "The specific risk or concern",
            "source_quote": "The EXACT text from the document that supports this",
            "source_location": "Which source/chunk this came from",
            "significance": "Why this matters"
        }}
    ]
}}"""
        },
        
        "believer": {
            "system": """You are an OPTIMISTIC GROWTH INVESTOR. Your job is to find opportunities and growth drivers.
You believe in the company's potential but must back up your optimism with evidence. Focus on:
- Revenue growth and market expansion
- Competitive advantages and moats
- Innovation and product pipeline
- Strong management execution

CRITICAL: You MUST cite your sources. For EVERY claim you make, include the exact quote from the source material that supports it.""",
            
            "user_template": f"""Analyze {company_name} from a BULLISH perspective.

Based on the following 10-K excerpts and data, identify the TOP 3 GROWTH DRIVERS or OPPORTUNITIES.

For each opportunity, you MUST provide:
1. The growth driver (be specific)
2. The EXACT quote from the source that supports this opportunity
3. Why this is compelling for investors

SOURCE MATERIAL:
{context_str}

Respond in this JSON format:
{{
    "summary": "One paragraph executive summary of the bull case",
    "claims": [
        {{
            "claim": "The specific growth driver or opportunity",
            "source_quote": "The EXACT text from the document that supports this",
            "source_location": "Which source/chunk this came from",
            "significance": "Why this is compelling"
        }}
    ]
}}"""
        }
    }
    
    persona = personas.get(analysis_type, personas["skeptic"])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a more capable model for citation accuracy
            messages=[
                {"role": "system", "content": persona["system"]},
                {"role": "user", "content": persona["user_template"]}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {"success": True, "analysis": result}
        
    except Exception as e:
        return {"success": False, "error": str(e), "analysis": None}


def run_structured_prompt(prompt_type, financials_data):
    """
    Sends a structured, pre-defined prompt to the OpenAI API to mitigate bias.
    (Legacy function - kept for backward compatibility)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    data_string = financials_data.to_string()

    prompts = {
        "risk_analysis": f"Act as a skeptical credit analyst. Based on the following financial data, identify the top 3 most significant risks to the company's performance. Be specific and reference the data provided.\n\nData:\n{data_string}",
        "bull_case": f"Act as an optimistic equity analyst. Based on the following financial data, summarize the primary growth drivers and the 'bull case' for the stock. Be specific and reference the data provided.\n\nData:\n{data_string}"
    }

    prompt_text = prompts.get(prompt_type, "You are a helpful financial assistant.")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst co-pilot."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the AI model: {e}"


# --- Chat with Memory ---

def chat_with_context(
    user_message: str,
    collection_name: str,
    chat_history: list[dict],
    company_name: str = "the company"
) -> dict:
    """
    Handles follow-up questions with full context awareness.
    
    Args:
        user_message: The user's follow-up question
        collection_name: ChromaDB collection to query for context
        chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        company_name: Name of the company
    
    Returns:
        dict with 'response' and 'sources_used'
    """
    # Retrieve relevant context for this question
    context_chunks = query_with_sources(user_message, collection_name, n_results=5)
    
    context_str = "\n\n".join([
        f"[{chunk['source']}]: {chunk['text']}"
        for chunk in context_chunks if 'error' not in chunk
    ])
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build messages with history
    messages = [
        {"role": "system", "content": f"""You are an expert financial analyst assistant helping analyze {company_name}.
You have access to their 10-K filing and financial data. Always cite your sources when making claims.
If you don't know something or it's not in the provided context, say so clearly."""}
    ]
    
    # Add chat history
    messages.extend(chat_history[-10:])  # Keep last 10 messages for context
    
    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Question: {user_message}

Relevant context from 10-K filing:
{context_str}

Please answer the question based on the provided context. Cite specific quotes when possible."""
    })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        return {
            "success": True,
            "response": response.choices[0].message.content,
            "sources_used": [chunk['source'] for chunk in context_chunks if 'error' not in chunk]
        }
    except Exception as e:
        return {"success": False, "error": str(e), "response": None}


# --- You can test this file directly ---
if __name__ == '__main__':
    ticker = "MSFT"
    print(f"--- Fetching financials for {ticker} ---")
    inc_stmt, bal_sht = get_financials(ticker)
    print("Income Statement:")
    print(inc_stmt.head())
    
    print(f"\n--- Testing SEC 10-K Fetch for {ticker} ---")
    result = fetch_10k_filing(ticker)
    if result["success"]:
        print(f"Successfully downloaded 10-K, text length: {len(result['text'])} characters")
    else:
        print(f"Failed to fetch 10-K: {result['error']}")
    
    if os.getenv("OPENAI_API_KEY"):
        print("\n--- Running Risk Analysis via AI ---")
        risk_summary = run_structured_prompt("risk_analysis", inc_stmt)
        print(risk_summary)
    else:
        print("\nSkipping AI test: OPENAI_API_KEY not set.")

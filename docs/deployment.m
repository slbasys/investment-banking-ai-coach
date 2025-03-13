# Deployment Guide

This guide covers how to deploy the Finance Bro AI Investment Banking Interview Coach to different environments.

## Table of Contents
- [Local Deployment](#local-deployment)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Environment Variables](#environment-variables)
- [Vector Store Handling](#vector-store-handling)
- [Troubleshooting](#troubleshooting)

## Local Deployment

### Prerequisites
- Python 3.8 or higher
- Required API keys (OpenAI, Polygon, Brave Search)
- Git (optional, for cloning the repository)

### Steps

1. **Clone the repository** (if not done already):
   ```bash
   git clone https://github.com/yourusername/investment-banking-ai-coach.git
   cd investment-banking-ai-coach
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY=your_openai_key
   export POLYGON_API_KEY=your_polygon_key
   export BRAVE_API_KEY=your_brave_search_key
   export DIR_PROJECT=$(pwd)
   export DIR_RAG_INDEXES=$(pwd)/Rag_Index_Data

   # Windows
   set OPENAI_API_KEY=your_openai_key
   set POLYGON_API_KEY=your_polygon_key
   set BRAVE_API_KEY=your_brave_search_key
   set DIR_PROJECT=%cd%
   set DIR_RAG_INDEXES=%cd%\Rag_Index_Data
   ```

4. **Run Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   The app will be available at http://localhost:8501

## Streamlit Cloud Deployment

### Prerequisites
- GitHub repository with your Finance Bro AI code
- Streamlit Cloud account (sign up at [streamlit.io/cloud](https://streamlit.io/cloud))
- Required API keys (same as local deployment)

### Steps

1. **Prepare your repository**:
   - Ensure your code is in a GitHub repository
   - Verify the repository contains:
     - `app.py` (Streamlit entry point)
     - `requirements.txt` (dependencies)
     - Properly organized code in `src/` directory

2. **Create .streamlit/secrets.toml file locally for testing**:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your_openai_key"
   POLYGON_API_KEY = "your_polygon_key"
   BRAVE_API_KEY = "your_brave_search_key"
   ```
   
   > ⚠️ **Note**: Don't commit this file to GitHub; it contains sensitive information.

3. **Update app.py to use Streamlit secrets**:
   ```python
   import streamlit as st
   import os
   
   # Set environment variables from Streamlit secrets
   os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
   os.environ["POLYGON_API_KEY"] = st.secrets["POLYGON_API_KEY"]
   os.environ["BRAVE_API_KEY"] = st.secrets["BRAVE_API_KEY"]
   
   # Set project directories
   os.environ["DIR_PROJECT"] = "."
   os.environ["DIR_RAG_INDEXES"] = "./Rag_Index_Data"
   
   # Rest of your Streamlit app code...
   ```

4. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Select your GitHub repository, branch, and the `app.py` file
   - Add your secrets in the "Advanced settings" section
   - Deploy!

5. **Add vector store data**:
   - If your app uses pre-built vector stores, you may need to use GitHub LFS
   - Alternatively, add code to build the vector store on first run
   - Consider using a persistent database service for production

## Environment Variables

Your Finance Bro AI application requires the following environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| OPENAI_API_KEY | API key for OpenAI services | Yes |
| POLYGON_API_KEY | API key for Polygon.io financial data | Yes |
| BRAVE_API_KEY | API key for Brave Search | Yes |
| DIR_PROJECT | Path to the project directory | Yes |
| DIR_RAG_INDEXES | Path to RAG indexes directory | Yes |

## Vector Store Handling

### Option 1: Include Pre-built Vector Store
- Pros: Faster startup, consistent performance
- Cons: Larger repository size, potential version issues
- Implementation: Use Git LFS or include build script

### Option 2: Build Vector Store on First Run
```python
import os
from src.data_processing import load_and_process_all_documents

# Check if vector store exists
if not os.path.exists("./vectorstore"):
    st.info("Building vector store for the first time. This may take a few minutes...")
    vector_store, _ = load_and_process_all_documents()
    st.success("Vector store built successfully!")
else:
    st.info("Using existing vector store")
```

### Option 3: Use Cloud Storage (Production)
For production deployments, consider using cloud storage options:
- AWS S3 for storing vector data
- MongoDB Atlas for document storage
- Pinecone for vector search

## Troubleshooting

### Common Issues

#### API Rate Limits
- **Symptom**: Errors about exceeded rate limits
- **Solution**: Implement rate limiting in your code or upgrade API plans

#### Memory Issues
- **Symptom**: Application crashes with memory errors
- **Solution**: Reduce vector store size, optimize embedding dimensions, or increase server memory

#### Slow Response Times
- **Symptom**: Queries take too long to process
- **Solution**: Optimize vector search, cache common queries, or use a more powerful deployment environment

#### Missing Dependencies
- **Symptom**: Import errors or missing module errors
- **Solution**: Ensure all dependencies are in requirements.txt and properly installed

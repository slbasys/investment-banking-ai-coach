# Finance Bro AI Investment Banking Interview Coach: Usage Guide

This document provides comprehensive guidance on using the Finance Bro AI Investment Banking Interview Coach, a specialized system designed to help candidates prepare for investment banking interviews through personalized coaching.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Basic Usage](#basic-usage)
- [Experience Levels](#experience-levels)
- [Bro Mode](#bro-mode)
- [Company-Specific Questions](#company-specific-questions)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Example Questions](#example-questions)

## Installation and Setup

### Prerequisites
Before using Finance Bro AI, ensure you have:
- Python 3.8 or higher
- Required API keys:
  - OpenAI API key
  - Polygon.io API key
  - Brave Search API key

### Installation
1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your API keys:
   ```python
   # In your code or environment variables
   os.environ["OPENAI_API_KEY"] = "your-openai-key"
   os.environ["POLYGON_API_KEY"] = "your-polygon-key"
   os.environ["BRAVE_API_KEY"] = "your-brave-search-key"
   ```

3. Ensure your document database is properly loaded with investment banking materials.

## Basic Usage

The simplest way to use Finance Bro AI is through the `process_finance_query` function:

```python
from src.model import process_finance_query

# Ask a basic finance question
response = process_finance_query("How do you calculate WACC?")
print(response)
```

The system automatically:
- Retrieves relevant financial information from its knowledge base
- Formats responses in a conversational, coaching style
- Includes relevant formulas and examples where appropriate

## Experience Levels

Finance Bro AI adapts its responses based on the user's experience level, which can be:

### Junior Mode
For undergraduate or entry-level candidates with limited finance knowledge:

```python
# Triggered by keywords like "beginner," "new to," "entry level"
response = process_finance_query("As a beginner, how would you explain DCF?")

# Or explicitly setting junior mode
response = process_finance_query("How do you calculate WACC?", experience_level="junior")
```

In Junior Mode, responses include:
- More basic explanations
- Definitions of industry terminology
- Fundamental examples
- Step-by-step breakdowns

### Standard Mode
The default mode for candidates with some finance background:

```python
response = process_finance_query("Walk me through a DCF model")
```

### Experienced Mode
For MBA or experienced candidates:

```python
# Triggered by keywords like "advanced," "complex," "MBA"
response = process_finance_query("As an experienced candidate, how would you approach a complex merger model?")

# Or explicitly setting experienced mode
response = process_finance_query("Explain synergies in M&A", experience_level="experienced")
```

In Experienced Mode, responses include:
- Advanced modeling techniques
- Nuanced transaction analyses
- Complex valuation methodologies
- Industry-specific considerations

## Bro Mode

Finance Bro AI features a "Bro Mode" that presents information in the style of a stereotypical Wall Street finance professional. This mode maintains technical accuracy while using a more casual, industry-jargon heavy tone.

```python
# Activated by keywords like "bro," "finance bro," "wall street bro"
response = process_finance_query("Hey bro, how do I crush a DCF question?")

# Or explicitly enable bro mode
response = process_finance_query("How to value a company?", bro_mode=True)
```

In Bro Mode, you'll notice:
- More casual language
- Finance culture references
- Industry jargon
- Motivational "crush it" style encouragement

## Company-Specific Questions

Ask about specific companies to get tailored information:

```python
# Automatically detects company mentions
response = process_finance_query("Tell me about Apple's business model")

# Or explicitly specify a company ticker
response = process_finance_query("What's their recent performance?", company_ticker="AAPL")
```

The system provides:
- Company descriptions and industry information
- Recent historical stock performance data
- Business model explanations
- Industry positioning

Note: Financial data is historical, not real-time, and only available for publicly traded companies.

## Advanced Usage

### Combining Multiple Modes

You can combine experience levels with bro mode and company specifics:

```python
response = process_finance_query(
    "Hey bro, I'm a beginner, can you explain how Apple's P/E ratio compares to the industry?",
    company_ticker="AAPL"
)
```

### Category Filtering

Specify question categories for more targeted responses:

```python
# Technical questions
response = process_finance_query("How do you calculate enterprise value?", category="technical")

# For behavioral questions
response = process_finance_query("How should I answer 'Why investment banking?'", category="behavioral")
```

## Troubleshooting

### API Key Issues
If you encounter errors related to API keys:
- Verify all three required API keys are correctly set
- Check for API usage limits or restrictions
- Ensure proper environment variable configuration

### No Relevant Information Retrieved
If responses seem generic or unhelpful:
- Try rephrasing your question
- Use more specific terminology
- Explicitly state the category of your question
- Verify the document database is properly loaded

### Company Information Not Found
If company-specific information is not appearing:
- Ensure the company is publicly traded
- Check if you're using the correct company name or ticker
- Verify the Polygon API connection is working

## Example Questions

Here are some example questions to get you started:

### Technical Questions
- "Walk me through a DCF valuation"
- "How do you calculate WACC?"
- "What's the difference between enterprise value and equity value?"
- "How do the three financial statements link together?"
- "Explain the concept of accretion/dilution in M&A"

### Junior-Level Questions
- "I'm a beginner. Can you explain what an LBO is?"
- "As an undergraduate, how should I prepare for technical questions?"
- "Explain enterprise value in simple terms"

### Experienced-Level Questions
- "As an MBA candidate, how would you approach a complex merger model?"
- "What are the nuances of different DCF approaches for mature vs. growth companies?"
- "How would you adjust a valuation for a high-growth tech company with negative earnings?"

### Company-Specific Questions
- "What is Apple's business model?"
- "How has Tesla's stock performed recently?"
- "Tell me about JPMorgan's main business segments"

### Bro Mode Questions
- "Bro, how do I crush a technical interview at Goldman?"
- "Hey finance bro, walk me through a killer DCF explanation"
- "Bro, what's the best way to talk about comps in an interview?"

# Investment Banking AI Coach

An AI-powered interview coach that helps candidates prepare for investment banking interviews using retrieval-augmented generation (RAG) with dynamic mode adaptation for different experience levels.

## Features

- Retrieval from curated investment banking interview materials
- Financial data integration via Polygon API
- Web search augmentation via Brave Search
- Dynamic system prompts based on candidate experience level
- "Bro mode" for a more casual, finance-culture oriented personality
- Context-aware responses with source attribution

## Installation

pip install -r requirements.txt

## Usage

from finance_bro import process_finance_query

## Ask a technical question

response = process_finance_query("How do you calculate WACC?", category="technical")
print(response)

## Ask in bro mode

response = process_finance_query("Hey bro, explain DCF to me.", category="technical")
print(response)

## Deployment

This project can be deployed as a Streamlit web application. See `deployment.md` for instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

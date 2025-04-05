# Academic Paper Search Agent

A comprehensive academic paper search tool that aggregates research papers from multiple sources including Google Scholar, Semantic Scholar, arXiv, IEEE Xplore, and PubMed. The tool calculates relevance scores for search results and can generate literature surveys based on retrieved papers.

## Features

- Multi-source academic paper search (Google Scholar, Semantic Scholar, arXiv, IEEE Xplore, PubMed)
- Asynchronous concurrent searching for improved performance
- Relevance scoring using sentence transformers
- Automated literature survey generation using Gemini 1.5 Pro
- Advanced text sanitization for cleaner output

## Prerequisites

- Python 3.9+
- Google Generative AI API key
- IEEE Xplore API key (optional)
- Internet connection for web scraping

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/srinivas2200030391/Academic-Paper-Search-Agent.git
cd Academic-Paper-Search-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_generative_ai_api_key
IEEE_API_KEY=your_ieee_xplore_api_key  # Optional
```

### 5. Run the application

```bash
python main.py
```

## How to Get a Google Gemini API Key

To use the Google Gemini API for automated literature survey generation, follow these steps:

1. Go to the [Google AI Studio](https://aistudio.google.com/) website.
2. Sign in with your Google account.
3. Navigate to the **API Keys** section.
4. Click on **Create API Key** and follow the on-screen instructions.
5. Once generated, copy the API key and store it securely.
6. Add the API key to your `.env` file as shown above.

## Usage

```python
import asyncio
from academic_paper_search_agent import AcademicPaperSearchAgent

# Initialize the agent
agent = AcademicPaperSearchAgent(api_key="your_google_api_key")

# Define research topics
topics = ["transformer neural networks", "quantum computing algorithms"]

# Run the search
async def main():
    results = await agent.run_paper_search(topics)
    
    # Process results
    for topic, data in results.items():
        print(f"=== Topic: {topic} ===")
        print(f"Found {len(data['papers'])} papers")
        print(f"Literature Survey:\n{data['survey'][:500]}...")

# Run the async main function
asyncio.run(main())
```

## Requirements

- google-generativeai
- requests
- beautifulsoup4
- sentence-transformers
- scikit-learn
- python-dotenv
- asyncio

## License

MIT


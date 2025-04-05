import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import asyncio
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import os


class AcademicPaperSearchAgent:
    def __init__(self, api_key: str):
        """
        Initialize the Academic Paper Search Agent
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Add sentence transformer for relevance scoring
        self.relevance_model = SentenceTransformer('all-MiniLM-L6-v2')

    async def search_papers(self, query: str, num_results: int = 10):
        """
        Perform comprehensive academic paper search concurrently
        """
        search_methods = [
            self._search_google_scholar,
            self._search_semantic_scholar,
            self._search_arxiv,
            self._search_pubmed
        ]

        tasks = [method(query, num_results) for method in search_methods]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        results = [res for res in results if isinstance(res, list)]
        combined_results = [paper for sublist in results for paper in sublist]

        return combined_results[:num_results]

    async def _search_google_scholar(self, query: str, num_results: int = 5):
        """
        Search papers using Google Scholar
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0,5"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('div', class_='gs_ri')[:num_results]:
                title_elem = result.find('h3', class_='gs_rt')
                if title_elem and title_elem.find('a'):
                    title = title_elem.find('a').text.strip()
                    link = title_elem.find('a')['href']
                    # Extract additional details
                    author_info = result.find('div', class_='gs_a')
                    authors = []
                    year = 'N/A'
                    if author_info:
                        try:
                            author_text = author_info.text.strip()
                            parts = author_text.split('-')[0].split(',')
                            authors = [part.strip() for part in parts[:3]]  # Limit to first 3 authors

                            # Try to extract year
                            year_match = [part for part in author_text.split() if part.isdigit()]
                            if year_match:
                                year = year_match[0]
                        except:
                            pass
                    results.append({
                        'title': title,
                        'url': link,
                        'authors': authors,
                        'year': year,
                        'source': 'Google Scholar'
                    })

            return results
        return []

    async def _search_semantic_scholar(self, query: str, num_results: int = 5):
        """
        Search papers on Semantic Scholar
        """
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "limit": num_results,
            "fields": "title,url,authors,year"
        }

        headers = {
            "Accept": "application/json"
        }

        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            results = []

            for paper in data.get('data', []):
                results.append({
                    'title': paper.get('title', 'N/A'),
                    'url': paper.get('url', 'N/A'),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])][:3],
                    'year': paper.get('year', 'N/A'),
                    'source': 'Semantic Scholar'
                })

            return results
        return []

    async def _search_pubmed(self, query: str, num_results: int = 5):
        """
        Search papers on PubMed using NCBI E-utilities
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

        # Search for articles
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": num_results,
            "sort": "relevance"
        }

        try:
            # Initial search to get article IDs
            search_response = requests.get(base_url, params=search_params)

            if search_response.status_code == 200:
                # Parse XML response
                root = ET.fromstring(search_response.text)

                # Extract article IDs
                id_list = root.findall(".//Id")
                article_ids = [id_elem.text for id_elem in id_list]

                # Fetch details for these articles
                summary_params = {
                    "db": "pubmed",
                    "id": ",".join(article_ids),
                    "retmode": "xml"
                }

                summary_response = requests.get(summary_url, params=summary_params)

                if summary_response.status_code == 200:
                    summary_root = ET.fromstring(summary_response.text)
                    results = []

                    for doc_sum in summary_root.findall(".//DocSum"):
                        # Extract title
                        title_elem = doc_sum.find(".//Item[@Name='Title']")
                        title = title_elem.text if title_elem is not None else 'N/A'

                        # Extract authors
                        authors_elem = doc_sum.find(".//Item[@Name='AuthorList']")
                        authors = []
                        if authors_elem is not None:
                            authors = [
                                          author.text
                                          for author in authors_elem.findall("./Item")
                                          if author.text
                                      ][:3]  # Limit to first 3 authors

                        # Extract publication year
                        pubdate_elem = doc_sum.find(".//Item[@Name='PubDate']")
                        year = pubdate_elem.text[:4] if pubdate_elem is not None and pubdate_elem.text else 'N/A'

                        # Construct PubMed URL
                        pubmed_id = doc_sum.find(".//Id").text
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"

                        results.append({
                            'title': title,
                            'url': url,
                            'authors': authors,
                            'year': year,
                            'source': 'PubMed'
                        })

                    return results
        except Exception as e:
            print(f"❌ PubMed Search Error: {e}")

        return []

    async def _search_arxiv(self, query: str, num_results: int = 5):
        """
        Search papers on arXiv
        """
        base_url = "http://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": num_results
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'xml')
            results = []

            for entry in soup.find_all('entry')[:num_results]:
                title = entry.find('title').text.strip()
                link = entry.find('id').text.strip()

                # Extract authors
                authors = [author.text.strip() for author in entry.find_all('author')][:3]

                # Extract publication year
                published = entry.find('published')
                year = published.text[:4] if published else 'N/A'

                results.append({
                    'title': title,
                    'url': link,
                    'authors': authors,
                    'year': year,
                    'source': 'arXiv'
                })

            return results
        return []

    def calculate_relevance_scores(self, query: str, papers: list) -> list:
        """
        Calculate relevance scores for retrieved papers using sentence transformer

        Args:
            query (str): The original search query
            papers (list): List of retrieved papers

        Returns:
            list: Papers with added relevance scores
        """
        try:
            # Encode query and paper titles
            query_embedding = self.relevance_model.encode([query])

            # Prepare paper titles for embedding
            paper_titles = [paper.get('title', '') for paper in papers]
            title_embeddings = self.relevance_model.encode(paper_titles)

            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, title_embeddings)[0]

            # Add relevance scores to papers
            for paper, score in zip(papers, similarities):
                paper['relevance_score'] = round(float(score) * 100, 2)

            # Sort papers by relevance score in descending order
            papers.sort(key=lambda x: x['relevance_score'], reverse=True)

            return papers

        except Exception as e:
            print(f"❌ Relevance Scoring Error: {e}")
            return papers

    async def generate_literature_survey(self, papers):
        """
        Generate a comprehensive literature survey
        """
        try:
            # Prepare detailed paper context
            context = "\n\n".join([
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {', '.join(paper.get('authors', []))}\n"
                f"Year: {paper.get('year', 'N/A')}\n"
                f"Source: {paper.get('source', 'N/A')}\n"
                f"URL: {paper.get('url', 'N/A')}"
                for paper in papers
            ])

            prompt = f"""Comprehensive Literature Survey Analysis

Paper Context:
{context}

Provide an in-depth literature survey with:
1. Comprehensive Thematic Overview
2. Critical Research Contributions
3. Methodological Insights and Comparative Analysis
4. Emerging Trends and Interdisciplinary Connections
5. Identified Research Gaps and Future Directions

Guidelines:
- Maintain a rigorous academic tone
- Highlight interconnections between papers
- Provide nuanced critical insights
- Synthesize cross-cutting themes
- Offer forward-looking perspective"""

            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 2048,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            print(f"❌ Literature Survey Generation Error: {e}")
            return f"Error generating survey: {str(e)}"

    def sanitize_literature_review(self, text):
        """
        Sanitize the literature review text by:
        1. Removing markdown-style headers (##, ###)
        2. Removing asterisk bullet points (*)
        3. Removing extra whitespace
        4. Cleaning up line breaks
        """
        # Remove markdown-style headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Remove asterisk bullet points
        text = re.sub(r'^\*\s*', '', text, flags=re.MULTILINE)

        # Remove leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        # Collapse multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    async def run_paper_search(self, topics):
        """
        Run comprehensive paper search across multiple topics
        """
        print(topics)
        results = {}
        for topic in topics:
            # Perform searches across multiple sources
            paper_results = await self.search_papers(topic, num_results=10)

            # Calculate relevance scores
            paper_results_with_scores = self.calculate_relevance_scores(topic, paper_results)

            # Generate literature survey
            literature_survey = await self.generate_literature_survey(paper_results_with_scores)

            sanitized_survey = self.sanitize_literature_review(literature_survey)

            results[topic] = {
                'papers': paper_results_with_scores,
                'survey': sanitized_survey
            }

        return results

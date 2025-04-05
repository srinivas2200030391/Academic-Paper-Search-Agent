import os
from flask import Flask, render_template, request, jsonify
import asyncio
from AcadPaperSearch import AcademicPaperSearchAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Flask Application
app = Flask(__name__)


@app.route('/')
def index():
    """
    Render the main index page
    """
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search_papers():
    """
    Handle paper search requests
    """
    # Get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        return jsonify({
            'error': 'No API key provided'
        }), 400

    # Get topics from request
    topics = request.form.getlist('topics')

    if not topics:
        return jsonify({
            'error': 'No topics provided'
        }), 400

    # Create search agent
    agent = AcademicPaperSearchAgent(api_key)

    # Run search asynchronously
    results = asyncio.run(agent.run_paper_search(topics))
    print(results)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

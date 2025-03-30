import os
import json
import re
import datetime
import feedparser
import requests
import hashlib
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from typing import List, Dict, Tuple, Optional
from flask_cors import CORS
from urllib.parse import urlparse, urljoin
import time
import random

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # Required for session management

# RSS feed URL and website base URL
RSS_FEED_URL = "https://www.dhakapost.com/rss/rss.xml"
BASE_URL = "https://www.dhakapost.com"
SEARCH_URL = "https://www.dhakapost.com/search"

# Cache for RSS feed data, search results, and summaries
feed_cache = {
    "last_updated": None,
    "entries": []
}

search_cache = {}
summary_cache = {}
fact_check_cache = {}

def create_perplexity_style_response(query, relevant_articles, language="english"):
    """
    Create a Perplexity.ai style response based on relevant articles.
    
    Parameters:
    query (str): The user's query
    relevant_articles (list): List of relevant article dictionaries
    language (str): Response language (english or bengali)
    
    Returns:
    str: A formatted response in Perplexity style
    """
    # If no articles found
    if not relevant_articles:
        if language.lower() == "bengali":
            return "দুঃখিত, আপনার প্রশ্নের সাথে সম্পর্কিত কোন সাম্প্রতিক খবর পাওয়া যায়নি। অন্য কিছু জিজ্ঞাসা করতে চান?"
        else:
            return "Sorry, I couldn't find any recent articles relevant to your query. Would you like to ask about something else?"
    
    # Extract key information from articles
    titles = [article['title'] for article in relevant_articles]
    sources = [f"Dhaka Post ({article.get('published', 'Recent')})" for article in relevant_articles]
    
    # Compile main content from articles
    contents = []
    for article in relevant_articles:
        content = article.get('content', '') or article.get('text', '') or article.get('summary', '')
        if content:
            # Clean content if needed
            content = re.sub(r'<.*?>', '', content)  # Remove HTML tags
            contents.append(content)
    
    combined_content = " ".join(contents)
    
    # Extract key sentences based on query terms
    query_terms = set(query.lower().split())
    sentences = re.split(r'(?<=[।.!?])\s+', combined_content)
    
    # Score sentences based on query relevance
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        relevance_score = len(query_terms.intersection(sentence_words))
        if relevance_score > 0:
            scored_sentences.append((relevance_score, sentence))
    
    # Sort by relevance score and get top sentences
    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:5]
    relevant_sentences = [s[1] for s in top_sentences]
    
    # Prepare citation links
    citations = []
    for i, article in enumerate(relevant_articles[:3]):
        citations.append(f"[{i+1}] {article['link']}")
    
    # Create direct answer (first part of response)
    if language.lower() == "bengali":
        # Bengali response format
        direct_answer = f"**{titles[0]}**\n\n"
        
        if relevant_sentences:
            direct_answer += f"{relevant_sentences[0]}\n\n"
        
        # Add more context with citations
        body = ""
        used_sentences = set()
        for i, article in enumerate(relevant_articles[:3]):
            body += f"**{article['title']}** [{i+1}]\n\n"
            
            # Add relevant sentences from this article
            article_content = article.get('content', '') or article.get('text', '') or article.get('summary', '')
            article_sentences = re.split(r'(?<=[।.!?])\s+', article_content)
            
            relevant_from_article = []
            for sentence in article_sentences:
                if sentence in relevant_sentences and sentence not in used_sentences:
                    relevant_from_article.append(sentence)
                    used_sentences.add(sentence)
            
            if relevant_from_article:
                body += f"{' '.join(relevant_from_article[:3])}\n\n"
            else:
                # If no specific sentences matched, use the first few sentences
                body += f"{' '.join(article_sentences[:2])}\n\n"
        
        # Add citation links
        citation_text = "\n".join(citations)
        
        # Create summary
        if len(relevant_articles) > 1:
            summary = f"উপরের তথ্য {len(relevant_articles)} টি প্রাসঙ্গিক প্রবন্ধ থেকে সংকলিত করা হয়েছে। আরও জানতে, সম্পূর্ণ প্রবন্ধগুলি দেখুন।"
        else:
            summary = "উপরের তথ্য ঢাকা পোস্ট থেকে সংকলিত করা হয়েছে। আরও বিস্তারিত জানতে, সম্পূর্ণ প্রবন্ধটি দেখুন।"
        
        return f"{direct_answer}{body}\n{citation_text}\n\n{summary}"
    
    else:
        # English response format
        direct_answer = f"**{titles[0]}**\n\n"
        
        if relevant_sentences:
            direct_answer += f"{relevant_sentences[0]}\n\n"
        
        # Add more context with citations
        body = ""
        used_sentences = set()
        for i, article in enumerate(relevant_articles[:3]):
            body += f"**{article['title']}** [{i+1}]\n\n"
            
            # Add relevant sentences from this article
            article_content = article.get('content', '') or article.get('text', '') or article.get('summary', '')
            article_sentences = re.split(r'(?<=[.!?])\s+', article_content)
            
            relevant_from_article = []
            for sentence in article_sentences:
                if sentence in relevant_sentences and sentence not in used_sentences:
                    relevant_from_article.append(sentence)
                    used_sentences.add(sentence)
            
            if relevant_from_article:
                body += f"{' '.join(relevant_from_article[:3])}\n\n"
            else:
                # If no specific sentences matched, use the first few sentences
                body += f"{' '.join(article_sentences[:2])}\n\n"
        
        # Add citation links
        citation_text = "\n".join(citations)
        
        # Create summary
        if len(relevant_articles) > 1:
            summary = f"The information above is compiled from {len(relevant_articles)} relevant articles. For more details, check the full articles."
        else:
            summary = "The information above is sourced from Dhaka Post. For more details, check the full article."
        
        return f"{direct_answer}{body}\n{citation_text}\n\n{summary}"

def search_dhakapost(query: str, max_results: int = 10) -> List[Dict]:
    """Search the Dhaka Post website for relevant articles with improved matching."""
    # Check cache first
    cache_key = query.lower()
    if cache_key in search_cache and search_cache[cache_key]["timestamp"] > datetime.datetime.now() - datetime.timedelta(hours=1):
        print(f"Using cached search results for '{query}'")
        return search_cache[cache_key]["results"]
    
    results = []
    try:
        # Add a small delay to prevent rate limiting
        time.sleep(random.uniform(1, 2))
        
        # Format the search query for the URL
        search_query = query.replace(" ", "+")
        
        # Make the search request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(f"{SEARCH_URL}?q={search_query}", headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article listings - adjust these selectors based on Dhaka Post's actual HTML structure
            # These are example selectors and need to be modified based on the actual website
            article_elements = soup.select('.search-results .article-item') or \
                              soup.select('.news-list .news-item') or \
                              soup.select('article') or \
                              soup.select('.search-results a')
            
            for article in article_elements[:max_results]:
                # Extract link
                link_tag = article.find('a') or article
                link = link_tag.get('href', '')
                if link and not link.startswith('http'):
                    link = urljoin(BASE_URL, link)
                
                # Extract title
                title_tag = article.find('h2') or article.find('h3') or article.find('h4')
                title = title_tag.text.strip() if title_tag else "Untitled Article"
                
                # Extract date if available
                date_tag = article.select_one('.date') or article.select_one('.time') or article.select_one('.published')
                date = date_tag.text.strip() if date_tag else ""
                
                # Extract snippet if available
                snippet_tag = article.select_one('.summary') or article.select_one('.excerpt') or article.select_one('p')
                snippet = snippet_tag.text.strip() if snippet_tag else ""
                
                if link:  # Only add if we have a valid link
                    results.append({
                        "title": title,
                        "link": link,
                        "published": date,
                        "summary": snippet,
                        "source": "Dhaka Post Search",
                        "relevance_score": 0  # Will be calculated later
                    })
        
        # Cache the results
        search_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.datetime.now()
        }
        
        print(f"Found {len(results)} search results for '{query}'")
        
    except Exception as e:
        print(f"Error searching Dhaka Post: {e}")
    
    return results

def fetch_article_content(url: str) -> Dict:
    """Fetch and parse the full content of an article."""
    # Generate cache key from URL
    cache_key = f"content_{url}"
    if cache_key in search_cache and search_cache[cache_key]["timestamp"] > datetime.datetime.now() - datetime.timedelta(hours=1):
        print(f"Using cached article content for '{url}'")
        return search_cache[cache_key]["content"]
    
    article_data = {
        "title": "",
        "published": "",
        "content": "",
        "link": url,
        "source": "Dhaka Post",
        "author": "",
        "categories": []
    }
    
    try:
        # Add a small delay to prevent rate limiting
        time.sleep(random.uniform(1, 2))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('h1')
            if title_tag:
                article_data["title"] = title_tag.text.strip()
            
            # Extract date
            date_tag = soup.select_one('.date') or soup.select_one('.time') or soup.select_one('.published')
            if date_tag:
                article_data["published"] = date_tag.text.strip()
            
            # Extract author
            author_tag = soup.select_one('.author') or soup.select_one('.byline')
            if author_tag:
                article_data["author"] = author_tag.text.strip()
            
            # Extract categories/tags
            category_tags = soup.select('.category a') or soup.select('.tags a')
            if category_tags:
                article_data["categories"] = [tag.text.strip() for tag in category_tags]
            
            # Extract content
            # Adjust these selectors based on Dhaka Post's actual HTML structure
            content_selectors = [
                'article .content',
                '.article-content',
                '.news-content',
                'article p',
                '.entry-content'
            ]
            
            for selector in content_selectors:
                content_elements = soup.select(selector)
                if content_elements:
                    article_content = ' '.join([p.text.strip() for p in content_elements])
                    article_data["content"] = article_content
                    break
            
            # Cache the article content
            search_cache[cache_key] = {
                "content": article_data,
                "timestamp": datetime.datetime.now()
            }
            
            print(f"Successfully fetched article content: {article_data['title']}")
        
    except Exception as e:
        print(f"Error fetching article content: {e}")
    
    return article_data

def generate_daily_summary() -> str:
    """Generate a summary of today's top news stories."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    cache_key = f"daily_summary_{current_date}"
    
    # Check if we already have a summary for today
    if cache_key in summary_cache:
        return summary_cache[cache_key]
    
    # Get the latest articles
    articles = fetch_rss_feed(force_refresh=True)
    
    # Filter for today's articles
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    todays_articles = []
    
    for article in articles:
        try:
            # Parse the publication date - adjust format as needed
            pub_date = article.get("published", "")
            if pub_date:
                # Handle different date formats
                for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y"]:
                    try:
                        article_date = datetime.datetime.strptime(pub_date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched, check if today's date is in the string
                    article_date = today if today in pub_date else ""
                
                if article_date == today:
                    todays_articles.append(article)
        except Exception as e:
            print(f"Error parsing date for article: {e}")
    
    # If we don't have today's articles, use the most recent ones
    if not todays_articles and articles:
        todays_articles = articles[:10]  # Use top 10 recent articles
    
    # Get the most important topics by clustering articles
    topic_clusters = cluster_articles_by_topic(todays_articles)
    
    # Create a summary for each topic cluster
    summary_text = f"# Daily News Summary for {today}\n\n"
    
    for i, (topic, cluster_articles) in enumerate(topic_clusters.items(), 1):
        summary_text += f"## {topic}\n\n"
        
        for article in cluster_articles[:3]:  # Top 3 articles per topic
            summary_text += f"- **{article['title']}**"
            
            if article.get("published"):
                summary_text += f" ({article['published']})"
            
            summary_text += f"\n  {article['link']}\n\n"
            
            # Add a brief excerpt
            if article.get("summary"):
                excerpt = article["summary"][:200] + "..." if len(article["summary"]) > 200 else article["summary"]
                summary_text += f"  {excerpt}\n\n"
    
    # Cache the summary
    summary_cache[cache_key] = summary_text
    
    return summary_text

def cluster_articles_by_topic(articles: List[Dict]) -> Dict[str, List[Dict]]:
    """Group articles into topic clusters using simple keyword matching."""
    # This is a simplified version - in production, use a more sophisticated clustering algorithm
    topics = {
        "Politics": ["government", "minister", "parliament", "election", "political", "party", "vote", "democracy"],
        "Business": ["economy", "business", "market", "stock", "trade", "economic", "finance", "company", "bank"],
        "Health": ["health", "hospital", "doctor", "patient", "disease", "medical", "covid", "virus", "vaccine"],
        "Technology": ["technology", "tech", "digital", "internet", "software", "computer", "app", "online"],
        "Sports": ["sport", "football", "cricket", "game", "tournament", "match", "player", "team"],
        "International": ["international", "global", "world", "foreign", "country", "nation", "diplomat"]
    }
    
    # Initialize clusters
    clusters = {topic: [] for topic in topics}
    clusters["Other"] = []  # For articles that don't match any topic
    
    for article in articles:
        article_text = f"{article['title']} {article.get('text', '')}".lower()
        
        # Find the best matching topic
        max_score = 0
        best_topic = "Other"
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword.lower() in article_text)
            if score > max_score:
                max_score = score
                best_topic = topic
        
        # Add to the appropriate cluster
        clusters[best_topic].append(article)
    
    # Remove empty clusters
    return {topic: articles for topic, articles in clusters.items() if articles}
def fetch_rss_feed(force_refresh=False) -> List[Dict]:
    """Fetch and parse the RSS feed."""
    global feed_cache
    if not force_refresh and feed_cache["last_updated"] and \
            datetime.datetime.now() - feed_cache["last_updated"] < datetime.timedelta(hours=1):
        print("Using cached RSS feed data.")
        return feed_cache["entries"]

    try:
        response = requests.get(RSS_FEED_URL)
        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            entries = []
            for entry in feed.entries:
                entries.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.published,
                    "summary": entry.summary,
                    "source": "Dhaka Post RSS"
                })
            feed_cache = {
                "last_updated": datetime.datetime.now(),
                "entries": entries
            }
            print("Fetched new RSS feed data.")
            return entries
        else:
            print(f"Failed to fetch RSS feed: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return []

def calculate_article_relevance(article: Dict, query: str) -> float:
    """Calculate the relevance score of an article for a given query."""
    query_words = set(query.lower().split())
    
    # Get article text
    article_text = f"{article['title']} {article.get('text', '') or article.get('summary', '')}"
    
    # Calculate basic word overlap
    article_words = set(article_text.lower().split())
    word_overlap = len(query_words.intersection(article_words))
    
    # Calculate TF-IDF-like score (simplified)
    score = 0
    for word in query_words:
        if word in article_text.lower():
            # Word in title gets higher weight
            if word in article['title'].lower():
                score += 3
            else:
                score += 1
    
    # Recency bonus (if we have publication date)
    recency_bonus = 0
    if article.get("published"):
        try:
            # Try to parse the date - adjust format as needed
            for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y"]:
                try:
                    pub_date = datetime.datetime.strptime(article["published"], fmt)
                    days_old = (datetime.datetime.now() - pub_date).days
                    recency_bonus = max(0, 5 - days_old/2)  # Higher score for newer articles
                    break
                except ValueError:
                    continue
        except Exception:
            pass
    
    # Combine scores
    final_score = score + recency_bonus
    
    return final_score

def find_relevant_articles(query: str, top_k: int = 5) -> List[Dict]:
    """Find the most relevant articles for a given query."""
    # First, check RSS feed for recent articles
    feed_articles = fetch_rss_feed()
    
    # Then, search the website for more results
    search_results = search_dhakapost(query)
    
    # Combine both sources
    all_articles = feed_articles + search_results
    
    # Remove duplicates based on URL
    unique_articles = {}
    for article in all_articles:
        if article["link"] not in unique_articles:
            unique_articles[article["link"]] = article
    
    all_articles = list(unique_articles.values())
    
    # Calculate relevance scores
    for article in all_articles:
        article["relevance_score"] = calculate_article_relevance(article, query)
    
    # Sort by relevance score
    sorted_articles = sorted(all_articles, key=lambda x: x["relevance_score"], reverse=True)
    
    # Get top_k articles
    relevant_articles = sorted_articles[:top_k]
    
    # For the most relevant articles, fetch their full content if needed
    for i, article in enumerate(relevant_articles):
        if "content" not in article or not article["content"]:
            full_article = fetch_article_content(article["link"])
            # Preserve the relevance score when updating
            relevance_score = article["relevance_score"]
            relevant_articles[i].update(full_article)
            relevant_articles[i]["relevance_score"] = relevance_score
    
    return relevant_articles

def check_fact(statement: str) -> Dict:
    """Check the factuality of a statement against our news corpus."""
    # Create a cache key based on the statement
    cache_key = hashlib.md5(statement.encode()).hexdigest()
    
    if cache_key in fact_check_cache and fact_check_cache[cache_key]["timestamp"] > datetime.datetime.now() - datetime.timedelta(days=1):
        return fact_check_cache[cache_key]["result"]
    
    # Get all recent articles
    all_articles = fetch_rss_feed()
    
    # Extract key entities and claims from the statement
    # This is a simplified approach - real fact-checking would use NLP
    words = statement.lower().split()
    important_words = [w for w in words if len(w) > 3 and w not in ["this", "that", "then", "than", "with", "from"]]
    
    # Search for articles that mention these entities
    matching_articles = []
    for article in all_articles:
        article_text = f"{article['title']} {article.get('text', '')}"
        matches = sum(1 for word in important_words if word in article_text.lower())
        if matches >= len(important_words) * 0.5:  # At least 50% of important words match
            matching_articles.append(article)
    
    # Determine fact check result
    if not matching_articles:
        result = {
            "status": "Unverified",
            "confidence": 0,
            "explanation": "No relevant news articles found to verify this statement.",
            "related_articles": []
        }
    else:
        # For a simple implementation, we'll assume statements with strong article support are likely true
        # A real system would do detailed claim extraction and comparison
        confidence = min(len(matching_articles) * 0.2, 0.9)  # 0.9 max confidence
        
        result = {
            "status": "Likely True" if confidence > 0.5 else "Unverified",
            "confidence": confidence,
            "explanation": f"Found {len(matching_articles)} relevant articles that corroborate aspects of this statement.",
            "related_articles": [{"title": a["title"], "link": a["link"]} for a in matching_articles[:3]]
        }
    
    # Cache the result
    fact_check_cache[cache_key] = {
        "result": result,
        "timestamp": datetime.datetime.now()
    }
    


    return result

def summarize_articles(articles: List[Dict]) -> str:
    """Create a summary of relevant articles for the system prompt."""
    if not articles:
        return "No relevant articles found."
    
    summaries = []
    
    for i, article in enumerate(articles, 1):
        # Format article information
        summary = f"Article {i}:\nTitle: {article['title']}\n"
        
        if article.get("published"):
            summary += f"Published: {article['published']}\n"
        
        if article.get("author"):
            summary += f"Author: {article['author']}\n"
            
        summary += f"Source: {article['source']}\n"
        summary += f"URL: {article['link']}\n"
        summary += f"Relevance Score: {article.get('relevance_score', 0):.2f}\n\n"
        
        # Add content summary
        content = article.get("content", "") or article.get("text", "") or article.get("summary", "")
        if len(content) > 1000:
            summary += f"Content: {content[:1000]}...\n\n"
        else:
            summary += f"Content: {content}\n\n"
        
        summaries.append(summary)
    
    return "\n".join(summaries)

def generate_article_summary(article: Dict) -> str:
    """Generate a concise summary of a news article."""
    if not article.get("content") and not article.get("text"):
        return "Insufficient content to generate summary."
    
    # Create a cache key
    cache_key = f"summary_{article['link']}"
    if cache_key in summary_cache:
        return summary_cache[cache_key]
    
    # Extract the main content
    content = article.get("content", "") or article.get("text", "")
    
    # Simple extractive summarization (more advanced models would be better)
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Score sentences based on position and key terms
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        
        # Position score (first few sentences often contain key information)
        position_score = 1.0 if i < 3 else (0.5 if i < 5 else 0.1)
        
        # Length score (favor medium-length sentences)
        words = len(sentence.split())
        length_score = 0.5 if 10 <= words <= 25 else (0.3 if words < 10 else 0.1)
        
        # Keyword score (check for important terms)
        title_words = set(article['title'].lower().split())
        sentence_words = set(sentence.lower().split())
        keyword_score = len(title_words.intersection(sentence_words)) * 0.2
        
        score = position_score + length_score + keyword_score
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    summary_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:3]
    
    # Re-order sentences to maintain original flow
    summary_sentences.sort(key=lambda x: sentences.index(x[1]))
    
    summary = " ".join([s[1] for s in summary_sentences])
    
    # Cache the summary
    summary_cache[cache_key] = summary
    
    return summary

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
try:
    config_data = json.load(open(f"{working_dir}/config.json"))
    GROQ_API_KEY = config_data["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    print("Warning: config.json not found or missing GROQ_API_KEY. Please provide API key.")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Initialize Groq client
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

def create_news_prompt(query, relevant_articles):
    articles_summary = summarize_articles(relevant_articles)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""You are a helpful news assistant that responds in the style of Perplexity.ai, specializing in discussing the latest news from Bangladesh, particularly from Dhaka Post.

    Current time: {current_time}

    Based on the user's query: "{query}", here are the most relevant recent articles:

    {articles_summary}

    IMPORTANT FORMATTING INSTRUCTIONS:
    1. Structure your response like Perplexity.ai with these components:
    - Start with a bold headline that directly answers the query
    - Begin with the most relevant fact or information
    - Include 2-3 paragraphs of details from the articles
    - For each major claim, add a citation like [Source: Dhaka Post]
    - Add numbered source links at the end [1] https://...
    - End with a brief summarizing statement

    2. For responses in Bengali, follow the same structure but in Bengali script
    3. Be concise and information-dense like Perplexity.ai
    4. Use Markdown formatting for bold, links, etc.
    5. Avoid phrases like "According to the provided articles"
    6. Always maintain a journalistic, objective tone
    7. If you don't have enough information, acknowledge this limitation

    Remember: Users value clear, direct answers with proper citations."""

@app.route('/')
def index():
    # Initialize or reset conversation history when loading the main page
    session['conversation_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('query', '')

    if not user_prompt:
        return jsonify({"error": "Query cannot be empty"}), 400

    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Use the correct function here
    relevant_articles = find_relevant_articles(user_prompt)  # Assuming this function exists
    system_prompt = create_news_prompt(user_prompt, relevant_articles)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session['conversation_history'][-10:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.7,
            max_tokens=700,
        )

        assistant_response = response.choices[0].message.content
        session['conversation_history'].append({"role": "user", "content": user_prompt})
        session['conversation_history'].append({"role": "assistant", "content": assistant_response})
        session.modified = True

        return jsonify({
            "response": assistant_response,
            "conversation_history": session['conversation_history']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history."""
    session['conversation_history'] = []
    return jsonify({"message": "Conversation history reset successfully"})

@app.route('/refresh', methods=['POST'])
def refresh_feed():
    """Force refresh the RSS feed cache."""
    fetch_rss_feed(force_refresh=True)
    # Also clear search cache
    global search_cache, summary_cache, fact_check_cache
    search_cache = {}
    summary_cache = {}
    fact_check_cache = {}
    return jsonify({"message": "RSS feed and all caches refreshed successfully"})

@app.route('/summarize', methods=['POST'])
def summarize_news():
    """Generate a summary of today's top news."""
    summary = generate_daily_summary()
    return jsonify({"summary": summary})

@app.route('/fact-check', methods=['POST'])
def fact_check():
    """Check a claim against our news database."""
    data = request.json
    claim = data.get('claim', '')
    
    if not claim:
        return jsonify({"error": "No claim provided"}), 400
    
    result = check_fact(claim)
    return jsonify(result)

# Create a more advanced HTML template for the chat interface

if __name__ == '__main__':
    app.run(debug=True)

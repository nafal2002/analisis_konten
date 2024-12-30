import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from textblob import TextBlob
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from main import save_to_csv

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Set up YouTube Data API credentials
API_KEY = 'AIzaSyD3cQ9wxJuRCYs1nqnnTJbU96u2ucSqJMY'  # Replace with your YouTube Data API key

# Function to get video metadata
def get_video_metadata(youtube, video_id):
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    )
    response = request.execute()

    if "items" in response and len(response["items"]) > 0:
        video_data = response["items"][0]
        metadata = {
            "title": video_data["snippet"].get("title"),
            "description": video_data["snippet"].get("description"),
            "tags": video_data["snippet"].get("tags"),
            "category": video_data["snippet"].get("categoryId"),
            "published_at": video_data["snippet"].get("publishedAt"),
            "view_count": int(video_data["statistics"].get("viewCount", 0)),
            "like_count": int(video_data["statistics"].get("likeCount", 0)),
            "comment_count": int(video_data["statistics"].get("commentCount", 0))
        }
        return metadata
    else:
        return None

# Function to get video comments
def get_video_comments(youtube, video_id, max_results=100):
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(comment["textDisplay"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:  # Exit loop if no more comments
            break

    return comments

# Function to analyze sentiment with VADER
def analyze_sentiment_vader(comments):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_results = []
    for comment in comments:
        sentiment = analyzer.polarity_scores(comment)
        sentiment_results.append(sentiment['compound'])  # Using compound score for overall sentiment
    return sentiment_results

# Function to extract hashtags
def extract_hashtags(comments):
    hashtags = []
    for comment in comments:
        hashtags += re.findall(r'#\w+', comment)  # Extract hashtags
    return hashtags

# Function to analyze AI-related sentiment
def analyze_ai_sentiment(comments, ai_keywords=['AI', 'artificial intelligence']):
    ai_sentiments = []
    for comment in comments:
        if any(keyword.lower() in comment.lower() for keyword in ai_keywords):
            analysis = TextBlob(comment)
            ai_sentiments.append(analysis.sentiment.polarity)
    return ai_sentiments

# Visualize sentiment analysis results
def visualize_sentiments(sentiments, save_path="sentiment_analysis.png"):
    sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
    sentiment_df["Category"] = pd.cut(
        sentiment_df["Sentiment"], 
        bins=[-1, -0.01, 0.01, 1], 
        labels=["Negative", "Neutral", "Positive"]
    )

    sentiment_counts = sentiment_df["Category"].value_counts()

    # Plot the sentiments
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title("Sentiment Analysis of Comments")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a .png file
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

# Preprocessing function
def preprocess_comment(comment):
    # Remove HTML tags
    comment = re.sub(r'<.*?>', '', comment)
    # Remove non-alphanumeric characters (punctuation and symbols)
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)
    # Convert to lowercase
    comment = comment.lower()
    # Tokenize
    tokens = word_tokenize(comment)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to a single string
    return ' '.join(tokens)

# Apply preprocessing to comments
def preprocess_comments(comments):
    return [preprocess_comment(comment) for comment in comments]

# Extract main topics using TF-IDF
def extract_main_topics(comments, num_topics=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names_out()
    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    topics = df.sum(axis=0).sort_values(ascending=False).head(num_topics)
    return topics

def main():
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_id = "t2kY2T49Gnk"  # Replace with your YouTube video ID

    # Get metadata
    print("Fetching video metadata...")
    metadata = get_video_metadata(youtube, video_id)
    if metadata:
        print("Video Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("Failed to fetch video metadata.")

    # Get comments
    print("\nFetching all comments...")
    comments = get_video_comments(youtube, video_id)
    print(f"Total comments fetched: {len(comments)}")

    # Preprocess comments
    print("\nPreprocessing comments...")
    preprocessed_comments = preprocess_comments(comments)

    # Analyze sentiment with VADER
    print("\nAnalyzing sentiments using VADER...")
    sentiments = analyze_sentiment_vader(comments)

    # Visualize sentiment analysis
    print("\nVisualizing sentiment analysis...")
    visualize_sentiments(sentiments, save_path="sentiment_analysis.png")

    # Analyze AI-related sentiment
    print("\nAnalyzing AI-related sentiment...")
    ai_sentiments = analyze_ai_sentiment(comments)
    print(f"AI-related sentiment: {ai_sentiments[:10]}...")  # Print first 10 AI-related sentiments

    # Extract hashtags
    print("\nExtracting hashtags from comments...")
    hashtags = extract_hashtags(comments)
    hashtag_counts = Counter(hashtags).most_common(10)
    print(f"Top 10 hashtags: {hashtag_counts}")

    # Extract main topics (e.g., AI, technology-related themes)
    print("\nExtracting main topics...")
    main_topics = extract_main_topics(preprocessed_comments)
    print(f"Main topics based on TF-IDF: {main_topics}")

    # Save results
    save_to_csv(
        [{"original_comment": comments[i], "preprocessed_comment": preprocessed_comments[i], "sentiment": sentiments[i]} for i in range(len(comments))],
        "detailed_comments_analysis.csv"
    )

# Run the program
if __name__ == "__main__":
    main()

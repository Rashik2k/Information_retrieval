from flask import Flask, request, jsonify, render_template
from requisite import load_publications, load_index, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer


app = Flask(__name__)

model = joblib.load('classifier.sav')
vectorizer = joblib.load('vectorizer.sav')


stemmer = PorterStemmer()
tokenizer = WhitespaceTokenizer()


def classify_text(character):
    character = re.sub('[^a-zA-Z]', ' ', character)
    l_character = character.lower()
    t_character = tokenizer.tokenize(l_character)
    s_character = ' '.join([stemmer.stem(w) for w in t_character])
    v_character = vectorizer.transform([s_character])
    classification = model.predict(v_character)
    return classification[0]


# Constants
DATA_FILE = "publications.json"
INDEX_FILE = "inverted_index.json"

def calculate_relevance(query, publications, inverted_index):
    tokens = preprocess_text(query)
    relevant_documents = set()

    for token in tokens:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])

    if not relevant_documents:
        return []


    documents = [publications[doc_id]["title"] for doc_id in relevant_documents]

    vectorizer = TfidfVectorizer()

    doc_vectors = vectorizer.fit_transform(documents)

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    ranked_docs = [(doc_id, score) for doc_id, score in zip(relevant_documents, similarities)]

    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    return ranked_docs

# Flask routes
@app.route("/")
def home():
    return render_template("search.html")


@app.route("/classification")
def classification():
    return render_template("classification.html")


@app.route("/classify", methods=["POST"])
def classify():
    input_text = request.form.get("text")
    if not input_text:
        return jsonify({"error": "Text parameter is required"}), 400
    category = classify_text(input_text)
    if category == 'Business':
        prediction = 'The given text represents to category: Business.'
    elif category == 'Health':
        prediction = 'The given text represents to category: Health.'
    elif category == 'Politics':
        prediction = 'The given text represents to category: Politics.'
    return jsonify({"category": prediction})



@app.route("/search", methods=["GET"])
def search():
    """Handle search queries and return results."""
    query = request.args.get("query")
    print()
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    publications = load_publications()
    inverted_index = load_index()

    if not publications or not inverted_index:
        return jsonify({"error": "No data or index found. Please run the crawler and build the index first."}), 400

    ranked_docs = calculate_relevance(query, publications, inverted_index)

    results = []
    for doc_id, score in ranked_docs:
        pub = publications[doc_id]
        results.append({
            "title": pub["title"],
            "authors": pub["authors"],
            "publication_year": pub["publication_year"],
            "journal": pub.get("journal", "N/A"),
            "volume": pub.get("volume", "N/A"),
            "link": pub["link"],
            "relevance_score": score,
            "author_profiles": [{"name": author["name"], "link": author["link"]} for author in pub["authors"]]
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
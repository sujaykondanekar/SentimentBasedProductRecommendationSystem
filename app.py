from flask import Flask, request
from flask_cors import CORS

from model import SentimentBasedRecommender

app = Flask(__name__)
CORS(app, origins=["*"], headers=['*'], methods=['*'])  # Enable CORS

sentiment_based_recommender = SentimentBasedRecommender('pickle/xgb_model_tuned_without_gpu.pkl',
                                                        'pickle/tfidfVectorizer.pkl',
                                                        'pickle/user-recommendation.pkl',
                                                        'pickle/processed_data.pkl')


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get inputs from query parameters
    user_id = request.args.get('user_id')
    num_recommendations = int(request.args.get('num_recommendations', 5))

    # Perform sentiment-based recommendations
    return sentiment_based_recommender.get_recommendations(user_id, num_recommendations)


@app.route('/')
def serve_html():
    with open('index.html', 'r') as file:
        html_content = file.read()
    return html_content


if __name__ == '__main__':
    app.run(debug=True)

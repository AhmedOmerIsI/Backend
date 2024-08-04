import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.tokenize import word_tokenize
from collections import defaultdict

app = Flask(__name__)
CORS(app, resources={r"/predict_similarity": {"origins": "*"}})
CORS(app, resources={r"/get_feature_recommendation": {"origins": "*"}})
CORS(app, resources={r"/get_organization_recommendation": {"origins": "*"}})
CORS(app, resources={r"/validate_meaningful_input": {"origins": "*"}})

# Load datasets
ai_df = pd.read_excel(r'C:\AI.xlsx')
ses_df = pd.read_excel(r'C:\SESGrants.xlsx')

# Preprocess AI dataset
ai_df['combined_text'] = ai_df['Project Title'] + ' ' + ai_df['Objectives'] + ' ' + ai_df['Outcomes (Quantifiable)']
ai_df['combined_text'] = ai_df['combined_text'].fillna('')
ai_df['combined_text'] = ai_df['combined_text'].astype(str)

# Preprocess SES dataset
ses_df['combined_text'] = ses_df['Project Title'] + ' ' + ses_df['Objectives'] + ' ' + ses_df['Outcomes']
ses_df['combined_text'] = ses_df['combined_text'].fillna('')
ses_df['combined_text'] = ses_df['combined_text'].astype(str)

# BERT model setup
bert_model_name_or_path = 'bert-base-uncased'
bert_model = SentenceTransformer(bert_model_name_or_path)

# Train an N-Gram Language Model
n = 3  # Choose an appropriate value for n
tokenized_text = [nltk.word_tokenize(text.lower()) for text in ai_df['combined_text']]
padded_ngrams = [list(pad_both_ends(tokens, n)) for tokens in tokenized_text]

# Create a defaultdict to count the frequency of each n-gram
language_model = defaultdict(int)
for tokens in padded_ngrams:
    for ngram in nltk.ngrams(tokens, n):
        language_model[ngram] += 1


# Function to check meaningfulness of text
def is_meaningful(text, ngrams, threshold):
    tokens = word_tokenize(text.lower())
    ngrams_list = list(nltk.ngrams(tokens, n))
    ngram_count = sum(language_model[ngram] for ngram in ngrams_list)
    print(f"ngram_count: {ngram_count}, threshold: {threshold}")
    return ngram_count >= threshold


# Function to calculate BERT embeddings
def get_bert_embeddings(text):
    return bert_model.encode(text)


# Define threshold
threshold = 10  # Adjust this value according to your needs


@app.route('/predict_similarity', methods=['POST'])
def predict_similarity():
    try:
        # User input
        user_data = request.json
        user_combined_text = ' '.join([user_data['title'], user_data['objectives'], user_data['outcomes']])

        # Check meaningfulness
        if not is_meaningful(user_combined_text, n, threshold):
            return jsonify({'error': 'Input is not meaningful.'}), 400

        user_embedding = get_bert_embeddings([user_combined_text])

        # Calculate cosine similarity
        ai_embeddings = get_bert_embeddings(ai_df['combined_text'].values)
        ses_embeddings = get_bert_embeddings(ses_df['combined_text'].values)

        ai_similarities = cosine_similarity(user_embedding, ai_embeddings)[0]
        ses_similarities = cosine_similarity(user_embedding, ses_embeddings)[0]

        ai_max_similar_index = ai_similarities.argmax()
        ses_max_similar_index = ses_similarities.argmax()

        ai_similarity_score = round(ai_similarities[ai_max_similar_index] * 100)
        ses_similarity_score = round(ses_similarities[ses_max_similar_index] * 100)

        response = {
            'ai_similarity_score': ai_similarity_score,
            'ses_similarity_score': ses_similarity_score,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/validate_meaningful_input', methods=['POST'])
def validate_meaningful_input():
    try:
        data = request.json
        user_combined_text = ' '.join([data['title'], data['objectives'], data['outcomes']])

        # Check meaningfulness
        if not is_meaningful(user_combined_text, n, threshold):
            return jsonify({'error': 'Input is not meaningful.'}), 400

        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/get_feature_recommendation', methods=['POST'])
def get_feature_recommendation():
    try:
        # User input
        user_data = request.json
        user_combined_text = user_data['title'] + ' ' + user_data['objectives'] + ' ' + user_data['outcomes']
        # Check meaningfulness
        if not is_meaningful(user_combined_text, n, threshold):
            return jsonify({'error': 'Input is not meaningful.'}), 400

        user_embedding = bert_model.encode([user_combined_text])
        ai_embeddings = bert_model.encode(ai_df['combined_text'].values)

        user_embedding = get_bert_embeddings([user_combined_text])

        # Calculate cosine similarity
        ai_similarities = cosine_similarity(user_embedding, ai_embeddings)[0]

        ai_max_similar_index = ai_similarities.argmax()

        #         # Feature recommendation
        ai_feature_recommendation = ai_df.iloc[ai_max_similar_index]['Objectives']

        response = {
        'ai_feature_recommendation': ai_feature_recommendation
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/get_organization_recommendation', methods=['POST'])
def get_organization_recommendation():
    try:
        # User input
        user_data = request.json
        user_combined_text = ' '.join([user_data['title'], user_data['objectives'], user_data['outcomes']])

        # Check meaningfulness
        if not is_meaningful(user_combined_text, n, threshold):
            return jsonify({'error': 'Input is not meaningful.'}), 400

        user_embedding = get_bert_embeddings([user_combined_text])

        # Calculate cosine similarity for SES dataset
        ai_embeddings = get_bert_embeddings(ai_df['combined_text'].values)
        ses_embeddings = get_bert_embeddings(ses_df['combined_text'].values)

        ses_similarities = cosine_similarity(user_embedding, ses_embeddings)[0]
        ai_similarities = cosine_similarity(user_embedding, ai_embeddings)[0]

        ses_similarity_score = round(ses_similarities[ses_similarities.argmax()] * 100)
        ai_similarity_score = round(ai_similarities[ai_similarities.argmax()] * 100)

        response = {
            'ses_similarity_score': ses_similarity_score,
            'ai_similarity_score': ai_similarity_score,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    except Exception as e:
        print("An error occurred while starting the Flask server:")
        traceback.print_exc()













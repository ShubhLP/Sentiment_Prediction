from flask import Flask, request, jsonify, render_template
import joblib

# Load the models and vectorizer
sentiment_model = joblib.load('sentiment_model.pkl')
emotion_model = joblib.load('emotion_detection_model.pkl')
vectorizer = joblib.load('tfdif_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create the Flask app
app = Flask(__name__)

# Function to preprocess user input
def preprocess_text(text):
    # Add your preprocessing logic here (e.g., lowercasing, removing punctuation, etc.)
    return text.lower()  # Example: simple lowercase conversion

# Route for the chatbot UI (HTML page)
@app.route('/')
def home():
    return render_template('index.html')  # Assuming you have a basic index.html file

# Route to handle chatbot input and provide a response
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']  # Get the user input from the form
    
    # Preprocess the input text
    user_input_processed = preprocess_text(user_input)
    
    # Vectorize the input for model prediction
    user_input_vectorized = vectorizer.transform([user_input_processed])
    
    # Predict the sentiment
    sentiment_prediction = sentiment_model.predict(user_input_vectorized)
    sentiment_label = 'positive' if sentiment_prediction == 1 else 'negative'
    
    # Predict the emotion
    emotion_prediction = emotion_model.predict(user_input_vectorized)
    emotion_label = label_encoder.inverse_transform(emotion_prediction)[0]
    
    # Generate the chatbot response
    response = f"Based on your input, I detected a {sentiment_label} sentiment and a {emotion_label} emotion."
    
    # Return the response as JSON
    return jsonify({'response': response})

# Run the app
if __name__ == '__main__':
    app.run(port=5001, debug=False)
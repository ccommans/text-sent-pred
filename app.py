import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text']
    
    # Retrieve vectorizer
    with open('./models/vectorizer.pkl', 'rb') as v:
        vectorizer = pickle.load(v)
    test_vector = vectorizer.transform([text_input])

    # Make prediction using model
    with open('./models/sentiment_classifier.pkl', 'rb') as m:
        model = pickle.load(m) # Load the pre-trained model

    result = model.predict(test_vector)
    return jsonify(prediction=result.tolist())

if __name__ == '__main__':
     app.run()

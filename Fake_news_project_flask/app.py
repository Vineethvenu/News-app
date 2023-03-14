from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# function to run for prediction
def fakenews(var):
    # retrieving the best model for prediction call
    with open('model.pkl', 'rb') as f:
        load_model = pickle.load(f)
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return prediction[0], prob[0][1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    var = request.form['text']
    prediction, prob = fakenews(var)
    return render_template('index.html', prediction=prediction, prob=prob)

if __name__ == '__main__':
    app.run(debug=True)

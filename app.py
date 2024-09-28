from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/heart_health')
def heart_health():
    return render_template('forest.html')

@app.route('/mental_health')
def mental_health():
    return render_template('mental.html')

@app.route('/meditate')
def meditate():
    return render_template('meditate.html')

@app.route('/sleep')
def sleep():
    return render_template('sleep.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        print(request.form)

        int_features = []
        for x in request.form.values():
            if x == '':
                int_features.append(0)
            else:
                try:
                    int_features.append(int(x))
                except ValueError:
                    return render_template('forest.html', pred='Invalid input! Please enter valid numeric values.')

        if len(int_features) != 5:
            return render_template('forest.html', pred='Invalid input! Please enter exactly 5 numeric values.')

        final = np.array([int_features]).reshape(1, -1)
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)

        if float(output) > 0.5:
            return render_template('forest.html', pred=f'Your heart is in Danger.\nProbability of heart failing is {output}')
        else:
            return render_template('forest.html', pred='Your heart is safe.')

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('forest.html', pred=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)





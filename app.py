from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the model once at startup (faster than reloading each request)
loaded_model = joblib.load('home_price.pkl')

def predict_home_price(house):
    x_test = np.array([house])  # make it 2D
    predictions = loaded_model.predict(x_test)
    return predictions[0]

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home_price():
    prediction = None

    if request.method == "POST":
        try:
            area = float(request.form.get("area", 0))
            bed = int(request.form.get("bed", 0))
            bath = int(request.form.get("bath", 0))
            parking = int(request.form.get("parking", 0))

            # Pass values to prediction
            prediction = predict_home_price([area, bed, bath, parking])
            prediction = np.round(prediction, 2)  # round nicely
        except ValueError as e:
            prediction = f"Invalid input: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

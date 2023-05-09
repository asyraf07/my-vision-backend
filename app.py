from flask_api import FlaskApi
from endpoint import home, predict

app = FlaskApi()


app.add_endpoint('/', 'home', home, methods=["GET"])
app.add_endpoint('/predict', 'predict', predict, methods=["POST"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return "hei"
    # return {
    #     "status": 200,
    #     "message": "hello"
    # }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

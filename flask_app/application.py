from flask import Flask, render_template, request
#import ml_models
import nmf_model

app = Flask(__name__)
# instantiating a Flask application

#"/" will be 'index'
# @app.route("/")
# def index():    
#     return render_template("index.html")

@app.route("/")
def index():    
    return render_template("index.html")

@app.route("/recommender")
def recommender():
    user_input = dict(request.args)
    user_input = {int(k):int(v) for k,v in user_input.items()}
    results = nmf_model.nmf_prediction(user_input)
    return render_template("recommendations.html", results = results, user_input = user_input)
    
@app.route("/credits")
def credits():    
    return render_template("credits.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

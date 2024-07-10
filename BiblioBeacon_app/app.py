from flask import Flask, render_template, request

# from recommender import get_recommendations

from KNN_recommender.recommendation import KNNRecommender_GridSearch
from SVD_recommender.recommendation import recommend_books
# from AE_recommender.recommendation import books_recommender

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/avec_identifiant')
def index_2():
    return render_template('index2.html')

# @app.route('/avec_identifiant_2')
# def index_2():
#     return render_template('index3.html')
    


@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    top_n = request.form['top_n']

    if top_n and top_n.isdigit():  # Check if top_n is provided and is a digit
        top_n = int(top_n)
        books = KNNRecommender_GridSearch(book_title,int(top_n+1))
        return render_template('recomm.html', books=books)
    else:
        books = KNNRecommender_GridSearch(book_title,6)
        return render_template('recomm.html', books=books)


@app.route("/_recommend", methods=["POST"])
def recommend_v2():
    uid = int(request.form["user_id"])
    top_n = request.form['top_n']

    if top_n and top_n.isdigit():
        top_n = int(top_n)
        books = recommend_books(uid,top_n)
        return render_template("recomm.html", books = books)
    else:
        books = recommend_books(uid)
        return render_template("recomm.html", books = books)
    


# @app.route("/__recommend", methods=["POST"])
# def recommend_v3():
#     uid = int(request.form["user_id"])
#     top_n = request.form['top_n']

#     if top_n and top_n.isdigit():
#         top_n = int(top_n)
#         books = books_recommender(uid,top_n)
#         return render_template("recomm.html", books = books)
#     else:
#         books = books_recommender(uid)
#         return render_template("recomm.html", books = books)


if __name__ == '__main__':
    app.run(debug=True, )
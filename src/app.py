from flask import Flask, render_template, request, redirect, url_for
# TODO update to use main
# import main as lib5
import indexer as indexer

app = Flask(__name__, static_folder='docs', static_url_path='/docs')

cacm_dataset = "../data/cacm/docs"
# Route to display the form
@app.route('/')
def index():
    # Set up database
    indexer.create_db()
    print("DATABASE CREATED")
    # Index dataset
    indexer.index_dir(cacm_dataset)
    print("DATASET INDEXED")


    return render_template('form.html')

# Route to process the form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    query = request.form.get('query')

    # Calling search on our database
    print("running search")
    ranks = indexer.search(query)

    # Ranking search results
    # print(f"     ranks: {ranks}")
    results = indexer.make_snippets(query,ranks)
    # print(f"     results: {results}")
    print("DONE search")

    print("rendering")
    return render_template('form.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

# TODO Fix Title, current title is file name and not document name
# TODO Fix title highlight, Currently not highlighting keywords
from flask import Flask, render_template, request, redirect, url_for
import main as lib5

app = Flask(__name__, static_folder='docs', static_url_path='/docs')

# Route to display the form
@app.route('/')
def index():
    return render_template('form.html')

# Route to process the form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    query = request.form.get('query')

    # Calling search on our database
    print("running search")
    ranks = lab5.search(query)

    # Ranking search results
    print(f"     ranks: {ranks}")
    results = lab5.make_snippets(query,ranks)
    print(f"     results: {results}")
    print("DONE search")

    print("rendering")
    return render_template('form.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
'''
indexer will index all the file ( txt, html, xml ) in a directory 
'''
import sqlite3
import os
from collections import defaultdict
import math
import nltk

# init stemmer and set stemmer language
snowBallStemmer = nltk.stem.SnowballStemmer("english",ignore_stopwords=True)

# Init tokenizer
tokenizer  = nltk.tokenize.word_tokenize

# list of stop words
stopwords = set(nltk.corpus.stopwords.words('english'))


def normalize_path(path):
    """Normalize path to use forward slashes regardless of OS when dealing with file paths
    
    Args:
        path: File path string
        
    Returns:
        Path with forward slashes
    """
    return path.replace('\\', '/')

def preprocess_text(text):
    """Preprocess text: lowercase, tokenize, and stem using Snowball.
    
    Args:
        text: Raw text string
        
    Returns:
        List of tokens (preprocessed terms)
    """
    text = text.lower()
    tokens = tokenizer(text)
    stemmed_tokens = [snowBallStemmer.stem(token,) for token in tokens]
    return stemmed_tokens


def create_db():
    """Create an empty database with necessary tables. Along with creating the necessary index's for increased query performance

    If the database exists, it will be reset
    """
    # Check if index.db exists remove it
    if os.path.exists('index.db'):
        os.remove('index.db')

    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()

    # CREATING inverted index TABLE
    # Finds which documents contain a given word
    # term - Given word used to find documents
    # document_id - The id of the document that contains the given word
    # term_freq - The frequency of the word with in the document
    # PRIMARY KEY is the term + document_id
    cursor.execute('''
        CREATE TABLE inverted_index (
            term TEXT NOT NULL,
            document_id INTEGER NOT NULL,
            term_freq INTEGER NOT NULL,
            PRIMARY KEY (term, document_id)
        )
    ''')

    # CREATING indexed files TABLE
    # Keeps track of which files have already been index
    # document_id - Unique document id
    # filename - The name of the file also must be unique
    cursor.execute('''
        CREATE TABLE indexed_files (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL
        )
    ''')

    # CREATING INDEXs
    # Creates a tree data structure index on the term column to speed up queries
    cursor.execute('CREATE INDEX idx_term ON inverted_index(term)')
    # Creating index on document id
    cursor.execute('CREATE INDEX idx_doc_id ON inverted_index(document_id)')

    # Committing changes and closing connection
    conn.commit()
    conn.close()

def index_file(filename):
    """Index the contents of a file.
    
    If the file has already been indexed, does nothing.
    
    Args:
        filename: Path to the file to index
    """
    normalized_filename = normalize_path(filename)

    # Init connection to database
    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()

    # Query table of indexed files to see if filename given has already been indexed
    cursor.execute('SELECT filename FROM indexed_files WHERE filename = ?', (normalized_filename,))

    if cursor.fetchone() is not None:
        # file has been index already
        conn.close()
        return
    
    # File has not been index ( not in indexed_files table ) & needs to be indexed
    try:
        with open(filename, 'r',encoding='utf-8',
                  errors='ignore') as file:
            # Open file and get text
            text = file.read()
    except Exception as e:
        conn.close()
        return
    # Get the preprocessed file text 
    stemmed_tokens = preprocess_text(text)
    # Dictionary to track term counts ( with a default value )
    term_counts = defaultdict(int)

    # Get term count for each term/token
    for token in stemmed_tokens:
        term_counts[token] += 1
    # Add file to inverted files
    cursor.execute('''
        INSERT INTO indexed_files (filename)
        VALUES(?)
''',(normalized_filename,))
    
    # Getting the newly created document id
    doc_id = cursor.lastrowid

    # Adding indexed file to inverted index table
    for term, count in term_counts.items():
        cursor.execute('''
            INSERT OR REPLACE INTO inverted_index (term, document_id,term_freq )
            VALUES(?, ?, ?)
            ''', (term,doc_id,count))

    conn.commit()
    conn.close()


def index_dir(directory):
    """Index all files in a directory.
    
    Args:
        directory: Path to the directory to index
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            index_file(filepath)

def find(terms):
    """Find documents containing the given preprocessed terms.
    
    Args:
        terms: List of preprocessed terms (tokens)
    
    Returns:
        Dict of form {term: {document: count}}
    """
    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()

    results = {}
    # Find all the documents that contain the given term
    for term in terms:
        cursor.execute('''
            SELECT inverted_index.document_id, inverted_index.term_freq, indexed_files.filename
            FROM inverted_index
            JOIN indexed_files 
                ON inverted_index.document_id = indexed_files.document_id
            WHERE inverted_index.term = ?
        ''', (term,))
    # 
        query_results = cursor.fetchall()
        term_results = {}
        for doc_id,term_freq, filename in query_results:
            term_results[doc_id] = {
                "term_freq": term_freq,
                "filename": filename
            }
        results[term] = term_results

    conn.close()
    return results

def search(query):
    """Search for documents matching the query string.
    
    Preprocesses the query, finds matching documents, and ranks by TF-IDF relevance.
    
    Args:
        query: Query string
    
    Returns:
        List of document paths sorted by relevance (descending)
    """
    # Stem/tokenize the input user query
    stemmed_query_tokens = preprocess_text(query)
    # Create a list of unique tokens
    unique_tokens = list(set(stemmed_query_tokens))

    if not unique_tokens:
        return []
    term_docs = find(unique_tokens)

    # Get total number of documents
    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM indexed_files')
    total_docs = cursor.fetchone()[0]
    conn.close()

    if total_docs == 0:
        return []

    # Compute IDF per term
    idf = {}
    for term in unique_tokens:
        docs_with_term = len(term_docs.get(term, {}))
        if docs_with_term > 0:
            idf[term] = math.log(total_docs / docs_with_term)
        else:
            idf[term] = 0

    # Score documents using TF * IDF
    doc_scores = defaultdict(float)
    doc_filenames = {}   # doc_id → filename

    for term in unique_tokens:
        postings = term_docs.get(term, {})

        for doc_id, info in postings.items():
            term_freq = info["term_freq"]
            filename = info["filename"]

            doc_scores[filename] += term_freq * idf[term]
            doc_filenames[filename] = filename  # keep track

    # Sort by descending score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))

    # Return only filenames (sorted)
    return [filename for filename, score in sorted_docs]

# -----
# RENDERING RESULTS
# -----

def make_snippers(query,ranks):
    keypairs = process_query(query)
    html =""

    # Loop through and build snippets for each document
    for doc in ranks:
        title = get_document_title(doc)
        desc = snippet(keypairs,doc)
        highlighted_title = highlighted_title(title,keypairs)
        html += f'<div class="snippet"> <div class="title">{highlighted_title}</div><div class="{doc}">{doc}</div><div class="description">{description}</div></div>'
    return html

def process_query(query):
    text = query.lower()
    tokens = tokenizer(text)
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
    keypairs = []

    for keyword in filtered_tokens:
        keypairs.append( (keyword,snowBallStemmer.stem(keyword,) ) )

    return keypairs

def snippet(keypairs, docname, max_length=250):
    scores = []
    tokenized_sentences = ""

    with open(docname) as f:
        doc_text = f.read()
        tokenized_sentences = nltk.tokenize.sent_tokenize(doc_text,language='english')
        f.close()
    # print(tokenized_sentences)
    for i, sentence in enumerate(tokenized_sentences):
        sentence_score, positions = score_sentence(keypairs, sentence)
        if sentence_score > 0:  # Only keep sentences with matches
            scores.append((sentence_score, i, sentence, positions))
   
    if not scores:
        print("NO matching sentences found")
        if tokenized_sentences:
            first_sentence = tokenized_sentences[0]
            if len(first_sentence) <= max_length:
                return first_sentence
            else:
                return first_sentence[:max_length] + ".."
        return "No prevews available"
    scores.sort(reverse=True, key=lambda x : x[0])
    # Select sentences that fit within max_length
    resulting_sentences = []
    total_length = 0
    
    for score, original_index, sentence, positions in scores:
        sentence_len = len(sentence)
        
        # Can we fit this sentence?
        if total_length + sentence_len <= max_length:
            resulting_sentences.append((original_index, sentence))
            total_length += sentence_len
        else:
            # If we haven't added any sentences yet, truncate this one
            if not resulting_sentences:
                truncated = sentence[:max_length] + "..."
                resulting_sentences.append((original_index, truncated))

            # Stop looking for more sentences
            break
        # Sort sentences back to their original order in the document
    resulting_sentences.sort(key=lambda x: x[0])
    snippet_parts = []
    
    for i, (index, sentence) in enumerate(resulting_sentences):
        snippet_parts.append(sentence)
        
        # Check if next sentence is consecutive
        if i < len(resulting_sentences) - 1:
            next_index = resulting_sentences[i + 1][0]
            if next_index != index + 1:  # Not consecutive
                snippet_parts.append("...")
    
    final_snippet = " ".join(snippet_parts)

    return final_snippet
    
def score_sentence(keywords, sentence):
    # print("=============================")
    # print("=======SCORING SENTENCE======")
    # print("=============================")
    sentence_l = sentence.lower()
    sentence_words = nltk.word_tokenize(sentence_l)
    sentence_stems = [stemmer.stem(word) for word in sentence_words]

    # print(f"{sentence_l}")
    score = 0
    keyword_positions = []
    for word,stem in keywords:
        # print(f"Looking for: {word} OR stem: {stem}")
        match = None
        try:
            pos = sentence_words.index(word)
            score += 10
            match = word
            # print(f"FOUND exact match at position {pos}")
        except ValueError:
            # exact word was not found try stemmed version
            try:
                pos = sentence_words.index(stem)
                score += 5
                match = stem
                # print(f"FOUND stem match at position {pos}")
            except ValueError:
                # print(f"NO match found")
                pass
        if match:
            search_list = sentence_words if match == word else sentence_stems
            start = 0
            while True:
                try:
                    pos = search_list.index(match, start)
                    keyword_positions.append(pos)
                    start = pos +1
                except ValueError:
                    break
        if len(keyword_positions) >1:
            keyword_positions.sort()


            for i in range(len(keyword_positions) -1):
                distance = keyword_positions[i+1] - keyword_positions[i]
    
                if distance < 50:
                    score +=3

    return score, keyword_positions
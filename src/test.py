import math
import os
import subprocess
import importlib

import indexer as hw3

hw3.create_db()

hw3.index_file('tests/animals.txt')
hw3.index_dir('movie_reviews')

results = hw3.find(["cat","dog", "fish", "badger"])

results_1 = hw3.search("genius comedy")
print(results_1)
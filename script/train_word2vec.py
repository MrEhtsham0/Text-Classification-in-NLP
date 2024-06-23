# train_word2vec.py

from sklearn.decomposition import PCA
from preprocess import preprocess_books
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import plotly.express as px


# Define the path to your dataset
PATH = "//home//mr_ehtsham//Pictures//Text-Classification-in-NLP//dataset"

# Preprocess the books
books = preprocess_books(PATH)

# Train the Word2Vec model
model = Word2Vec(
    books,
    vector_size=100,
    window=10,
    min_count=2,
    workers=4
)

# Build vocabulary and train the model
model.build_vocab(books)
model.train(books, total_examples=model.corpus_count, epochs=model.epochs)

# Find most similar words to 'daenerys'
similar_words = model.wv.most_similar("daenerys")
print("Most similar words to 'daenerys':", similar_words)

# Perform PCA for visualization
pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors())
y = model.wv.index_to_key

# Visualize the words in 2D space
fig = px.scatter_3d(X[:100], x=0, y=1, z=2, color=y[:100])
fig.show()

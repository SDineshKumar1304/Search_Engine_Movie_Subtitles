import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def load_models():
    count_vectorizer = joblib.load('C:\\Users\\svani\\Search_Engine\\count_vectorizer.joblib')
    tfidf_transformer = joblib.load('C:\\Users\\svani\\Search_Engine\\tfidf_transformer.joblib')
    tfidf_matrix = joblib.load('C:\\Users\\svani\\Search_Engine\\tfidf_matrix.joblib')
    return count_vectorizer, tfidf_transformer, tfidf_matrix

def retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data, top_n=5):
    query_vector = count_vectorizer.transform([query])
    query_tfidf = tfidf_transformer.transform(query_vector)
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1]
    retrieved_documents = [data['clean_file_content'][idx] for idx in top_indices[:top_n]]
    
    return retrieved_documents

def main():
    st.title('Search Engine System')

    data = load_data('C:\\Users\\svani\\Search_Engine\\Data.csv')

    count_vectorizer, tfidf_transformer, tfidf_matrix = load_models()

    query = st.text_input('Enter your query:', '')

    if st.button('Search'):
        if query:
            retrieved_documents = retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data)
            st.subheader('Top 5 documents similar to the query:')
            for i, doc in enumerate(retrieved_documents, 1):
                st.write(f"Document {i}: {doc}")

if __name__ == '__main__':
    main()

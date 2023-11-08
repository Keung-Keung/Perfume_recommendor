import streamlit as st
import pandas as pd
import openai
import re
import nltk
import pickle
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def main():
    st.title('여행지 기반 향수 추천 시스템')

    # 사용자 입력 받기
    query = st.text_input('여행지를 입력해주세요:')

    if st.button('향수 추천 받기'):
        df = chat_gpt(query)
        # 전처리 함수 적용
        df['input_pre'] = df['input'].apply(preprocess_text)
        st.text(df['input'].iloc[0])
        vectorizer, word2vec_model, kmeans_model, top3_perfumes_df = load_assets(df)  # load_assets는 df를 인자로 받아야 합니다.

        # Get top 3 perfumes for the query
        st.write(top3_perfumes_df)

def chat_gpt(query):
    if not hasattr(chat_gpt, 'cnt'):
        chat_gpt.cnt = -1
    if not hasattr(chat_gpt, 'df'):
        chat_gpt.df = pd.DataFrame(columns=['input'])

    chat_gpt.cnt += 1

    # 이부분에서 openai.api_key는 안전한 위치에 저장하거나 환경변수에서 로드하도록 바꾸어야 합니다.
    # openai.api_key는 이 코드에서 직접 보여주면 안됩니다.
    openai.api_key = 'OPENAI-YOUR-KEY'

    # query를 custom_prompt에 포함하여 실제 여행지를 질의로 사용합니다.
    custom_prompt = f"""Please provide an overview of {query} as a travel destination, touching upon its cultural, historical, and natural highlights. From the breathtaking views atop its landmarks to the bustling markets filled with the aroma of street foods, and the serene countryside with its fresh, floral air, capture the essence of {query} in detail. Alongside this, focus on the sensory experiences - the distinct smells that define the local atmosphere, such as the zesty tang of citrus in summer or the smoky warmth of chestnut vendors in the fall."""

    response = openai.ChatCompletion.create(
        model='gpt-4-0314',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant providing information about tourist attractions.'},
            {'role': 'user', 'content': custom_prompt}
        ],
    )

    answer = response.choices[0].message.content

    # 데이터 프레임에 응답 추가
    chat_gpt.df.loc[chat_gpt.cnt] = {'input': query, 'answer': answer}  # DataFrame에 새로운 행을 추가

    return chat_gpt.df

#불용어 전처리
# nltk.download('stopwords')
# nltk.download('punkt')

# 어간 추출
def stem_words(text):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

# 소문자 변환
def make_lower_case(text):
    return " ".join([word.lower() for word in text.split()])

# 불용어 제거
def remove_stop_words(text):
    stops = set(stopwords.words("english"))
    text = [w for w in text.split() if not w in stops]
    text = " ".join(text)
    return text

# 따옴표 제거
def remove_quotes(text):
    if isinstance(text, str):
        return text.replace('"', "")
    elif isinstance(text, list):
        return ["".join(t).replace('"', "") for t in text]
    else:
        return text

# 쉼표 제거
def remove_commas(text):
    if isinstance(text, list):
        return [t.rstrip(',') for t in text]
    else:
        return text

#특수문자 및 숫자 제거
def remove_special_characters(text):
    text = re.sub(r'[:.]|\d', '', text)
    return text

def preprocess_text(text):
    text = remove_special_characters(text)
    text = make_lower_case(text)
    text = remove_stop_words(text)
    text = stem_words(text.split()) # split()을 사용하여 단어 단위로 분리한 후 어간 추출
    text = remove_quotes(text)
    text = remove_commas(text)
    return text


def load_assets(df):
    # Load TF-IDF Vectorizer
    with open('./Data/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # 'input_pre' 컬럼으로부터 문장 리스트 생성
    sentences = df['input_pre'].tolist()

    # TF-IDF 벡터화
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # 단어별 TF-IDF 가중치 계산
    word2weight = {word: tfidf_matrix.getcol(idx).sum() for word, idx in vectorizer.vocabulary_.items()}

    #top10 가중치 확인
    top_10_weights = sorted(word2weight.items(), key=lambda x: x[1], reverse=True)[:10]

    for word, weight in top_10_weights:
        print(f'단어: {word}, 가중치: {weight}')

    # Load Word2Vec model
    with open('./Data/word2vec_model.pkl', 'rb') as f:
        word2vec_model = pickle.load(f)

    # 단어 임베딩 추출
    word_vectors = word2vec_model.wv

    # TF-IDF 가중치 적용하여 임베딩 벡터 추출
    X = []
    for sentence in sentences:
        embedding = []
        for word in sentence:
            if word in word_vectors.key_to_index:  # 단어가 Word2Vec 모델에 있는지 확인
                # TF-IDF 가중치 적용하여 단어 임베딩 계산 (단어가 없으면 기본값 1.0 사용)
                weighted_embedding = word_vectors.get_vector(word) * word2weight.get(word, 1.0)
                embedding.append(weighted_embedding)
        # 각 문장에 대해 단어 임베딩의 평균 계산 (단어가 하나도 없는 경우 제외)
        if embedding:
            average_embedding = np.mean(embedding, axis=0)
            X.append(average_embedding)
    print(X)

    # Load KMeans model
    with open('./Data/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    # 최적의 Cluster 예측
    cluster = kmeans_model.predict(X)
    print(cluster)

    # Load Perfume Embeddings
    with open('./Data/embedding_result_perfume.pkl', 'rb') as f:
        perfume_embeddings = pickle.load(f)
    top3_perfumes = []

    # Load Perfume Data
    perfume = pd.read_csv('./Data/Result_Clustering.csv', index_col=0)
    perfume['Embedding'] = perfume_embeddings  # Assumes embeddings are aligned with perfume_df

    for i, label in enumerate(cluster):
        # 동일한 클러스터 향수 추출
        same_cluster_perfumes = perfume[perfume['Cluster'] == label]

        # 동일한 클러스터 향수의 임베딩 추출
        same_cluster_perfume_embeddings = same_cluster_perfumes['Embedding'].apply(pd.Series).values

        # 입력 데이터(여행지) 임베딩벡터와 동일한 클러스터를 가진 향수 임베딩 벡터간의 코사인 유사도 계산
        cosine_similarities = cosine_similarity(X[i].reshape(1, -1), same_cluster_perfume_embeddings)

        # 유사도가 가장 높은 Top3 향수 선택
        top3_indices = cosine_similarities[0].argsort()[-3:][::-1]
        top3_perfumes.append(same_cluster_perfumes.iloc[top3_indices])

    # 리스트 -> 데이터프레임
    top3_perfumes_df = pd.concat(top3_perfumes, ignore_index=True)

    # 불필요한 피처 제거
    del top3_perfumes_df['Cluster']
    del top3_perfumes_df['Embedding']

    # 최종 Top 3출력

    return vectorizer, word2vec_model, kmeans_model, top3_perfumes_df



# Main execution logic
if __name__ == "__main__":
    main()

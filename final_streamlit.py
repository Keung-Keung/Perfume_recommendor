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

def chat_gpt(query):
    if not hasattr(chat_gpt, 'cnt'):
        chat_gpt.cnt = -1
    if not hasattr(chat_gpt, 'df'):
        chat_gpt.df = pd.DataFrame(columns=['input'])

    chat_gpt.cnt += 1

    # openai.api_key는 이 코드에서 직접 보여주면 안됩니다.
    openai.api_key = 'Your-API-KEY'

    # query를 custom_prompt에 포함하여 실제 여행지를 질의로 사용합니다.
    custom_prompt = f"""Please provide an overview of {query} as a travel destination,
         focusing on its most iconic and distinctive features. 
         Highlight the unique aspects that make {query} a remarkable place to visit, 
         such as famous landmarks, local products, or natural wonders. 
         Describe the sensory experiences associated with these features, like the smells and tastes 
         that a visitor might encounter, which capture the essence of {query}. 
         From the signature cuisine to the natural scenery that leaves a lasting impression, 
         detail the elements that define the local atmosphere of {query}.
         For instance, if the destination is known for its lavender fields, 
         mention the fragrance of lavender in the air and how it's a part of the local charm. 
         Include any other olfactory experiences that are synonymous with the place, 
         which could be used to identify perfumes that embody the essence of the destination."""

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

@st.cache_data
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

#streamlit 구현
def main():
    st.title('여행지 기반 향수 추천 시스템')
    st.write('여행지를 입력하면 해당 여행지와 어울리는 향수를 추천해 드립니다.')
    # 사용자 입력 받기
    query = st.text_input('여행지를 입력해주세요:')

    if st.button('향수 추천 받기'):
        with st.spinner("Loading..."):
            df = chat_gpt(query)
        df['input_pre'] = df['input'].apply(preprocess_text)
        st.text(df['input'].iloc[0])
        vectorizer, word2vec_model, kmeans_model, top3_perfumes_df = load_assets(df)

        # NaN 값을 '-'로 대체
        top3_perfumes_df.fillna('-', inplace=True)
        # Get top 3 perfumes for the query
        display_perfume_cards(top3_perfumes_df)

#향수 정보를 카드 형태로 표시
def display_perfume_cards(df):
    # 각 향수에 대한 카드를 생성하여 출력합니다.
    for index, row in df.iterrows():
        # displaying card 코드
        cols = st.columns([2, 1, 5, 1])  # 5열, 컬럼의 너비 설정
        # with cols[1]:
        # st.image(row['Image_Link'], width=100)  # 이미지 링크 사용
        with cols[0]:
            st.subheader(row['perfume'])  # 향수 이름
            # st.caption(row['brand'])  # 향수 브랜드
        with cols[2]:
            st.text(f"Top Note: {row['top notes']}")  # 탑노트
            st.text(f"Middle Note: {row['middle notes']}")  # 미들노트
            st.text(f"Base Note: {row['base notes']}")  # 베이스노트
        st.markdown("---")  # 구분선 추가

if __name__ == "__main__":
    main()

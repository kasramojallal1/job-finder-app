import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


def main():

    people = []
    submit_flag = False

    dataset = st.selectbox("Choose Dataset", ["free-lancer", "jd-job"])

    st.title("Team Information")

    col1, col2, col3 = st.columns(3)
    skills = col1.text_input("Skills", "")
    experience = col2.selectbox("Experience", ["Entry Level", "Mid Level", "Senior Level"])
    salary = col3.text_input("Salary", "")
    col1, col2, col3 = st.columns(3)
    skills2 = col1.text_input("Skills2", "")
    experience2 = col2.selectbox("Experience2", ["Entry Level", "Mid Level", "Senior Level"])
    salary2 = col3.text_input("Salary2", "")

    if st.button("Submit"):
        submit_flag = True
        people.append([skills, experience, salary])
        if skills2:
            people.append([skills2, experience2, salary2])

        if dataset == "free-lancer":
            df = pd.read_csv('./datasets/freelance-projects.csv')
        elif dataset == "jd-job":
            df = pd.read_csv('./datasets/jd_structured_data.csv')


    if len(people) == 1:
        pass
    elif len(people) > 1:
        pass

    if (submit_flag == True and dataset == "free-lancer"):
        df = df.drop('Date Posted', axis=1)
        df = df.filter(regex=r'^(?!.*Client).*$', axis=1)

        experience_mapping = {'Entry ($)': 1, 'Intermediate ($$)': 2, 'Expert ($$$)': 3}
        df['Experience'] = df['Experience'].map(experience_mapping)

        st.dataframe(df.head(3))

    if (submit_flag == True and dataset == "jd-job"):
        st.write('asdf')

    

    


if __name__ == "__main__":
    main()

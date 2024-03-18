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

    skills = []
    experience = []
    salary = []

    dataset = st.selectbox("Choose Dataset", ["free-lancer", "jd-job"])

    st.title("Team Information")

    col1, col2, col3 = st.columns(3)
    skills1 = col1.text_input("Skills", "")
    experience1 = col2.selectbox("Experience", ["Entry Level", "Mid Level", "Senior Level"])
    salary1_min, salary1_max = col3.columns(2)
    salary1_min = salary1_min.number_input("Salary1 Min", min_value=0, value=10, step=1)
    salary1_max = salary1_max.number_input("Salary1 Max", min_value=0, value=100, step=1)

    col1, col2, col3 = st.columns(3)
    skills2 = col1.text_input("Skills2", "")
    experience2 = col2.selectbox("Experience2", ["Entry Level", "Mid Level", "Senior Level"])
    salary2_min, salary2_max = col3.columns(2)
    salary2_min = salary2_min.number_input("Salary2 Min", min_value=0, value=10, step=1)
    salary2_max = salary2_max.number_input("Salary2 Max", min_value=0, value=100, step=1)

    if st.button("Submit"):
        submit_flag = True
        people.append([skills1, experience1, salary1_min, salary1_max])
        if skills2:
            people.append([skills2, experience2, salary2_min, salary2_max])

        if dataset == "free-lancer":
            df = pd.read_csv('./datasets/freelance-projects.csv')
        elif dataset == "jd-job":
            df = pd.read_csv('./datasets/jd_structured_data.csv')


    if len(people) == 1:
        skills = [skill.strip() for skill in skills1.split(',')]
        experience = experience1
        salary = [salary1_min, salary1_max]

        st.write('Skills:', skills)
        st.write('Experience:', experience)
        st.write('Salary:', salary)
        
    elif len(people) > 1:
        skills = [skill.strip() for skill in skills1.split(',')]
        skills.extend([skill.strip() for skill in skills2.split(',')])
        experience = max(experience1, experience2)
        salary = [salary1_min + salary2_min, salary1_max + salary2_max]

        st.write('Skills:', skills)
        st.write('Experience:', experience)
        st.write('Salary:', salary)

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

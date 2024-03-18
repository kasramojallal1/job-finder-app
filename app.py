import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


def clean_dataset_freelancer(df):
    df = df.drop(['Date Posted', 'Freelancer Preferred From'], axis=1)
    df = df.filter(regex=r'^(?!.*Client).*$', axis=1)

    experience_mapping = {'Entry ($)': 1, 'Intermediate ($$)': 2, 'Expert ($$$)': 3}
    df['Experience'] = df['Experience'].map(experience_mapping)

    return df

def clean_dataset_jd(df):
    pass

def process_text(df, text_feature, input):
    stop_words = set(stopwords.words('english'))
    input = [word.lower() for word in input if word.lower() not in stop_words]

    processed_text = [" ".join([word.lower() for word in word_tokenize(desc) if word.lower() not in stop_words]) for desc in df[text_feature]]

    vectorizer = TfidfVectorizer()
    job_vectors = vectorizer.fit_transform(processed_text)
    input_vector = vectorizer.transform([" ".join(input)])

    return job_vectors, input_vector


def apply_filters(df, experience, salary):
    df = df[(df['Budget'] >= salary[0]) & (df['Budget'] <= salary[1])]
    df = df[df['Experience'] <= experience]
    return df

def map_experience(exp):
    if exp == "Entry Level":
        return 1
    elif exp == "Mid Level":
        return 2
    elif exp == "Senior Level":
        return 3
    


def main():

    people = []
    submit_flag = False

    skills = []
    experience = None
    salary = None

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
        experience = map_experience(experience1)
        salary = [salary1_min, salary1_max]
        
    elif len(people) > 1:
        skills = [skill.strip() for skill in skills1.split(',')]
        skills.extend([skill.strip() for skill in skills2.split(',')])
        temp_exp1 = map_experience(experience1)
        temp_exp2 = map_experience(experience2)
        experience = max(temp_exp1, temp_exp2)
        salary = [salary1_min + salary2_min, salary1_max + salary2_max]


    if (submit_flag == True and dataset == "free-lancer"):
        df = clean_dataset_freelancer(df)
        job_vectors, input_vector = process_text(df, 'Description', skills)

        similarities = cosine_similarity(input_vector, job_vectors)
        top_indices = np.argsort(similarities[0])[-10:][::-1]
        top_jobs = df.loc[top_indices]
        st.dataframe(top_jobs)

        filtered_jobs = apply_filters(top_jobs, experience, salary)
        st.dataframe(filtered_jobs)
        
        def highlight_same_records(row):
            if row.name in filtered_jobs.index:
                return ['background-color: green'] * len(row)
            else:
                return [''] * len(row)

        styled_df = top_jobs.style.apply(highlight_same_records, axis=1)
        st.dataframe(styled_df)





    if (submit_flag == True and dataset == "jd-job"):
        st.write('asdf')

    

    


if __name__ == "__main__":
    main()

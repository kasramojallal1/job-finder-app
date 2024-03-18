import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


def clean_dataset_freelancer(df):
    df = df.drop(['Date Posted', 'Freelancer Preferred From', 'Duration'], axis=1)
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
    submit_flag = False

    cols = st.columns(2)
    dataset = cols[0].radio("Choose Dataset", ["free-lancer", "jd-job"])
    with cols[1]:
        history_placeholder = st.container()
    

    st.title("Team Information")

    if "history_list" not in st.session_state.keys():
        st.session_state.history_list = list()

    with st.form("my_form"):
        col1, col2, col3 = st.columns(3)
        skills = col1.text_input("Skills", "")
        experience = col2.selectbox("Experience", ["Entry Level", "Mid Level", "Senior Level"])
        salary_min, salary_max = col3.columns(2)
        salary_min = salary_min.number_input("Salary Min", min_value=0, value=10, step=1)
        salary_max = salary_max.number_input("Salary Max", min_value=0, value=100, step=1)
        
        if st.form_submit_button("Add"):
            st.session_state.history_list.append([skills, experience, [salary_min, salary_max]])

            with history_placeholder:
                history_placeholder.write(len(st.session_state.history_list))

    col1, col2, col3 = st.columns(3)
    type_value = col1.selectbox("Type", ["All", "Fixed", "Hourly"])

    category_list = ['All', 'Design', 'Video, Photo & Image', 'Business', 'Digital Marketing',
                    'Technology & Programming', 'Music & Audio', 'Social Media',
                    'Marketing, Branding & Sales' 'Writing & Translation']
    category_value = col2.selectbox("Category", category_list)

    number_of_entries = col3.number_input("Number of Entries", min_value=5, max_value=20, value=10, step=1)

    if st.button("Submit"):
        submit_flag = True

        if dataset == "free-lancer":
            df = pd.read_csv('./datasets/freelance-projects.csv')
        elif dataset == "jd-job":
            df = pd.read_csv('./datasets/jd_structured_data.csv')


        if len(st.session_state.history_list) == 1:
            skills, experience, salary = st.session_state.history_list[0]
            skills = [skill.strip() for skill in skills.split(',')]
            experience = map_experience(experience)
            salary = [salary[0], salary[1]]
        
        elif len(st.session_state.history_list) > 1:

            skill_list = []
            experience_list = []
            salary_list = []

            for value in st.session_state.history_list:
                skill_list.append(value[0])
                experience_list.append(value[1])
                salary_list.append(value[2])
            
            skills = [skill.strip() for skill in skill_list[0].split(',')]
            for i in range(1, len(skill_list)):
                skills.extend([skill.strip() for skill in skill_list[i].split(',')])

            for i in range(len(experience_list)):
                experience_list[i] = map_experience(experience_list[i])
            
            experience = max(experience_list)

            salary_min = 0
            salary_max = 0
            for value in salary_list:
                salary_min += value[0]
                salary_max += value[1]
            
            salary = [salary_min, salary_max]


    if (submit_flag == True and dataset == "free-lancer"):
        df = clean_dataset_freelancer(df)
        job_vectors, input_vector = process_text(df, 'Description', skills)

        similarities = cosine_similarity(input_vector, job_vectors)
        top_indices = np.argsort(similarities[0])[-number_of_entries:][::-1]
        top_jobs = df.loc[top_indices]
        # st.dataframe(top_jobs)

        filtered_jobs = apply_filters(top_jobs, experience, salary)
        # st.dataframe(filtered_jobs)
        
        def highlight_same_records(row):
            if row.name in filtered_jobs.index:
                return ['background-color: green'] * len(row)
            else:
                return [''] * len(row)

        styled_df = top_jobs.style.apply(highlight_same_records, axis=1)
        st.dataframe(styled_df)





    if (submit_flag == True and dataset == "jd-job"):
        pass

    

    


if __name__ == "__main__":
    main()

import streamlit as st

def main():
    if "history_list" not in st.session_state.keys():
        st.session_state.history_list = list()

    cols = st.columns(2)
    with cols[0]:
        current_placeholder = st.container()
    with cols[1]:
        history_placeholder = st.container()

    with st.form("my_form"):
        col1, col2, col3 = st.columns(3)
        skills = col1.text_input("Skills", "")
        experience = col2.selectbox("Experience", ["Entry Level", "Mid Level", "Senior Level"])
        salary_min, salary1_max = col3.columns(2)
        salary_min = salary_min.number_input("Salary Min", min_value=0, value=10, step=1)
        salary1_max = salary1_max.number_input("Salary Max", min_value=0, value=100, step=1)
        
        if st.form_submit_button("Submit"):
            with current_placeholder:
                pass
            with history_placeholder:
                st.session_state.history_list.append([skills, experience, salary_min, salary1_max])
                history_placeholder.write(st.session_state.history_list)

if __name__ == "__main__":
    main()
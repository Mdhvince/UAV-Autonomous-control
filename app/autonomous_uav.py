import streamlit as st



def app():
    st.set_page_config(
        page_title="Autonomous UAV",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Autonomous UAV :airplane:")


if __name__ == "__main__":
    app()


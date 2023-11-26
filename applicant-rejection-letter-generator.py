import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st


# Load OpenAI API key
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the LLM model
llm_model = "gpt-3.5-turbo"

# Create an OpenAI LLM instance
open_ai = OpenAI(temperature=0.7)


def reject_applicant(applicant_name, company_name, current_skills):

    # Create the prompt template
    template = """
        As an HR manager or talent acquisition specialist please come up with a
        short, thoughtful, honest, and helpful feedback or letter for applicant {applicant_name}
        you are going to reject as Data Scientist at your organization {company_name}. 
    """

    # Create the prompt
    prompt = PromptTemplate(input_variables=["applicant_name", "company_name"], template=template)

    # Create the LLM chain
    chain_letter = LLMChain(llm=open_ai, prompt=prompt, output_key="rejection_letter", verbose=True)
    
    # ======= Sequential Chain =====
    # chain to determine skills lacking
    template_update = """
    Include in the {rejection_letter} a list of skills they still need to learn in the future 
    based on current skills {current_skills}. For the basis of the skills lacked use this
    this job description skills requirement: 
        1. Programming Languages:
        Python, R, SQL
        Knowledge of libraries such as NumPy, Pandas, Scikit-learn (for Python), and tidyverse (for R)
        2. Statistics and Mathematics:
        Probability and Statistics
        Linear Algebra
        Calculus
        Bayesian thinking
        3. Data Exploration and Cleaning:
        Data wrangling
        Exploratory Data Analysis (EDA)
        Handling missing data
        4. Data Visualization:
        Matplotlib, Seaborn (for Python)
        ggplot2 (for R)
        Tableau, Power BI
        5. Machine Learning:
        Supervised learning (classification, regression)
        Unsupervised learning (clustering, dimensionality reduction)
        Ensemble methods
        Deep learning basics
        Model evaluation and selection
        6. Big Data Technologies:
        Apache Hadoop
        Apache Spark
        Distributed computing
        7. Database Knowledge:
        SQL (Structured Query Language)
        Database design
        NoSQL databases (e.g., MongoDB)
        8. Data Engineering:
        Extract, Transform, Load (ETL) processes
        Data pipelines
        Database management
        9. Feature Engineering:
        Creating new features from existing data
        Handling categorical data
        Feature scaling
        10. Model Deployment:
        Deployment frameworks (e.g., Flask for Python)
        Containerization (e.g., Docker)
        Cloud services (e.g., AWS, Azure, Google Cloud)
 
    The lack of skills will be the reason for rejection. Make sure to use a hopeful tone.

    
    REJECTION REASON:
    """

    prompt_reason = PromptTemplate(input_variables=["rejection_letter", "current_skills"],
                                    template=template_update)

    chain_reason = LLMChain(
        llm=open_ai,
        prompt=prompt_reason,
        output_key="rejection_reason"
    )


    # ==== Create the Sequential Chain ===
    overall_chain = SequentialChain(
        chains=[chain_letter, chain_reason],
        input_variables=["applicant_name", "company_name", "current_skills"],
        output_variables=["rejection_letter", "rejection_reason"], # return story and the translated variables!
        verbose=True
    )

    response = overall_chain({"applicant_name": applicant_name, 
                              "company_name": company_name,
                               "current_skills": current_skills})

    # Return the rejection letter
    return response


def main():

    # Set the page layout
    st.set_page_config(page_title="Applicant Rejection Letter Generator", layout="centered")

    # Display the header
    st.title("Applicant Rejection Letter Generator")

    # Display the applicant name input field
    applicant_name = st.text_input(label="Applicant Name:")
    
    # Display the applicant name input field
    company_name = st.text_input(label="Company Name:")

    # Display the reason for rejection input field
    current_skills = st.text_input(label="Applicant Current Skills:")

    # Display the submit button
    submit_button = st.button(label="Generate Rejection Letter")

    # Generate the rejection letter when the submit button is clicked
    if applicant_name and company_name and current_skills and submit_button:
        if submit_button:
            with st.spinner("Generating rejection letter..."):        
                response = reject_applicant(applicant_name=applicant_name,
                                            company_name= company_name,
                                            current_skills=current_skills)
        
        rejection_letter = reject_applicant(applicant_name=applicant_name, company_name=company_name, current_skills=current_skills)

        # Display the rejection letter
        st.markdown(f"### Rejection Letter")
        st.markdown(rejection_letter)


# Run the main function
if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from streamlit_chat import message
import openai
import sqlite3
import streamlit as st
import datetime as dt
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly
from PIL import Image
import streamlit as st
from streamlit_chat import message
import requests
import pandas as pd
from mdtable import MDTable
from pathlib import Path

st.markdown(
    """
    <style>
    .block-container {
        text-align: center;

    }
    footer {visibility: hidden;}

    .title {
        align-self: flex-start;
     </style>
    """,
    unsafe_allow_html=True
)

background = Image.open("en-logo.png")
col1, col2, col3 = st.columns([0.8, 5, 0.2])
col2.image(background, width=500)

selected_page = option_menu(
    menu_title=None,
    options=["BizMatch", "BizBot", "Idea Oasis", "Stock Sense"],
    icons=["map", "person-circle", "info", "geo"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


if selected_page == "BizMatch":
    conn = sqlite3.connect('investors.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS investors
                (name TEXT, description TEXT, funding REAL, industry TEXT, contact TEXT)''')
    conn.commit()

    # Page 1: Investor Profile
    def investor_profile():
        st.subheader("Match with a Startup Founder!")
        name = st.text_input("Name")
        description = st.text_area("Description")
        funding = st.number_input("Funding Amount", min_value=0.0)
        interests = st.text_input("Interested Industries (comma-separated)")
        contact = st.text_input("Contact Information")
        if st.button("Submit"):
            interests_list = [interest.strip().lower()
                              for interest in interests.split(",")]
            for interest in interests_list:
                c.execute("INSERT INTO investors VALUES (?, ?, ?, ?, ?)",
                          (name, description, funding, interest, contact))
            conn.commit()
            st.success("Profile submitted successfully!")

    # Page 2: Startup Founder
    def startup_founder():
        st.subheader("Match with an Investor!")
        industries = set()
        # Fetch all unique industries from the database
        for row in c.execute("SELECT DISTINCT industry FROM investors"):
            industries.add(row[0])
        selected_industry = st.selectbox(
            "Select an Industry", list(industries))
        if st.button("Find Investors"):
            # Fetch investors with similar interests from the database (case-insensitive)
            c.execute(
                "SELECT * FROM investors WHERE LOWER(industry) = LOWER(?)", (selected_industry,))
            results = c.fetchall()
            st.subheader("Matching Investors:")
            for result in results:
                st.write("Name:", result[0])
                st.write("Description:", result[1])
                st.write("Funding:", result[2])
                st.write("Contact:", result[4])
                st.write("---")

    # Streamlit App
    def main():
        st.header("Investor-Founder Matcher")

        page = option_menu(
            menu_title=None,
            options=["Investor Profile", "Startup Founder"],
            icons=["map", "person-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            key="nav_bar"
        )
        st.markdown("<br>", unsafe_allow_html=True)  # Add a line break

        if page == "Investor Profile":
            investor_profile()
        elif page == "Startup Founder":
            startup_founder()

    if __name__ == '__main__':
        main()

elif selected_page == "BizBot":
    AI21_API_KEY = st.secrets["AI21_API_KEY"]
    st.subheader("Chat about a CSV Document ")
    uploaded_file = st.file_uploader('', type='csv', accept_multiple_files=False)

    if uploaded_file:
        
        some_bytes = uploaded_file.getvalue()
        with open("my_file.csv", "wb") as binary_file:
            binary_file.write(some_bytes)
            base = Path.cwd()
            PATH_TO_FILE = f"{base}/my_file.csv"

        markdown = MDTable(PATH_TO_FILE)
        markdown_string_table = markdown.get_table()
        with st.expander("Input Table", expanded=False):
            st.write(markdown_string_table)
        markdown.save_table('out.csv')

        def get_answer(user_input):
            response = requests.post("https://api.ai21.com/studio/v1/experimental/j1-grande-instruct/complete",
            headers={"Authorization": "Bearer " + AI21_API_KEY},
            json={
            "prompt": markdown_string_table + "\nQ: " + user_input + "\nA:",
            "numResults": 1,
            "maxTokens": 200,
            "temperature": 0,
            "topKReturn": 0,
            "topP":1,
            "countPenalty": {
                    "scale": 0,
                    "applyToNumbers": False,
                    "applyToPunctuations": False,
                    "applyToStopwords": False,
                    "applyToWhitespaces": False,
                    "applyToEmojis": False
            },
            "frequencyPenalty": {
                    "scale": 0,
                    "applyToNumbers": False,
                    "applyToPunctuations": False,
                    "applyToStopwords": False,
                    "applyToWhitespaces": False,
                    "applyToEmojis": False
            },
            "presencePenalty": {
                    "scale": 0,
                    "applyToNumbers": False,
                    "applyToPunctuations": False,
                    "applyToStopwords": False,
                    "applyToWhitespaces": False,
                    "applyToEmojis": False
            },
            "stopSequences":["↵↵"]
            }
        )

            return response.json()
        
        def get_text():
            input_text = st.text_input("You: ","")
            return input_text 

        user_inputs = get_text()

        if user_inputs:
            res = get_answer(user_inputs)
            # st.write(res)
            try:
                answer = res["completions"][0]["data"]["text"]
                message(user_inputs, is_user=True)
                message(answer)
            except:
                st.error(res['detail'])
                
    openai.api_key = st.secrets["OPENAI_KEY"]
    st.subheader("Or ask any General Questions")
    query = st.text_input("Have any business questions?")

    if query:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=query,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.5,
        )
        answer = response.choices[0].text

        st.write(answer)

openai.api_key = st.secrets["OPENAI_KEY"]

def generate_name(industry, budget, target_audience):
        prompt_text = f"Generate a unique business name with no period in the {industry} industry with a budget of {budget} , targeting {target_audience}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].text.strip()

def generate_idea(industry, budget, target_audience):
        prompt_text = f"Generate a unique business idea in the {industry} industry with a budget of {budget} , targeting {target_audience}."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].text.strip()

def get_competition(idea):
    prompt_text = f"Find potential competitors or similar companies for a business idea like: {idea}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_text,
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

if selected_page == "Idea Oasis":
    st.subheader("Generate a Unique Idea ")
    
    industry = st.text_input("Industry")
    budget = st.text_input("Budget")
    target_audience = st.text_input("Target Audience")

    if st.button("Generate Idea"):
        name = generate_name(industry,budget,target_audience)
        idea = generate_idea(industry, budget, target_audience)

        st.markdown(f"### {name}\n{idea}")
    
    st.subheader("Discover the Competition")

    competition_idea = st.text_input("Enter your business idea")
    
    if st.button("Get Competitors"):
        competitors = get_competition(competition_idea)
        st.markdown(f"### Potential Competitors\n{competitors}")

elif selected_page == "Stock Sense":
    TODAY = dt.datetime.now()
    START = dt.datetime(TODAY.year - 10, TODAY.month, TODAY.day)

    st.header("Stock Analysis & Prediction")

    stock_choice = st.text_input("Choose a stock")
    st.write("The stock you chose is ", stock_choice)

    #This function generates the plot graph for the stock's value in the past
    def plot_generate_data():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Open'], name='opening_stock_value'))
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'],
                    name='closing_stock_value'))
        fig.layout.update(title_text="History of Stock",
                        xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    #This saves the searches already made in cache 
    @st.cache_data

    def generate_data(stock):
        data = yf.download(stock, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_loading = st.text("Loading....")
    parsed = True
    if stock_choice != "":
        data = generate_data(stock_choice)
        data_loading.text("....done!")

        st.subheader("Stock History")
        st.write(data.tail())
        plot_generate_data()

        # Predictions
        t_days = st.slider("Days of prediction:", 1, 365)

        model = Prophet()
        data = data.reset_index()
        data[["ds", "y"]] = data[["Date", "Adj Close"]]
        #This code will only continue if the inputted stock was a valid stock inside the yahoo finance database
        try:
            prediction_loading = st.text("Loading....")
            model.fit(data)
            #Calculates the future values of the stock
            future = model.make_future_dataframe(periods=t_days)
            prediction = model.predict(future)
            model.plot(prediction)

            #plots the prediction
            st.write(prediction.tail())

            st.write("Predicted Data")
            pred1 = plot_plotly(model, prediction)
            st.plotly_chart(pred1)

            # This creates a vareity of graphs that represent dfferent types of predictions relative to the stock
            st.write("Prediction Components")
            pred2 = model.plot_components(prediction)
            st.write(pred2)

            today_value = int(data['High'][2150])
            prediction_value = int(prediction['yhat_upper'][2150])
            difference = prediction_value - today_value
            percentage = round((difference / today_value) * 100, 1)
            st.write("The difference between the predicted value and today's value is " + str(difference))
            st.write("There is a percentage difference of " + str(percentage) + "%")
            prediction_loading.text("....done!")
        except ValueError:
            st.write("INVALID INPUT!")

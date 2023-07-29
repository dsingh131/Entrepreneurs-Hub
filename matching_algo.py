import streamlit as st
import sqlite3

# Create a SQLite database and table
conn = st.experimental_connection('investors_db', type='sql')

c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS investors
             (name TEXT, description TEXT, funding REAL, industry TEXT, contact TEXT)''')
conn.commit()

# Page 1: Investor Profile
def investor_profile():
    st.title("Investor Profile")
    name = st.text_input("Name")
    description = st.text_area("Description")
    funding = st.number_input("Funding Amount", min_value=0.0)
    interests = st.text_input("Interested Industries (comma-separated)")
    contact = st.text_input("Contact Information")
    if st.button("Submit"):
        interests_list = [interest.strip().lower() for interest in interests.split(",")]
        for interest in interests_list:
            c.execute("INSERT INTO investors VALUES (?, ?, ?, ?, ?)",
                      (name, description, funding, interest, contact))
        conn.commit()
        st.success("Profile submitted successfully!")

# Page 2: Startup Founder
def startup_founder():
    st.title("Startup Founder")
    industries = set()
    # Fetch all unique industries from the database
    for row in c.execute("SELECT DISTINCT industry FROM investors"):
        industries.add(row[0])
    selected_industry = st.selectbox("Select an Industry", list(industries))
    if st.button("Find Investors"):
        # Fetch investors with similar interests from the database (case-insensitive)
        c.execute("SELECT * FROM investors WHERE LOWER(industry) = LOWER(?)", (selected_industry,))
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
    st.header("Investor-Founder Matching App")
    page = st.sidebar.selectbox("Select a page", ("Investor Profile", "Startup Founder"))

    if page == "Investor Profile":
        investor_profile()
    elif page == "Startup Founder":
        startup_founder()

if __name__ == '__main__':
    main()


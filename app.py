import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
from groq import Groq, APIError, RateLimitError
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        background-color: #E2E8F0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        padding-top: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #BFDBFE;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
    }
    .sidebar-header {
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB Connection with Retry
for attempt in range(3):
    try:
        client = MongoClient(MONGO_URI)
        db = client['business_rag']
        business_collection = db['business_attributes']
        question_collection = db['questions']
        listings_collection = db['business_listings']
        st.write("Connected to MongoDB")
        break
    except pymongo.errors.ConnectionError as e:
        st.warning(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(2)
else:
    st.error("Failed to connect to MongoDB after retries. Please check your connection details.")
    st.stop()

# Groq API Setup
groq_client = Groq(api_key=GROQ_API_KEY)

# Helper Functions
def safe_float(value, default=0):
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return default

@st.cache_data(ttl=3600)
def get_business(business_name):
    return business_collection.find_one({"business_name": business_name})

@st.cache_data(ttl=3600)
def get_all_businesses(limit=2072):
    return list(business_collection.find().limit(limit))

def match_question(query_embedding, questions):
    best_match = None
    highest_similarity = -1
    for q in questions:
        similarity = 1 - cosine(query_embedding, q['embedding'])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = q
    return best_match

def groq_qna(query, context=None):
    try:
        context_str = f"Context: {context}" if context else "No specific context provided."
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert business analyst. Provide detailed, accurate, and actionable responses."},
                {"role": "user", "content": f"{context_str}\n\nQuery: {query}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except RateLimitError:
        st.error("Rate limit exceeded. Please try again later.")
        return "Rate limit exceeded."
    except APIError as e:
        st.error(f"Groq API error: {e}")
        return "Failed to get response from AI."
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return "An unexpected error occurred."

# Get list of business names
business_names = [b['business_name'] for b in get_all_businesses()]

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Business Insights Hub")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", [
        "🔍 Smart Q&A",
        "💰 Company Valuation",
        "📊 Business Assessment",
        "🌐 Marketplace"
    ])

    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

# Session State Initialization
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_responses' not in st.session_state:
    st.session_state.assessment_responses = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'valuation_step' not in st.session_state:
    st.session_state.valuation_step = 0
if 'sample_question' not in st.session_state:
    st.session_state.sample_question = None

# Pre-populate query from sample question if set
if st.session_state.sample_question:
    sample_query = st.session_state.sample_question
    st.session_state.sample_question = None  # Reset sample question
else:
    sample_query = ""

# 1. Smart Q&A
if "Smart Q&A" in page:
    st.markdown("# 🔍 Smart Business Intelligence")
    st.markdown("Get expert answers to your business questions powered by AI.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask a question about business strategy, valuation, market trends, etc.",
                              placeholder="E.g., How does this business make money?", value=sample_query)
    with col2:
        business_name = st.selectbox("Select Business Context (Optional)", ["None"] + business_names)

    if query:
        submit_button = st.button("Get Insights", use_container_width=True)
        if submit_button:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing your question..."):
                    if business_name != "None":
                        business = get_business(business_name)
                        response = groq_qna(query, str(business))
                    else:
                        response = groq_qna(query)

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### 💡 Expert Analysis")
                    st.markdown(response)
                    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Sample Questions"):
        sample_questions = [
            "What are typical SaaS business valuation multiples?",
            "How can I improve my business's customer retention?",
            "What are common cash flow challenges for startups?",
            "How do I determine the right pricing strategy for my products?"
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.sample_question = q
                st.rerun()

# 2. Company Valuation Estimator
elif "Company Valuation" in page:
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using multiple industry-standard valuation methods.")

    valuation_questions = [
        "What is your company's annual revenue (in USD)?",
        "What are your company's annual earnings (net income, in USD)?",
        "What is your company's EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization, in USD)?",
        "What industry does your company operate in?",
        "What is your company's total assets value (in USD)?",
        "What is your company's total liabilities (in USD)?",
        "What are your projected cash flows for the next 5 years (comma-separated, in USD)?",
        "What is your company's growth rate (e.g., High, Moderate, Low)?"
    ]

    total_steps = len(valuation_questions)
    current_step = st.session_state.valuation_step

    st.progress(current_step / total_steps)
    st.markdown(f"##### Step {current_step + 1} of {total_steps}")

    if current_step < total_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        current_question = valuation_questions[current_step]
        st.markdown(f"### {current_question}")

        help_texts = {
            0: "Enter your total annual revenue before expenses.",
            1: "Enter your annual profit after all expenses and taxes.",
            2: "EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization.",
            3: "Select the industry that best describes your business.",
            4: "Total value of all assets owned by your company.",
            5: "Total of all debts and obligations owed by your company.",
            6: "Estimate your cash flows for each of the next 5 years, separated by commas.",
            7: "Assess your company's growth trend compared to industry standards."
        }

        if current_step in help_texts:
            st.markdown(f"*{help_texts[current_step]}*")

        if current_step in [0, 1, 2, 4, 5]:
            answer = st.number_input("USD", min_value=0, step=1000, format="%i", key=f"val_step_{current_step}")
            answer = str(answer)
        elif current_step == 3:
            industries = ["Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Healthcare", "Financial Services", "Real Estate", "Hospitality", "Technology", "Energy", "Other"]
            answer = st.selectbox("Select", industries, key=f"val_step_{current_step}")
        elif current_step == 6:
            year_cols = st.columns(5)
            cash_flows = []
            for i, col in enumerate(year_cols):
                with col:
                    cf = col.number_input(f"Year {i+1}", min_value=0, step=1000, format="%i", key=f"cf_{i}")
                    cash_flows.append(str(cf))
            answer = ",".join(cash_flows)
        elif current_step == 7:
            answer = st.select_slider("Select", options=["Low", "Moderate", "High"], key=f"val_step_{current_step}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if current_step > 0:
                if st.button("Back"):
                    st.session_state.valuation_step -= 1
                    st.rerun()
        with col2:
            if st.button("Next", use_container_width=True):
                st.session_state.valuation_data[current_question] = answer
                st.session_state.valuation_step += 1
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if current_step >= total_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Company Information Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Industry:**")
            st.markdown("**Annual Revenue:**")
            st.markdown("**Net Income:**")
            st.markdown("**EBITDA:**")
        with col2:
            st.markdown(f"{st.session_state.valuation_data.get(valuation_questions[3], 'N/A')}")
            st.markdown(f"${safe_float(st.session_state.valuation_data.get(valuation_questions[0], '0')):,.2f}")
            st.markdown(f"${safe_float(st.session_state.valuation_data.get(valuation_questions[1], '0')):,.2f}")
            st.markdown(f"${safe_float(st.session_state.valuation_data.get(valuation_questions[2], '0')):,.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

        revenue = safe_float(st.session_state.valuation_data.get(valuation_questions[0], "0"))
        earnings = safe_float(st.session_state.valuation_data.get(valuation_questions[1], "0"))
        ebitda = safe_float(st.session_state.valuation_data.get(valuation_questions[2], "0"))
        industry = st.session_state.valuation_data.get(valuation_questions[3], "Other")
        assets = safe_float(st.session_state.valuation_data.get(valuation_questions[4], "0"))
        liabilities = safe_float(st.session_state.valuation_data.get(valuation_questions[5], "0"))
        cash_flows_str = st.session_state.valuation_data.get(valuation_questions[6], "0,0,0,0,0")
        cash_flows = [safe_float(cf) for cf in cash_flows_str.split(",")]
        growth = st.session_state.valuation_data.get(valuation_questions[7], "Low")

        industry_data_list = list(business_collection.find({"Business Attributes.Business Fundamentals.Industry Classification.Primary Industry": industry}))
        industry_avg_pe = 15.0
        industry_avg_ebitda_multiple = 8.0
        if industry_data_list:
            pe_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('P/E Ratio', industry_avg_pe) for b in industry_data_list]
            ebitda_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('EV/EBITDA Multiple', industry_avg_ebitda_multiple) for b in industry_data_list]
            industry_avg_pe = np.mean([float(p) for p in pe_list if isinstance(p, (int, float)) and p > 0]) if any(isinstance(p, (int, float)) and p > 0 for p in pe_list) else industry_avg_pe
            industry_avg_ebitda_multiple = np.mean([float(e) for e in ebitda_list if isinstance(e, (int, float)) and e > 0]) if any(isinstance(e, (int, float)) and e > 0 for e in ebitda_list) else industry_avg_ebitda_multiple

        with st.spinner("Calculating company valuation..."):
            valuation_prompt = f"""
            You are an expert in business valuation. Given the following data about a company and industry benchmarks, calculate its valuation using all applicable methods:
            - Company Data:
              - Annual Revenue: ${revenue:,.2f}
              - Annual Earnings (Net Income): ${earnings:,.2f}
              - EBITDA: ${ebitda:,.2f}
              - Industry: {industry}
              - Total Assets: ${assets:,.2f}
              - Total Liabilities: ${liabilities:,.2f}
              - Projected Cash Flows (5 years): {', '.join([f'${cf:,.2f}' for cf in cash_flows])}
              - Growth Rate: {growth}
            - Industry Benchmarks:
              - Average P/E Ratio: {industry_avg_pe}
              - Average EV/EBITDA Multiple: {industry_avg_ebitda_multiple}

            Valuation Methods to Use:
            1. Market-Based:
               - Comparable Company Analysis (CCA): Use P/E Ratio (Company Value = Earnings × P/E Multiple) and EV/EBITDA.
               - Precedent Transactions: Suggest a multiplier based on industry norms if data is insufficient.
            2. Income-Based:
               - Discounted Cash Flow (DCF): Use a discount rate of 10% (WACC) unless industry suggests otherwise. Formula: Sum(CF_t / (1 + r)^t).
               - Earnings Multiplier (EV/EBITDA): Enterprise Value = EBITDA × Industry Multiple.
            3. Asset-Based:
               - Book Value: Assets - Liabilities.
               - Liquidation Value: Estimate based on assets (assume 70% recovery unless specified).

            Provide a detailed response with:
            - Calculated valuation for each method (if applicable).
            - Explanation of why each method is suitable or not for this company based on the industry and data.
            - A recommended valuation range combining the results.

            Format your response with clear headings and bullet points. Make sure to include a final summary section with a recommended valuation range at the end.
            """
            valuation_result = groq_qna(valuation_prompt)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Valuation Results")
        st.markdown(valuation_result)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start New Valuation", use_container_width=True):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.rerun()

# 3. Interactive Business Assessment
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    # Initialize session state for business assessment
    if 'assessment_started' not in st.session_state:
        st.session_state.assessment_started = False
        st.session_state.assessment_responses = []
        st.session_state.current_question = None
        st.session_state.business_type = None
        st.session_state.business_context = {}

    if not st.session_state.assessment_started:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Let's start by understanding your business")
        
        business_type = st.selectbox(
            "What type of business are you analyzing?",
            ["Convenience Store", "Mechanical Shop", "Grocery Store", "Other Small Business"]
        )
        
        if business_type == "Other Small Business":
            business_type = st.text_input("Please specify the type of business")
        
        if st.button("Start Assessment"):
            if business_type:
                st.session_state.business_type = business_type
                st.session_state.assessment_started = True
                st.session_state.current_question = {
                    "question": "How long has your business been operating?",
                    "category": "Business History"
                }
                st.rerun()
            else:
                st.warning("Please select or specify a business type")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Display progress
        total_questions = len(st.session_state.assessment_responses) + 1
        st.progress(min(1.0, total_questions / 15))
        
        if st.session_state.current_question:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### Question {total_questions}")
            st.markdown(f"**{st.session_state.current_question['question']}**")
            
            response = st.text_area("Your Answer", height=100, key="current_answer")
            
            if st.button("Submit Answer", use_container_width=True):
                if response.strip():
                    # Save the current Q&A
                    st.session_state.assessment_responses.append({
                        "question": st.session_state.current_question["question"],
                        "answer": response,
                        "category": st.session_state.current_question.get("category", "General")
                    })
                    
                    # Update business context
                    st.session_state.business_context.update({
                        st.session_state.current_question["question"]: response
                    })
                    
                    # Generate next question based on context
                    next_question = generate_context_aware_question(
                        st.session_state.assessment_responses,
                        st.session_state.business_type
                    )
                    
                    st.session_state.current_question = {
                        "question": next_question["question"],
                        "category": "Follow-up"
                    }
                    
                    st.rerun()
                else:
                    st.warning("Please provide an answer before proceeding")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## Business Assessment Results")
            
            # Format assessment data for analysis
            assessment_data = "\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in st.session_state.assessment_responses
            ])
            
            analysis_prompt = f"""
            You are an expert business consultant analyzing a {st.session_state.business_type} business.
            Based on the following assessment responses:

            {assessment_data}

            Please provide a comprehensive analysis that includes:
            1. Executive Summary
            2. Business Model Analysis
            3. Market Position and Competition
            4. Financial Health Assessment
            5. Operational Strengths and Weaknesses
            6. Growth Opportunities
            7. Risk Factors
            8. Strategic Recommendations
            9. Investment Potential

            Format your response with clear headings and bullet points.
            Be specific and provide actionable insights based on the business type and responses.
            """
            
            with st.spinner("Generating business assessment report..."):
                analysis_result = groq_qna(analysis_prompt)
            
            st.markdown(analysis_result)
            
            if st.button("Start New Assessment", use_container_width=True):
                st.session_state.assessment_started = False
                st.session_state.assessment_responses = []
                st.session_state.current_question = None
                st.session_state.business_type = None
                st.session_state.business_context = {}
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

# 4. Showcase Listings for Investors
elif "Marketplace" in page:
    st.markdown("# 🌐 Business Marketplace")
    st.markdown("Connect businesses with investors.")

    tabs = st.tabs(["🏢 List Your Business", "💸 Investor Dashboard"])

    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Create Your Business Listing")

        col1, col2 = st.columns(2)
        with col1:
            listing_name = st.text_input("Business Name", placeholder="E.g., Acme Technologies")
            listing_industry = st.selectbox("Industry", [
                "Technology", "E-commerce", "Healthcare", "Finance",
                "Real Estate", "Manufacturing", "Retail", "Services",
                "Food & Beverage", "Education", "Other"
            ])
            listing_revenue = st.number_input("Annual Revenue (USD)", min_value=0, step=10000, format="%i")
        with col2:
            listing_location = st.text_input("Location", placeholder="City, Country")
            founding_year = st.number_input("Year Founded", min_value=1900, max_value=datetime.now().year, value=datetime.now().year)
            team_size = st.number_input("Team Size", min_value=1, value=5)

        listing_description = st.text_area("Business Description", height=150,
                                          placeholder="Describe your business, value proposition, market opportunity, and why investors should be interested.")

        col3, col4 = st.columns(2)
        with col3:
            investment_sought = st.number_input("Investment Amount Sought (USD)", min_value=0, step=50000, format="%i")
            equity_offered = st.slider("Equity Offered (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        with col4:
            listing_contact = st.text_input("Contact Email", placeholder="your.email@example.com")
            website = st.text_input("Website URL", placeholder="https://yourbusiness.com")

        if st.button("Submit Listing", use_container_width=True):
            listing = {
                "business_name": listing_name,
                "industry": listing_industry,
                "revenue": listing_revenue,
                "description": listing_description,
                "contact": listing_contact,
                "location": listing_location,
                "founded": founding_year,
                "team_size": team_size,
                "investment_sought": investment_sought,
                "equity_offered": equity_offered,
                "website": website,
                "listed_date": datetime.now().isoformat()
            }
            listings_collection.insert_one(listing)
            st.success("✅ Business listed successfully!")

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Investor Dashboard")

        col1, col2, col3 = st.columns(3)
        with col1:
            industry_filter = st.multiselect("Filter by Industry", [
                "Technology", "E-commerce", "Healthcare", "Finance",
                "Real Estate", "Manufacturing", "Retail", "Services",
                "Food & Beverage", "Education", "Other"
            ])
        with col2:
            min_revenue = st.number_input("Minimum Revenue (USD)", min_value=0, step=50000, value=0)
        with col3:
            max_investment = st.number_input("Maximum Investment (USD)", min_value=0, step=100000, value=1000000)

        query = {}
        if industry_filter:
            query["industry"] = {"$in": industry_filter}
        if min_revenue > 0:
            query["revenue"] = {"$gte": min_revenue}
        if max_investment > 0:
            query["investment_sought"] = {"$lte": max_investment}

        listings = list(listings_collection.find(query))

        if listings:
            for listing in listings:
                st.markdown(f"""
                <div style='padding: 1.2rem; background-color: white; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h3 style='margin: 0; color: #1E3A8A;'>{listing.get('business_name', 'Unnamed Business')}</h3>
                        <span style='font-size: 0.8rem; background-color: #EFF6FF; padding: 0.2rem 0.5rem; border-radius: 4px; color: #1E3A8A;'>{listing.get('industry', 'Uncategorized')}</span>
                    </div>
                    <div style='display: flex; gap: 1rem; margin-top: 0.8rem; font-size: 0.85rem; color: #64748B;'>
                        <div><span style='font-weight: 500;'>📍 Location:</span> {listing.get('location', 'Not specified')}</div>
                        <div><span style='font-weight: 500;'>🏢 Founded:</span> {listing.get('founded', 'Not specified')}</div>
                        <div><span style='font-weight: 500;'>👥 Team:</span> {listing.get('team_size', 'Not specified')}</div>
                    </div>
                    <p style='margin-top: 0.8rem; margin-bottom: 0.8rem; font-size: 0.95rem;'>{listing.get('description', 'No description provided.')}</p>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;'>
                        <div>
                            <div style='font-weight: 500; color: #1E3A8A;'>Seeking ${listing.get('investment_sought', 0):,}</div>
                            <div style='font-size: 0.85rem; color: #64748B;'>For {listing.get('equity_offered', 0)}% equity</div>
                        </div>
                        <div>
                            <div style='font-weight: 500; color: #1E3A8A;'>Revenue: ${listing.get('revenue', 0):,}</div>
                            <div style='font-size: 0.85rem; color: #64748B;'>Annual</div>
                        </div>
                        <a href='mailto:{listing.get('contact', '')}' style='text-decoration: none; background-color: #1E3A8A; color: white; padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.9rem;'>Contact</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No businesses match your filter criteria.")

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub © 2025 | Powered by Groq AI
</div>
""", unsafe_allow_html=True)

def generate_context_aware_question(previous_qa, business_type):
    """
    Generate a follow-up question based on the previous answer and business context.
    This function uses the LLM to analyze the previous answer and generate a relevant follow-up.
    """
    # Ensure Groq client is initialized
    if not hasattr(st, 'groq_client'):
        try:
            st.groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {str(e)}")
            return {
                "question": "Can you tell me more about your business operations?",
                "context": "General follow-up to gather more information"
            }

    qa_context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in previous_qa])
    
    prompt = f"""
    Expert Business Investor Interview System
System Role Definition
You are an expert business analyst and investor interviewer, combining the analytical precision of Kevin O'Leary, the technical insight of Mark Cuban, and the strategic vision of other top investors from "Shark Tank" and "Dragon's Den" while maintaining a professional, neutral tone. Your purpose is to conduct in-depth interviews with business owners to comprehensively evaluate their companies for potential investment or acquisition.

Interview Context & Objectives
You have access to a database of approximately 1021 unique questions from investor shows like Shark Tank and Dragon's Den. Your goal is to leverage these questions strategically while adapting them to each specific business. The interview should gather all information necessary to:
    1. Build a complete business profile 
    2. Assess viability and growth potential 
    3. Identify strengths, weaknesses, and opportunities 
    4. Determine appropriate valuation methods and ranges 
    5. Generate an investor-ready business summary 

Adaptive Interview Methodology
Phase 1: Initial Discovery (3-5 questions)
Begin with general questions to identify fundamental business parameters:
- "Tell me about your business and what problem you're solving."
- "How long have you been operating and what's your current stage?"
- "What industry are you in and who are your target customers?"
- "What's your revenue model and current traction?"

Phase 2: Business Model Deep Dive (5-7 questions)
Tailor questions based on the business model identified in Phase 1:
For Digital/SaaS businesses: Focus on metrics like MRR/ARR, churn rate, CAC, LTV, and scalability
- "What's your monthly recurring revenue and growth rate?"
- "What's your customer acquisition cost compared to lifetime value?"
- "What's your churn rate and retention strategy?"
For Physical Product businesses: Focus on production, supply chain, margins, and distribution
- "What are your production costs and gross margins?"
- "How do you manage your supply chain and inventory?"
- "What are your distribution channels and retail strategy?"
For Service businesses: Focus on scalability, capacity utilization, pricing models
- "How do you scale your service delivery beyond your personal time?"
- "What's your hourly/project rate structure and utilization rate?"
- "How do you maintain quality as you expand your team?"

Phase 3: Market & Competition Analysis (4-6 questions)
Adapt questions based on market maturity and competitive landscape:
- "What's your total addressable market size and how did you calculate it?"
- "Who are your top 3 competitors and how do you differentiate?"
- "What barriers to entry exist in your market?"
- "What market trends are impacting your growth potential?"

Phase 4: Financial Performance (5-8 questions)
Tailor financial questions based on business stage:
For Pre-revenue/Early stage:
- "What's your burn rate and runway?"
- "What are your financial projections for the next 24 months?"
- "What assumptions underlie your revenue forecasts?"
For Revenue-generating businesses:
- "What has your year-over-year revenue growth been?"
- "Break down your cost structure between fixed and variable costs."
- "What's your path to profitability and timeline?"
- "What are your gross and net margins?"
For Profitable businesses:
- "What's your EBITDA and how has it evolved over time?"
- "What's your cash conversion cycle?"
- "How do you reinvest profits back into the business?"

Phase 5: Team & Operations (3-5 questions)
- "Tell me about your founding team and key executives."
- "What critical roles are you looking to fill next?"
- "How is equity distributed among founders and employees?"
- "What operational challenges are limiting your growth?"

Phase 6: Investment & Growth Strategy (4-6 questions)
- "How much capital are you raising and at what valuation?"
- "How will you allocate the investment funds?"
- "What specific milestones will this funding help you achieve?"
- "What's your long-term exit strategy?"

Dynamic Adaptation Requirements
Pattern Recognition Flags
Throughout the interview, identify patterns that require deeper investigation:
Red Flags - Require immediate follow-up:
    • Inconsistent financial numbers 
    • Unrealistic market size claims 
    • Vague answers about competition 
    • Excessive founder salaries relative to revenue 
    • Unreasonable valuation expectations 
Opportunity Signals - Areas to explore further:
    • Unusually high margins for the industry 
    • Proprietary technology or IP 
    • Evidence of product-market fit 
    • Strong team with relevant experience 
    • Clear customer acquisition strategy with proven ROI 
Jump Logic Instructions
    • If a response reveals a critical issue or opportunity, immediately pivot to explore that area more deeply before returning to your sequence 
    • If you detect inconsistency between answers, flag it and seek clarification 
    • If the business has unusual characteristics that don't fit standard models, adapt your questioning approach accordingly 
Response Analysis
Continuously evaluate:
    • Answer quality and thoroughness 
    • Internal consistency across topics 
    • Information gaps requiring additional questions 
    • Unique business aspects that warrant customized questions 
Strategic Database Utilization
When selecting or formulating questions:
    1. Start with general questions from your database that match the current business context 
    2. Adapt database questions to the specific business type, size, and stage 
    3. Create logical follow-up questions based on previous answers 
    4. When encountering unique business aspects, formulate new questions inspired by patterns in your database 
Communication Guidelines
Interview Flow
    • Maintain a conversational but purposeful tone 
    • Ask one question at a time to ensure clarity 
    • Begin with open-ended questions before narrowing focus 
    • Acknowledge and build upon previous answers to show active listening 
    • Use transitional phrases when changing topics: "Now I'd like to understand more about..." 
Question Formulation
    • Be direct and specific in your questions 
    • Avoid leading questions that suggest preferred answers 
    • Use neutral language that doesn't assume success or failure 
    • When needed, request quantifiable metrics rather than generalities 
    • Frame follow-up questions that refer to previous answers: "You mentioned X earlier. How does that relate to...?" 
Business Valuation Framework
Apply appropriate valuation methods based on business type and stage:
    1. For Pre-Revenue Companies: 
        ◦ Team and IP assessment 
        ◦ Market opportunity sizing 
        ◦ Comparable early-stage funding rounds 
    2. For Early-Stage Revenue Companies: 
        ◦ Revenue multiples based on growth rate 
        ◦ Customer acquisition economics assessment 
        ◦ Comparable transaction analysis 
    3. For Established Companies: 
        ◦ P/E ratios 
        ◦ EV/EBITDA multiples 
        ◦ Discounted Cash Flow analysis 
        ◦ Book value and asset-based valuations 
Analysis & Deliverables
After completing the interview, prepare:
    1. Business Profile Summary including: 
        ◦ Company overview and value proposition 
        ◦ Market opportunity assessment 
        ◦ Competitive positioning 
        ◦ Team evaluation 
        ◦ Business model analysis 
    2. Financial Analysis including: 
        ◦ Revenue and profitability metrics 
        ◦ Growth trajectory 
        ◦ Unit economics 
        ◦ Capital efficiency 
    3. Valuation Assessment including: 
        ◦ Methodologies applied 
        ◦ Comparable company/transaction benchmarks 
        ◦ Recommended valuation range 
        ◦ Key value drivers and detractors 
    4. Investment Considerations including: 
        ◦ Key strengths and differentiators 
        ◦ Risk factors and mitigation strategies 
        ◦ Growth opportunities 
        ◦ Strategic recommendations
    """
    
    try:
        response = st.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert business analyst and investor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured question
            return {
                "question": response.choices[0].message.content.strip(),
                "context": "Generated based on previous conversation"
            }
    except Exception as e:
        st.error(f"Error generating follow-up question: {str(e)}")
        return {
            "question": "Can you tell me more about your business operations?",
            "context": "General follow-up to gather more information"
        }

def get_business_category_questions(business_type):
    """
    Get category-specific questions from MongoDB based on business type
    """
    try:
        # Search for questions in the database that match the business type
        category_questions = list(question_collection.find({
            "business_type": {"$regex": business_type, "$options": "i"}
        }))
        
        if not category_questions:
            # If no specific questions found, get general small business questions
            category_questions = list(question_collection.find({
                "category": "small_business"
            }))
        
        return category_questions
    except:
        return []

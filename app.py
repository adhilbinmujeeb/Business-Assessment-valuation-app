import statistics 
import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# Gemini API Setup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')  # Adjust model name as per Gemini API documentation

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

def gemini_qna(query, context=None):
    try:
        context_str = f"Context: {context}" if context else "No specific context provided."
        system_prompt = """
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
        # Combine system prompt and user query
        full_prompt = f"{system_prompt}\n\n{context_str}\n\nQuery: {query}"
        
        # Generate response using Gemini API
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8000
            )
        )
        return response.text
    except genai.types.BlockedPromptException:
        st.error("The prompt was blocked by Gemini API. Please try rephrasing your query.")
        return "Prompt blocked by API."
    except genai.types.GenerationError as e:
        st.error(f"Gemini API error: {e}")
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
        "💰 Company Valuation",
        "📊 Business Assessment",
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

# 2. Company Valuation Estimator
if page == "💰 Company Valuation":
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using standard methods benchmarked against real pitch data.")

    # --- Step 0: Collect Basic Info (Name, Industry, Revenue) ---
    if st.session_state.valuation_step == 0:
        st.progress(0 / 4)
        st.markdown("##### Step 1 of 4: Basic Information")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Tell us about your company")

        company_name = st.text_input(
            "What is your Company Name?",
            key="val_company_name",
            placeholder="Enter your company name here",
            value=st.session_state.valuation_data.get('company_name', '') # Retain value if going back
        )

        industry = st.selectbox(
            "What industry best describes your company?",
            st.session_state.available_industries,
            key="val_industry",
            index=st.session_state.available_industries.index(st.session_state.valuation_data.get('industry')) if st.session_state.valuation_data.get('industry') in st.session_state.available_industries else 0,
            help="Select the closest matching industry."
        )
        revenue = st.number_input(
            "What is your company's most recent Annual Revenue (USD)?",
            min_value=0.0,
            step=1000.0,
            format="%.2f",
            key="val_revenue",
            value=st.session_state.valuation_data.get('revenue', 0.0), # Retain value
            help="Total sales/turnover over the last 12 months."
        )

        if st.button("Next: Get Benchmarks", use_container_width=True):
            # Validation checks
            if not company_name.strip():
                st.warning("Please enter your company name.")
            elif industry == "Select an Industry...":
                st.warning("Please select a valid industry.")
            elif revenue <= 0:
                 st.warning("Please enter a positive annual revenue.")
            else:
                # Store initial data
                st.session_state.valuation_data['company_name'] = company_name.strip()
                st.session_state.valuation_data['industry'] = industry
                st.session_state.valuation_data['revenue'] = revenue
                # Reset benchmarks from previous runs before fetching new ones
                st.session_state.benchmarks = None
                st.session_state.comparables = None

                with st.spinner(f"Finding comparable deals for '{industry}'..."):
                    time.sleep(0.5) # Small delay for spinner visibility
                    st.session_state.comparables = get_comparable_deals(industry, listings_collection)

                    if st.session_state.comparables:
                         st.session_state.benchmarks = calculate_sharktank_benchmarks(st.session_state.comparables)
                         # Check if benchmarks are meaningful
                         if st.session_state.benchmarks and st.session_state.benchmarks.get("count", 0) > 2: # Require at least 3 deals for meaningful benchmarks
                            st.session_state.valuation_step = 1 # Move to show benchmarks
                            st.rerun()
                         else:
                            st.warning(f"Found only {st.session_state.benchmarks.get('count', 0)} comparable deals for '{industry}'. Proceeding without specific dataset benchmarks.")
                            st.session_state.benchmarks = None # Ensure benchmarks are None
                            st.session_state.valuation_step = 2 # Skip benchmark display
                            st.rerun()
                    else:
                         st.warning(f"Could not find comparable deals for '{industry}' in the dataset. Proceeding without dataset benchmarks.")
                         st.session_state.benchmarks = None # Ensure benchmarks are None
                         st.session_state.valuation_step = 2 # Skip benchmark display
                         st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 1: Display Benchmarks (if found and valid) ---
    elif st.session_state.valuation_step == 1:
        # Double-check if benchmarks exist (in case user navigates weirdly)
        if not st.session_state.benchmarks or st.session_state.benchmarks.get("count", 0) <= 2:
            st.warning("Benchmarks not available or insufficient for this industry. Please proceed to enter financial details.")
            st.session_state.valuation_step = 2 # Move user forward
            time.sleep(1) # Allow message visibility
            st.rerun()
        else:
            st.progress(1 / 4)
            st.markdown("##### Step 2 of 4: Dataset Benchmarks")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            b = st.session_state.benchmarks
            industry_name = st.session_state.valuation_data.get('industry','N/A')
            st.markdown(f"### Benchmarks from {b['count']} Deals in '{industry_name}'")
            st.caption("_Note: Based on data from investor pitches. Reflects negotiated outcomes._")

            # Function to format metrics consistently
            def fmt_usd(value): return f"${value:,.0f}"
            def fmt_mult(value): return f"{value:.2f}x" if value else "N/A"
            def fmt_pct(value): return f"{value:.1f}%" if value else "N/A"

            col1, col2, col3 = st.columns(3)
            with col1:
                 with st.container(border=True): # Use border container for metric groups
                    st.metric(label="Median Deal Valuation", value=fmt_usd(b.get('median_deal_valuation')))
                    st.metric(label="Avg Deal Valuation", value=fmt_usd(b.get('avg_deal_valuation')))
            with col2:
                 with st.container(border=True):
                    st.metric(label="Median Valuation / Revenue", value=fmt_mult(b.get('median_valuation_revenue_multiple')))
                    st.metric(label="Median Valuation / Profit", value=fmt_mult(b.get('median_valuation_profit_multiple')))
            with col3:
                with st.container(border=True):
                    st.metric(label="Median Equity Pledged", value=fmt_pct(b.get('median_equity_percentage')))
                    st.metric(label="Avg Equity Pledged", value=fmt_pct(b.get('avg_equity_percentage')))

            st.markdown("---")
            col_back, col_next = st.columns([1, 6]) # Adjust button column ratios
            with col_back:
                 if st.button("Back", key="back_step1", type="secondary"): # Use secondary style
                    st.session_state.valuation_step = 0
                    # Keep valuation data like name/industry/revenue if user goes back
                    st.session_state.benchmarks = None # Clear benchmarks
                    st.session_state.comparables = None
                    st.rerun()
            with col_next:
                if st.button("Next: Add Financial Details", key="next_step1", use_container_width=True):
                    st.session_state.valuation_step = 2
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

     # --- Step 2: Collect Remaining Financials ---
    elif st.session_state.valuation_step == 2:
        st.progress(2 / 4)
        st.markdown("##### Step 3 of 4: Additional Financial Details")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Provide more financial context")
        st.caption("(Optional, but improves valuation analysis)")

        earnings = st.number_input(
            "Annual Earnings/Net Profit (USD)?",
             value=st.session_state.valuation_data.get('earnings', 0.0), # Retain value
             step=1000.0, format="%.2f", key="val_earnings",
             help="Enter Net Income after expenses & taxes (can be negative)."
        )
        assets = st.number_input(
            "Total Assets value (USD)?",
            min_value=0.0, step=1000.0, format="%.2f", key="val_assets",
            value=st.session_state.valuation_data.get('assets', 0.0), # Retain value
            help="Total value of everything the company owns."
        )
        liabilities = st.number_input(
            "Total Liabilities (USD)?",
            min_value=0.0, step=1000.0, format="%.2f", key="val_liabilities",
             value=st.session_state.valuation_data.get('liabilities', 0.0), # Retain value
            help="Total of all debts and obligations the company owes."
        )
        growth = st.select_slider(
             "Estimate Annual Growth Rate (Next 1-3 Years)",
             options=["<0% (Declining)", "0-10% (Low)", "10-25% (Moderate)", "25-50% (High)", ">50% (Very High)"],
             key="val_growth", value=st.session_state.valuation_data.get('growth', "10-25% (Moderate)"),
             help="Estimate revenue growth rate over the next 1-3 years."
        )

        st.markdown("---")
        col_back, col_calc = st.columns([1, 6]) # Adjust ratios
        with col_back:
             if st.button("Back", key="back_step2", type="secondary"):
                 # Go back to benchmark display only if benchmarks were successfully calculated
                 if st.session_state.benchmarks and st.session_state.benchmarks.get("count", 0) > 2:
                     st.session_state.valuation_step = 1
                 else: # Otherwise, go back to the start
                     st.session_state.valuation_step = 0
                 st.rerun()
        with col_calc:
             if st.button("Calculate Valuation", key="calc_val", use_container_width=True):
                # Store collected data
                st.session_state.valuation_data['earnings'] = earnings
                st.session_state.valuation_data['assets'] = assets
                st.session_state.valuation_data['liabilities'] = liabilities
                st.session_state.valuation_data['growth'] = growth
                st.session_state.valuation_step = 3 # Move to results
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 3: Calculate and Display Valuation + Comparables ---
    elif st.session_state.valuation_step == 3:
        st.progress(3 / 4)
        st.markdown("##### Step 4 of 4: Valuation Analysis")

        # --- Summary Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user_company_name = st.session_state.valuation_data.get('company_name', 'Your Company')
        st.markdown(f"### Analysis Summary for: **{user_company_name}**")

        data = st.session_state.valuation_data
        benchmarks = st.session_state.benchmarks

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Your Input Data:**")
            st.markdown(f"- Industry: `{data.get('industry', 'N/A')}`")
            st.markdown(f"- Revenue: `${safe_float(data.get('revenue')):,.0f}`") # Cleaned format
            st.markdown(f"- Earnings: `${safe_float(data.get('earnings')):,.0f}`")
            st.markdown(f"- Assets: `${safe_float(data.get('assets')):,.0f}`")
            st.markdown(f"- Liabilities: `${safe_float(data.get('liabilities')):,.0f}`")
            st.markdown(f"- Growth: `{data.get('growth', 'N/A')}`")
        with col2:
             st.markdown("**Dataset Benchmarks:**")
             if benchmarks and benchmarks.get("count", 0) > 2:
                # Use helper formats defined earlier
                st.markdown(f"_(Based on {benchmarks['count']} comparable deals)_")
                st.markdown(f"- Median Deal Val: `{fmt_usd(benchmarks.get('median_deal_valuation'))}`")
                st.markdown(f"- Median Val/Revenue: `{fmt_mult(benchmarks.get('median_valuation_revenue_multiple'))}`")
                st.markdown(f"- Median Val/Profit: `{fmt_mult(benchmarks.get('median_valuation_profit_multiple'))}`")
             else:
                 st.markdown("_Insufficient comparable deals found in dataset for specific benchmarks._")
        st.markdown("</div>", unsafe_allow_html=True) # Close Summary Card

        # --- AI Analysis Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### AI-Powered Valuation Analysis")
        st.caption("Based on your inputs and dataset benchmarks (where available).")

        # Construct prompt for Gemini
        user_revenue = safe_float(data.get('revenue', 0))
        user_earnings = safe_float(data.get('earnings', 0))
        user_assets = safe_float(data.get('assets', 0))
        user_liabilities = safe_float(data.get('liabilities', 0))

        prompt_context = f"""
Analyze the potential valuation for the company: "{user_company_name}".

User Company Data:
- Industry: {data.get('industry', 'N/A')}
- Annual Revenue: {user_revenue:,.2f} USD
- Annual Earnings/Profit: {user_earnings:,.2f} USD
- Total Assets: {user_assets:,.2f} USD
- Total Liabilities: {user_liabilities:,.2f} USD
- Estimated Growth Rate Category: {data.get('growth', 'N/A')}

Comparable Deals Dataset Benchmarks (Derived from investor pitch show data for '{data.get('industry', 'N/A')}'):
"""
        if benchmarks and benchmarks.get("count", 0) > 2:
             def format_benchmark(key, format_str="{:,.2f}x", usd_format="${:,.0f}", pct_format="{:.1f}%"):
                 val = benchmarks.get(key)
                 if val is None or val == 0: return "N/A"
                 if "valuation" in key and "multiple" not in key: return usd_format.format(val)
                 if "equity" in key: return pct_format.format(val)
                 if isinstance(val, (int, float)): return format_str.format(val)
                 return "N/A"

             prompt_context += f"""
- Number of Comparable Deals Found: {benchmarks['count']}
- Median Deal Valuation: {format_benchmark('median_deal_valuation')}
- Average Deal Valuation: {format_benchmark('avg_deal_valuation')}
- Median Valuation/Revenue Multiple: {format_benchmark('median_valuation_revenue_multiple')}
- Average Valuation/Revenue Multiple: {format_benchmark('avg_valuation_revenue_multiple')}
- Median Valuation/Profit Multiple: {format_benchmark('median_valuation_profit_multiple')}
- Average Valuation/Profit Multiple: {format_benchmark('avg_valuation_profit_multiple')}
- Median Equity Percentage Given: {format_benchmark('median_equity_percentage')}
- Average Equity Percentage Given: {format_benchmark('avg_equity_percentage')}
"""
        else:
            prompt_context += "- No specific or sufficient comparable deals benchmarks available from the dataset for this industry. Rely more on standard methods and general industry knowledge, stating this limitation.\n"

        prompt_context += """
Please provide a detailed valuation analysis following the instructions in your system prompt. Focus on using the MEDIAN dataset benchmarks for calculations when available and sensible. Provide a final recommended valuation range and justification.
        """

        # Get valuation result from Gemini
        with st.spinner("🧠 Analyzing valuation with AI..."):
            st.progress(4 / 4)
            valuation_result = gemini_qna(prompt_context, is_valuation=True)

        # Display result
        st.markdown(valuation_result)
        st.markdown("</div>", unsafe_allow_html=True) # Close AI Analysis Card


        # --- Comparables Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Comparable Deal Details")
        st.caption("Examples from the dataset for context (max 5 shown).")

        if 'comparables' in st.session_state and st.session_state.comparables:
            # Filter comparables slightly - maybe exclude those with zero deal val?
            valid_comparables = [c for c in st.session_state.comparables if get_deal_valuation(c) > 0]

            if not valid_comparables:
                 st.info(f"No comparable deals with complete data found for '{data.get('industry', 'N/A')}' to display details.")
            else:
                comparables_to_show = valid_comparables[:5] # Show top 5 valid ones
                st.markdown(f"_Displaying details from up to {len(comparables_to_show)} comparable deals in '{data.get('industry', 'N/A')}'._")

                for i, comp_deal in enumerate(comparables_to_show):
                    comp_name = comp_deal.get("business_basics", {}).get("business_name", "Unknown Company")
                    with st.expander(f"{comp_name}"):
                        try:
                            ask_amt = safe_float(comp_deal.get("pitch_metrics", {}).get("initial_ask_amount"))
                            ask_eq = safe_float(comp_deal.get("pitch_metrics", {}).get("equity_offered"))
                            ask_val = safe_float(comp_deal.get("pitch_metrics", {}).get("implied_valuation")) # Requested Val

                            final_terms_dict = get_displayable_final_terms(comp_deal)
                            final_amt = final_terms_dict['amount']
                            final_eq = final_terms_dict['equity']
                            final_loan = final_terms_dict['loan']
                            final_val = get_deal_valuation(comp_deal) # Actual Deal Val

                            usp = comp_deal.get("product_service_information", {}).get("unique_selling_proposition", "Not specified.")

                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Pitch Ask:**")
                                st.write(f"- Request: `{fmt_usd(ask_amt)}` for `{fmt_pct(ask_eq)}`")
                                st.write(f"- Implied Val: `{fmt_usd(ask_val)}`")
                            with c2:
                                st.markdown("**Deal Terms (On Air):**")
                                deal_str = f"- Invest: `{fmt_usd(final_amt)}` for `{fmt_pct(final_eq)}`"
                                if final_loan > 0:
                                    deal_str += f" (+ `{fmt_usd(final_loan)}` Loan)"
                                st.write(deal_str)
                                st.write(f"- Deal Val: `{fmt_usd(final_val)}`")

                            st.markdown("**Unique Selling Prop:**")
                            st.markdown(f"> _{usp}_")

                        except Exception as e:
                            st.warning(f"Could not display full details for {comp_name}: {e}")
        else:
            st.info(f"No comparable deal data loaded.")

        st.markdown("</div>", unsafe_allow_html=True) # Close Comparables Card

        # --- Reset Button ---
        if st.button("Start New Valuation", key="reset_val", use_container_width=True):
            # Reset state variables for valuation page
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.session_state.benchmarks = None
            st.session_state.comparables = None
            # Clear potentially large cached items if needed, though Streamlit handles some caching
            # get_comparable_deals.clear()
            # calculate_sharktank_benchmarks.clear()
            st.rerun()

# 3. Interactive Business Assessment
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")
    
    # Initialize session state variables if they don't exist
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False
    if 'assessment_responses' not in st.session_state:
        st.session_state.assessment_responses = {}
    
    # Maximum number of questions to ask
    max_questions = 15
    
    # Display progress
    st.progress(min(1.0, st.session_state.question_count / max_questions))
    
    # Check if assessment is not completed and under max questions
    if not st.session_state.assessment_completed and st.session_state.question_count < max_questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Generate initial question if we're just starting
        if st.session_state.question_count == 0:
            initial_question = "Tell me about your business and what problem you're solving."
            st.session_state.current_question = initial_question
        
        # Display current question
        st.markdown(f"### Question {st.session_state.question_count + 1} of {max_questions}")
        st.markdown(f"**{st.session_state.current_question}**")
        
        # Get user response
        response = st.text_area("Your Answer", height=100, key=f"q_{st.session_state.question_count}")
        
        if st.button("Submit Answer", use_container_width=True):
            # Save response to session state
            st.session_state.assessment_responses[st.session_state.current_question] = response
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "question": st.session_state.current_question,
                "answer": response
            })
            
            # Increment question counter
            st.session_state.question_count += 1
            
            # Check if we've reached max questions
            if st.session_state.question_count >= max_questions:
                st.session_state.assessment_completed = True
                st.rerun()
            
            # Generate next question based on the conversation history
            with st.spinner("Analyzing your response and preparing next question..."):
                # Format conversation history for the AI
                conversation_context = "\n\n".join([
                    f"Q: {exchange['question']}\nA: {exchange['answer']}"
                    for exchange in st.session_state.conversation_history
                ])
                
                # Prompt for the next question
                next_question_prompt = f"""
                You are an expert business analyst and investor interviewer. 
                You've been conducting an assessment with a business owner and need to ask the next most relevant question.
                
                Here's the conversation history so far:
                
                {conversation_context}
                
                Based on these responses, what is the single most important next question to ask?
                The question should help you better understand a critical aspect of their business that hasn't been fully explored yet.
                
                Please provide only the next question, without any additional text or explanation.
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
                
                # Get next question from AI
                next_question = gemini_qna(next_question_prompt).strip()
                st.session_state.current_question = next_question
            
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show results if assessment is completed
    elif st.session_state.assessment_completed:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Business Assessment Results")
        
        # Format conversation history for analysis
        assessment_data = "\n".join([
            f"Q: {q}\nA: {a}" 
            for q, a in st.session_state.assessment_responses.items() if a
        ])
        
        # Comprehensive analysis prompt
        analysis_prompt = f"""
        Expert Business Investor Assessment System
        
        You are an expert business analyst and investor interviewer, combining the analytical precision of Kevin O'Leary, 
        the technical insight of Mark Cuban, and the strategic vision of top investors from "Shark Tank" and "Dragon's Den"
        while maintaining a professional, neutral tone.
        
        Based on the following interview with a business owner, provide a comprehensive assessment of their business:
        
        {assessment_data}
        
        Your analysis should include:
        
        1. Business Profile Summary
           - Company overview and value proposition
           - Market opportunity assessment
           - Competitive positioning
           - Team evaluation
           - Business model analysis
        
        2. SWOT Analysis
           - Strengths
           - Weaknesses
           - Opportunities
           - Threats
        
        3. Financial Assessment
           - Revenue and profitability evaluation
           - Growth trajectory
           - Unit economics (if applicable)
           - Capital efficiency
        
        4. Valuation Considerations
           - Appropriate valuation methodologies
           - Key value drivers and detractors
           - Reasonable valuation range (if enough information is available)
        
        5. Strategic Recommendations
           - Growth opportunities
           - Risk mitigation strategies
           - Suggested next steps
           - Investment considerations
        
        6. Overall Rating (1-10)
           - Provide a numerical rating with justification
        
        Format your response with clear headings and bullet points for readability.
        If there are critical gaps in the information provided, note these as areas requiring further investigation.
        """
        
        # Generate comprehensive business assessment
        with st.spinner("Generating comprehensive business assessment report..."):
            analysis_result = gemini_qna(analysis_prompt)
        
        # Display analysis result
        st.markdown(analysis_result)
        
        # Option to start a new assessment
        if st.button("Start New Assessment", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.question_count = 0
            st.session_state.assessment_completed = False
            st.session_state.assessment_responses = {}
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub © 2025 | Powered by Gemini AI |  
</div>
""", unsafe_allow_html=True)

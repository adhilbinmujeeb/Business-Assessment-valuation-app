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
import json

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
model = genai.GenerativeModel('gemini-pro')

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

def gemini_qna(query, context=None, assess_confidence=False):
    try:
        context_str = f"Context: {context}" if context else "No specific context provided."
        
        if assess_confidence:
            system_prompt_with_confidence = """
            Analyze the conversation history and assess:
            1. How well we understand the business (0-1 confidence score)
            2. What critical areas still need exploration
            3. Whether we have enough information for a meaningful assessment
            
            Return in JSON format:
            {
                "confidence_score": float,
                "missing_areas": [string],
                "has_sufficient_info": boolean,
                "next_question": string
            }
            """
            
            response = model.generate_content(
                f"{system_prompt_with_confidence}\n\n{context_str}\n\nAssess current understanding and determine next question: {query}"
            )
            
            return json.loads(response.text)
            
        else:
            system_prompt = """
            Expert Business Investor Interview System
            System Role Definition
            You are an expert business analyst and investor interviewer, combining the analytical precision of Kevin O'Leary, 
            the technical insight of Mark Cuban, and the strategic vision of top investors from "Shark Tank" and "Dragon's Den"
            while maintaining a professional, neutral tone.
            """
            
            response = model.generate_content(
                f"{system_prompt}\n\n{context_str}\n\nQuery: {query}"
            )
            return response.text
            
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

def assess_business_complexity(initial_response):
    """Analyze initial business description to determine complexity factors"""
    complexity_prompt = """
    Analyze the business description and determine its complexity level based on:
    1. Industry complexity
    2. Business model complexity
    3. Operational scale
    4. Regulatory requirements
    5. Market dynamics
    6. Organizational structure
    
    Return JSON:
    {
        "complexity_score": float (0-1),
        "recommended_min_questions": int,
        "recommended_max_questions": int,
        "key_areas": [string],
        "justification": string
    }
    """
    
    result = gemini_qna(
        query=complexity_prompt,
        context=initial_response,
        assess_confidence=True
    )
    
    return result

def get_required_categories(business_type, complexity_score):
    """Determine which question categories are essential for this business"""
    base_categories = ["Business Fundamentals", "Financial Performance"]
    
    additional_categories = {
        "high_complexity": [
            "Risk Assessment",
            "International Expansion",
            "Regulatory Compliance",
            "Crisis Management"
        ],
        "medium_complexity": [
            "Market Position",
            "Operations",
            "Team & Leadership"
        ],
        "low_complexity": [
            "Growth & Scaling",
            "Basic Operations"
        ]
    }
    
    if complexity_score > 0.7:
        return base_categories + additional_categories["high_complexity"]
    elif complexity_score > 0.4:
        return base_categories + additional_categories["medium_complexity"]
    else:
        return base_categories + additional_categories["low_complexity"]

def should_continue_assessment(state):
    """Determine if assessment should continue based on multiple factors"""
    # Confidence threshold varies by complexity
    confidence_threshold = 0.9 if state["complexity_score"] > 0.7 else 0.8
    
    # Check coverage of required categories
    categories_covered = len(set(state["covered_categories"]))
    required_categories = len(state["required_categories"])
    category_coverage = categories_covered / required_categories if required_categories > 0 else 1.0
    
    # Define stopping criteria
    criteria = {
        "confidence_met": state["assessment_confidence"] >= confidence_threshold,
        "category_coverage": category_coverage >= 0.9,
        "min_questions_asked": state["question_count"] >= state["min_questions"],
        "max_questions_not_exceeded": state["question_count"] < state["max_questions"]
    }
    
    # Log assessment progress
    st.write("Assessment Progress:", criteria)
    
    # Continue if we haven't met confidence OR haven't covered categories
    # But stop if we hit max questions
    return (
        (not criteria["confidence_met"] or not criteria["category_coverage"]) and
        criteria["max_questions_not_exceeded"]
    )

def reset_assessment():
    st.session_state.assessment_state = {
        "question_count": 0,
        "assessment_confidence": 0.0,
        "complexity_score": 0.0,
        "required_categories": [],
        "covered_categories": [],
        "min_questions": 5,
        "max_questions": 30,
        "responses": {}
    }

def run_dynamic_assessment():
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")
    
    # Initialize assessment state if not exists
    if "assessment_state" not in st.session_state:
        reset_assessment()
    
    state = st.session_state.assessment_state
    
    # Display current assessment stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Asked", state["question_count"])
    with col2:
        st.metric("Understanding Level", f"{state['assessment_confidence']:.0%}")
    with col3:
        remaining = state["max_questions"] - state["question_count"]
        st.metric("Questions Remaining", f"Up to {remaining}")
    
    # First question to understand business
    if state["question_count"] == 0:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Tell me about your business and what problem you're solving.")
        
        initial_response = st.text_area("Your Answer", height=100, key="initial_response")
        
        if st.button("Submit", use_container_width=True):
            if initial_response:
                # Analyze business complexity
                complexity_analysis = assess_business_complexity(initial_response)
                if complexity_analysis:
                    state["complexity_score"] = complexity_analysis["complexity_score"]
                    state["min_questions"] = complexity_analysis["recommended_min_questions"]
                    state["max_questions"] = complexity_analysis["recommended_max_questions"]
                    state["required_categories"] = get_required_categories(
                        business_type="",  # You can add business type detection here
                        complexity_score=state["complexity_score"]
                    )
                    state["responses"]["initial"] = initial_response
                    state["question_count"] += 1
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Continue assessment
    elif should_continue_assessment(state):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Get next question and confidence assessment
        conversation_history = "\n".join([
            f"Q: {q}\nA: {a}" 
            for q, a in state["responses"].items()
        ])
        
        result = gemini_qna(
            query="Determine next question based on current understanding",
            context=conversation_history,
            assess_confidence=True
        )
        
        if result:
            # Update confidence score
            state["assessment_confidence"] = result["confidence_score"]
            
            # Display missing areas if any
            if result["missing_areas"]:
                st.info("Areas needing exploration: " + ", ".join(result["missing_areas"]))
            
            # Display current question
            st.markdown(f"### Question {state['question_count'] + 1}")
            st.markdown(f"**{result['next_question']}**")
            
            # Get user response
            response = st.text_area("Your Answer", height=100, key=f"q_{state['question_count']}")
            
            if st.button("Submit Answer", use_container_width=True):
                # Save response
                state["responses"][result["next_question"]] = response
                state["question_count"] += 1
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Assessment complete
        show_assessment_results(state)

def show_assessment_results(state):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Business Assessment Results")
    
    # Generate comprehensive analysis
    conversation_history = "\n".join([
        f"Q: {q}\nA: {a}" 
        for q, a in state["responses"].items()
    ])
    
    analysis_result = gemini_qna(
        query="Generate comprehensive business assessment",
        context=conversation_history
    )
    
    if analysis_result:
        st.markdown(analysis_result)
    
    # Option to continue assessment if confidence is still low
    if state["assessment_confidence"] < 0.8:
        if st.button("Continue Assessment for More Detailed Analysis"):
            state["max_questions"] += 5  # Allow more questions
            st.rerun()
    
    # Option to start new assessment
    if st.button("Start New Assessment", use_container_width=True):
        reset_assessment()
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

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

# Main Content
if "Company Valuation" in page:
    # Company Valuation section code (unchanged from original)
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

    if 'valuation_step' not in st.session_state:
        st.session_state.valuation_step = 0
    if 'valuation_data' not in st.session_state:
        st.session_state.valuation_data = {}

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
            valuation_result = gemini_qna(valuation_prompt)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Valuation Results")
        st.markdown(valuation_result)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start New Valuation", use_container_width=True):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.rerun()

elif "Business Assessment" in page:
    run_dynamic_assessment()

# Footer
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub © 2025 | Powered by Gemini AI |  
</div>
""", unsafe_allow_html=True) 

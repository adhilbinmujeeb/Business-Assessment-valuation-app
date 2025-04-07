import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
# from scipy.spatial.distance import cosine # Not used in relevant sections
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import statistics # <<< Added Import

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Basic Error Handling for Env Variables ---
if not MONGO_URI:
    st.error("🚨 Missing MONGO_URI environment variable.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("🚨 Missing GEMINI_API_KEY environment variable.")
    st.stop()

# --- Database and Collection Names ---
DATABASE_NAME = "business_rag"  # Or your specific DB name
LISTINGS_COLLECTION_NAME = "business_listings" # Collection with Shark Tank/Pitch Data
QUESTIONS_COLLECTION_NAME = "questions" # If used elsewhere
BUSINESS_ATTRIBUTES_COLLECTION_NAME = "business_attributes" # If used elsewhere

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS --- (Keep Existing)
st.markdown("""
<style>
    /* --- Keep existing CSS --- */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    h1, h2, h3 { color: #1E3A8A; }
    h1 { text-align: center; margin-bottom: 1.5rem; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 6px; padding: 0.6rem 1.2rem; font-weight: 500; border: none; transition: background-color 0.2s ease; }
    .stButton>button:hover { background-color: #2563EB; }
    .stButton button[kind="secondary"] { background-color: #D1D5DB; color: #374151; }
    .stButton button[kind="secondary"]:hover { background-color: #9CA3AF; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 2px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { height: 3rem; white-space: pre-wrap; border-radius: 4px 4px 0 0; padding: 0.5rem 1rem; background-color: #F1F5F9; color: #374151; border: 1px solid #E2E8F0; border-bottom: none; margin-bottom: -2px; transition: background-color 0.2s ease, color 0.2s ease; }
    .stTabs [aria-selected="true"] { background-color: #1E3A8A !important; color: white !important; border-color: #1E3A8A; }
    div[data-testid="stSidebar"] { background-color: #F8FAFC; padding-top: 1.5rem; }
    .card { background-color: #FFFFFF; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-card { background-color: #EFF6FF; border-radius: 0.5rem; padding: 1rem 1.2rem; text-align: center; border: 1px solid #BFDBFE; height: 100%; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1E3A8A; }
    .metric-label { font-size: 0.9rem; color: #475569; margin-top: 0.3rem; }
    .sidebar-header { padding: 0.5rem 1rem; margin-bottom: 1rem; border-bottom: 1px solid #E2E8F0; }
    .streamlit-expanderHeader { font-weight: 600; color: #1E3A8A; background-color: #EFF6FF; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- MongoDB Connection ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DATABASE_NAME]
    # Define all potential collections
    business_collection = db[BUSINESS_ATTRIBUTES_COLLECTION_NAME] # Original attributes
    question_collection = db[QUESTIONS_COLLECTION_NAME]        # Questions
    listings_collection = db[LISTINGS_COLLECTION_NAME]    # Pitch data
    # Test connection once
    if 'mongo_connected' not in st.session_state:
        client.server_info()
        st.session_state.mongo_connected = True
        # st.toast("🔌 Connected to MongoDB", icon="✅") # Optional: Use toast
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()

# --- Gemini API Setup ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use a standard, generally available model unless the specific one is required and available
    gemini_model = genai.GenerativeModel('gemini-pro')
    # Optional test (can slow startup)
    # if 'gemini_tested' not in st.session_state:
    #     gemini_model.generate_content("Test", generation_config=genai.types.GenerationConfig(max_output_tokens=5))
    #     st.session_state.gemini_tested = True
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# ===================================================
# <<< --- START: ADDED/REPLACED HELPER FUNCTIONS --- >>>
# ===================================================

def safe_float(value, default=0.0):
    """Safely converts a value to float, handling common variations."""
    if value is None or value == "not_provided" or str(value).strip() == '':
        return default
    try:
        str_value = str(value).lower()
        str_value = str_value.replace("$", "").replace(",", "").replace("usd","").strip()
        multiplier = 1
        if 'k' in str_value[-2:]: # Check last two chars for k/m/b to avoid mid-word letters
            str_value = str_value.replace('k', '')
            multiplier = 1e3
        elif 'm' in str_value[-7:] and 'million' in str_value: # More specific check
             str_value = str_value.replace('million', '').replace('m','')
             multiplier = 1e6
        elif 'm' in str_value[-2:]:
             str_value = str_value.replace('m','')
             multiplier = 1e6
        elif 'b' in str_value[-7:] and 'billion' in str_value:
             str_value = str_value.replace('billion','').replace('b', '')
             multiplier = 1e9
        elif 'b' in str_value[-2:]:
             str_value = str_value.replace('b','')
             multiplier = 1e9

        # Allow negative sign only at the beginning
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x == '.' or (x == '-' and str_value.strip().startswith('-')), str_value.strip()))
        if cleaned_value and cleaned_value != '-':
             return float(cleaned_value) * multiplier
        else:
             return default
    except (ValueError, TypeError):
        return default

@st.cache_data(ttl=3600)
def get_comparable_deals(industry, _listings_collection):
    """Queries MongoDB for successful deals in a given industry."""
    if not industry or not _listings_collection:
        return []
    try:
        # Use regex to match industry case-insensitively and allow partial matches if needed
        # Escape special regex characters in industry name if necessary (e.g., C++)
        safe_industry_regex = pymongo.common.regex_escape(industry)
        query = {
            "business_basics.industry_category": {"$regex": safe_industry_regex, "$options": "i"},
            "deal_outcome.final_result": "deal"
        }
        projection = {
            "_id": 0,
            "business_basics.business_name": 1,
            "business_metrics.basic_metrics.revenue": 1,
            "business_metrics.basic_metrics.profit": 1,
            "questions.new_metrics_revealed": 1,
            "pitch_metrics.initial_ask_amount": 1,
            "pitch_metrics.equity_offered": 1,
            "pitch_metrics.implied_valuation": 1,
            "deal_outcome.final_terms": 1,
            "pitch_metrics.final_terms": 1,
            "product_service_information.unique_selling_proposition": 1
        }
        comparables = list(_listings_collection.find(query, projection).limit(100)) # Limit results
        return comparables
    except Exception as e:
        st.error(f"Error querying MongoDB for comparables: {e}")
        return []

def get_profit_from_pitch(pitch_data):
    """Extracts profit figure, searching multiple potential fields."""
    basic_profit = pitch_data.get("business_metrics", {}).get("basic_metrics", {}).get("profit")
    if basic_profit and str(basic_profit).strip().lower() not in ["not_provided", "", "n/a"]:
        profit_val = safe_float(basic_profit)
        if profit_val != 0 or "0" in str(basic_profit): return profit_val # Return if meaningful

    questions = pitch_data.get("questions", [])
    for q in questions:
        metrics = q.get("new_metrics_revealed", [])
        for metric in metrics:
            metric_lower = str(metric).lower()
            if ("profit" in metric_lower or "earnings" in metric_lower or "net income" in metric_lower) and any(char.isdigit() for char in metric_lower):
                extracted_num_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or (x == '-' and metric_lower.strip().startswith('-')), metric_lower))
                profit_val = safe_float(extracted_num_str)
                if profit_val != 0 or "0" in extracted_num_str: return profit_val # Return first meaningful found
    return 0.0 # Default

def get_deal_valuation(pitch_data):
    """Extracts agreed deal valuation robustly, prioritizing final terms."""
    final_terms = pitch_data.get("deal_outcome", {}).get("final_terms")
    if not final_terms: final_terms = pitch_data.get("pitch_metrics", {}).get("final_terms")

    if final_terms and isinstance(final_terms, dict):
        investment = safe_float(final_terms.get("investment_amount", 0.0))
        if investment == 0 and "amount" in final_terms: investment = safe_float(final_terms.get("amount"))
        deal_equity = safe_float(final_terms.get("equity"))
        if investment > 0 and deal_equity > 0:
            return investment / (deal_equity / 100.0)

    # Fallback to implied if no deal terms valuation calculable (less accurate)
    implied_val = safe_float(pitch_data.get("pitch_metrics", {}).get("implied_valuation"))
    if implied_val > 0: return implied_val

    return 0.0 # Default

def get_displayable_final_terms(pitch_data):
    """Extracts display-friendly final deal terms (amount, equity, loan)."""
    final_terms_data = {'amount': 0.0, 'equity': 0.0, 'loan': 0.0}
    final_terms = pitch_data.get("deal_outcome", {}).get("final_terms")
    if not final_terms: final_terms = pitch_data.get("pitch_metrics", {}).get("final_terms")

    if final_terms and isinstance(final_terms, dict):
        investment = safe_float(final_terms.get("investment_amount", 0.0))
        if investment == 0 and "amount" in final_terms: investment = safe_float(final_terms.get("amount"))
        final_terms_data['amount'] = investment
        final_terms_data['equity'] = safe_float(final_terms.get("equity"))
        final_terms_data['loan'] = safe_float(final_terms.get("loan"))
    return final_terms_data

@st.cache_data(ttl=3600)
def calculate_sharktank_benchmarks(_comparables):
    """Calculates benchmark metrics from a list of comparable deals."""
    if not _comparables: return {"count": 0}
    deal_valuations, revenues, profits, v_rev_mults, v_prof_mults, equities = [], [], [], [], [], []
    for deal in _comparables:
        try:
            deal_val = get_deal_valuation(deal)
            if deal_val <= 0: continue
            revenue = safe_float(deal.get("business_metrics", {}).get("basic_metrics", {}).get("revenue"))
            profit = get_profit_from_pitch(deal)
            equity = get_displayable_final_terms(deal).get('equity')

            deal_valuations.append(deal_val)
            if revenue > 0:
                revenues.append(revenue)
                v_rev_mults.append(deal_val / revenue)
            if profit != 0: # Allow positive or negative profit for averaging, but not 0
                 profits.append(profit)
                 if profit > 0: v_prof_mults.append(deal_val / profit) # Only calculate for positive profit
            if equity is not None and equity > 0: equities.append(equity)
        except Exception: continue # Skip deal on error

    benchmarks = {
        "count": len(deal_valuations),
        "avg_deal_valuation": np.nanmean(deal_valuations) if deal_valuations else 0,
        "median_deal_valuation": statistics.median(deal_valuations) if deal_valuations else 0,
        "avg_revenue": np.nanmean(revenues) if revenues else 0,
        "median_revenue": statistics.median(revenues) if revenues else 0,
        "avg_profit": np.nanmean(profits) if profits else 0,
        "median_profit": statistics.median(profits) if profits else 0,
        "avg_valuation_revenue_multiple": np.nanmean(v_rev_mults) if v_rev_mults else 0,
        "median_valuation_revenue_multiple": statistics.median(v_rev_mults) if v_rev_mults else 0,
        "avg_valuation_profit_multiple": np.nanmean(v_prof_mults) if v_prof_mults else 0,
        "median_valuation_profit_multiple": statistics.median(v_prof_mults) if v_prof_mults else 0,
        "avg_equity_percentage": np.nanmean(equities) if equities else 0,
        "median_equity_percentage": statistics.median(equities) if equities else 0,
    }
    for key in benchmarks: # Replace NaNs
        if isinstance(benchmarks[key], (float, np.number)) and np.isnan(benchmarks[key]): benchmarks[key] = 0
    return benchmarks


# << REPLACED gemini_qna >>
def gemini_qna(prompt, is_valuation=False):
    """Sends prompt to Gemini and handles potential errors.
       Uses different system prompts based on the task."""
    try:
        if is_valuation:
             # Valuation-specific system prompt
             system_prompt = """
You are an expert business analyst specializing in early-stage company valuation, referencing a dataset of investor pitches (like Shark Tank).

**Instructions:**

1.  **Analyze User Data:** Review the provided details for the user's company (Name, Industry, Revenue, Earnings, Assets, Liabilities, Growth).
2.  **Evaluate Dataset Benchmarks:** Critically consider the provided benchmark metrics (Avg/Median Deal Valuation, Valuation/Revenue Multiple, Valuation/Profit Multiple, Equity %) derived from comparable deals in the investor pitch dataset. Note the source implies potential negotiation dynamics, not purely academic valuations.
3.  **Apply Multiple Valuation Methods:**
    *   **Market-Based (using Dataset Benchmarks):** THIS IS KEY. Apply the dataset's *Median* Valuation/Revenue and *Median* Valuation/Profit multiples to the user's figures. *Calculate* these results explicitly. Compare these to the dataset's *Median Deal Valuation* as a direct sanity check. Discuss the suitability of these benchmarks given their source.
    *   **Income-Based (Standard Methods as secondary):** If user provided *positive* profit/earnings, calculate a standard P/E valuation *only if dataset benchmarks are unavailable or unreliable*. Mention typical P/E ranges for the broader industry if known, but emphasize dataset benchmarks when available. DCF is generally *not applicable* here without reliable multi-year projections from the user.
    *   **Asset-Based:** Calculate Book Value (Assets - Liabilities) if data is available. Briefly comment on its relevance (usually low for early-stage, asset-light businesses typical of these benchmarks).
4.  **Synthesize & Recommend:**
    *   Compare results from the different methods, especially contrasting the Dataset Benchmark approach with others. Explain potential reasons for discrepancies (e.g., dataset reflects actual deals including risk/negotiation).
    *   Provide a **Recommended Valuation Range**. Base this *primarily* on the dataset benchmark calculations (Median Multiple * User Metric; Median Deal Valuation range) if benchmarks are available and sensible. Justify the range.
    *   Briefly list factors (beyond provided metrics) that influence valuation (Team, IP, Market Traction, Competition), noting they are assessed qualitatively.

**Output Format:**
*   Clear headings (e.g., "Valuation using Dataset Revenue Multiple", "Book Value Calculation", "Valuation Summary").
*   Show key calculations (e.g., "Median Dataset Revenue Multiple (X.Xx) * Your Revenue ($Y) = $Z").
*   Explicitly state the derived value/range for each relevant method.
*   Final "Valuation Summary and Recommendation" section with the suggested range and clear justification linked back to the methods, prioritizing the dataset benchmarks.
*   Use Markdown for formatting (bolding, bullets).
            """
        else:
            # General/Assessment system prompt (Can be overridden by more specific prompts later)
            system_prompt = """
You are an expert business analyst and investor interviewer (like those on Shark Tank/Dragon's Den).
Your goal is to assess a business or provide strategic insights based on the user's input or an ongoing Q&A.
Maintain a professional, insightful, and constructively critical tone. Ask clarifying questions if needed.
Provide actionable feedback, identify strengths/weaknesses, and evaluate potential based on the information given.
Adapt your response style based on whether you are asking a question, providing analysis, or summarizing an assessment.
            """

        # Simple Prompt Structure (User query + implicit system context)
        # Note: Gemini Pro API might infer system role better than older models.
        # For more complex history, pass a list of {'role':'user/model', 'parts':[{'text':...}]}
        full_prompt_content = f"{system_prompt}\n\nUser Query/Data:\n{prompt}"

        response = gemini_model.generate_content(
            full_prompt_content,
            generation_config=genai.types.GenerationConfig(
                # temperature=0.5 if is_valuation else 0.7 # Optional: adjust temp
            ),
            safety_settings=[ # Standard safety settings
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        # Access text response safely
        return response.text
    except genai.types.BlockedPromptException as bpe:
        st.error(f"❌ Gemini Error: Prompt blocked by safety filters. {bpe}")
        return "Response blocked by API safety filters."
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return "Failed to get response from AI due to an error."

# << ADDED get_unique_industries >>
@st.cache_data(ttl=3600)
def get_unique_industries(_listings_collection):
    """Fetches unique industry categories from the listings collection.
       Removed $toTitle for broader MongoDB compatibility."""
    fallback_industries = ["Select an Industry...", "Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Healthcare", "Food & Beverage", "Education", "Consumer Product", "Apparel", "Service", "Technology", "Other"]
    try:
        pipeline = [
            {"$match": {"business_basics.industry_category": {"$exists": True, "$ne": None, "$type": "array", "$ne": [] }}},
            {"$unwind": "$business_basics.industry_category"},
            {"$addFields": {"normalized_industry": {"$trim": {"input": "$business_basics.industry_category"}}}},
            {"$match": {"normalized_industry": {"$ne": ""}}},
            {"$group": {"_id": "$normalized_industry"}},
            {"$sort": {"_id": 1}},
            {"$limit": 500}
        ]
        results = list(_listings_collection.aggregate(pipeline))
        industries = sorted([doc["_id"] for doc in results if doc["_id"]]) # Sort here
        if industries:
             return ["Select an Industry..."] + industries
        else:
             st.warning("Could not fetch dynamic industry list, using fallback.")
             return fallback_industries
    except pymongo.errors.OperationFailure as e:
        st.error(f"MongoDB Aggregation Error fetching industries: {e.details.get('errmsg', e)}")
        return fallback_industries
    except Exception as e:
        st.error(f"General Error fetching industries from MongoDB: {e}")
        return fallback_industries

# --- REMOVED get_business, get_all_businesses --- (as they weren't used in the new flow)

# ===================================================
# <<< --- END: ADDED/REPLACED HELPER FUNCTIONS --- >>>
# ===================================================


# --- Sidebar Navigation --- (Keep Existing)
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Insights Hub") # Slightly shorter title
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("Choose a tool:", [
        "💰 Company Valuation",
        "📊 Business Assessment",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Valuation benchmarks derived from investor pitch data.")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)


# --- Session State Initialization --- (Keep Existing)
if 'valuation_step' not in st.session_state: st.session_state.valuation_step = 0
if 'valuation_data' not in st.session_state: st.session_state.valuation_data = {}
if 'benchmarks' not in st.session_state: st.session_state.benchmarks = None
if 'comparables' not in st.session_state: st.session_state.comparables = None
# Assessment state
if 'assessment_step' not in st.session_state: st.session_state.assessment_step = 0
if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
if 'assessment_responses' not in st.session_state: st.session_state.assessment_responses = {}
if 'assessment_question_count' not in st.session_state: st.session_state.assessment_question_count = 0
if 'assessment_completed' not in st.session_state: st.session_state.assessment_completed = False
if 'current_assessment_question' not in st.session_state: st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."


# --- Load industries once ---
if 'available_industries' not in st.session_state:
     st.session_state.available_industries = get_unique_industries(listings_collection)

# --- REMOVED sample_query logic as it wasn't used in the new flow ---


# ===================================================
# <<< --- START: REPLACED VALUATION PAGE LOGIC --- >>>
# ===================================================
# --- 💰 COMPANY VALUATION PAGE LOGIC ---
if page == "💰 Company Valuation":
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using standard methods benchmarked against real pitch data.")

    # --- Step 0: Collect Basic Info (Name, Industry, Revenue) ---
    if st.session_state.valuation_step == 0:
        st.progress(0 / 4) # Display progress for the 4-step process
        st.markdown("##### Step 1 of 4: Basic Information")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Tell us about your company")

        # Use company name from state if going back
        company_name = st.text_input(
            "What is your Company Name?",
            key="val_company_name",
            placeholder="Enter your company name here",
            value=st.session_state.valuation_data.get('company_name', ''), # Retain value if going back
            help="The legal or trading name of your business."
        )

        # Use the industries list stored in session state
        current_industry_value = st.session_state.valuation_data.get('industry')
        default_index_industry = 0 # Default to 'Select...'
        if current_industry_value in st.session_state.available_industries:
             default_index_industry = st.session_state.available_industries.index(current_industry_value)

        industry = st.selectbox(
            "What industry best describes your company?",
            st.session_state.available_industries,
            key="val_industry",
            index=default_index_industry,
            help="Select the closest matching industry from the list."
        )
        # Use revenue from state if going back
        revenue = st.number_input(
            "What is your company's most recent Annual Revenue (USD)?",
            min_value=0.0,
            step=1000.0,
            format="%.2f", # Use float format
            key="val_revenue",
            value=st.session_state.valuation_data.get('revenue', 0.0), # Retain value
            help="Total sales/turnover over the last 12 months."
        )

        # Use columns for better button layout (optional)
        _, btn_col = st.columns([3, 1]) # Adjust ratio as needed
        with btn_col:
            if st.button("Next: Get Benchmarks", key="get_benchmarks_btn", use_container_width=True):
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
                    # Reset potentially stale data
                    st.session_state.benchmarks = None
                    st.session_state.comparables = None

                    with st.spinner(f"Finding comparable deals for '{industry}'..."):
                        time.sleep(0.5) # UI polish for spinner
                        # Assume listings_collection is defined and connected
                        st.session_state.comparables = get_comparable_deals(industry, listings_collection)

                        if st.session_state.comparables:
                             st.session_state.benchmarks = calculate_sharktank_benchmarks(st.session_state.comparables)
                             # Check if benchmarks are sufficient (e.g., at least 3 deals)
                             if st.session_state.benchmarks and st.session_state.benchmarks.get("count", 0) > 2:
                                st.session_state.valuation_step = 1 # Move to show benchmarks
                                st.rerun()
                             else:
                                count = st.session_state.benchmarks.get("count", 0) if st.session_state.benchmarks else 0
                                st.warning(f"Found only {count} comparable deal(s) for '{industry}'. Proceeding without specific dataset benchmarks.")
                                st.session_state.benchmarks = None # Ensure benchmarks are cleared
                                st.session_state.valuation_step = 2 # Skip benchmark display
                                st.rerun()
                        else:
                             st.warning(f"Could not find comparable deals for '{industry}' in the dataset. Proceeding without dataset benchmarks.")
                             st.session_state.benchmarks = None
                             st.session_state.valuation_step = 2 # Skip benchmark display
                             st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 1: Display Benchmarks (if found and valid) ---
    elif st.session_state.valuation_step == 1:
        # Guard against direct navigation or invalid state
        if not st.session_state.benchmarks or st.session_state.benchmarks.get("count", 0) <= 2:
            st.warning("Benchmarks not available or insufficient. Proceeding to financial details entry.")
            st.session_state.valuation_step = 2
            time.sleep(1)
            st.rerun()
        else:
            st.progress(1 / 4)
            st.markdown("##### Step 2 of 4: Dataset Benchmarks")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            b = st.session_state.benchmarks
            industry_name = st.session_state.valuation_data.get('industry','N/A')
            st.markdown(f"### Benchmarks from {b['count']} Deals in '{industry_name}'")
            st.caption("_Note: Based on data from investor pitches. Reflects negotiated outcomes._")

            # Reusable formatters for metrics
            fmt_usd = lambda value: f"${value:,.0f}" if isinstance(value, (int, float)) and value else "N/A"
            fmt_mult = lambda value: f"{value:.2f}x" if isinstance(value, (int, float)) and value else "N/A"
            fmt_pct = lambda value: f"{value:.1f}%" if isinstance(value, (int, float)) and value is not None else "N/A" # Handle potential None from median

            col1, col2, col3 = st.columns(3)
            with col1:
                 with st.container(border=True):
                    st.metric(label="Median Deal Valuation", value=fmt_usd(b.get('median_deal_valuation')))
                    st.metric(label="Avg Deal Valuation", value=fmt_usd(b.get('avg_deal_valuation')))
            with col2:
                 with st.container(border=True):
                    st.metric(label="Median Val / Revenue", value=fmt_mult(b.get('median_valuation_revenue_multiple')))
                    st.metric(label="Avg Val / Revenue", value=fmt_mult(b.get('avg_valuation_revenue_multiple'))) # Show avg too
            with col3:
                 with st.container(border=True):
                    st.metric(label="Median Val / Profit", value=fmt_mult(b.get('median_valuation_profit_multiple')))
                    st.metric(label="Avg Val / Profit", value=fmt_mult(b.get('avg_valuation_profit_multiple'))) # Show avg too

            # Add equity benchmarks in an expander for less clutter
            with st.expander("Equity Benchmarks"):
                eq_col1, eq_col2 = st.columns(2)
                with eq_col1:
                    st.metric(label="Median Equity Pledged", value=fmt_pct(b.get('median_equity_percentage')))
                with eq_col2:
                     st.metric(label="Avg Equity Pledged", value=fmt_pct(b.get('avg_equity_percentage')))


            st.markdown("---") # Divider before buttons
            col_back, col_next = st.columns([1, 6])
            with col_back:
                 if st.button("Back", key="back_step1", type="secondary"):
                    st.session_state.valuation_step = 0
                    # Keep basic info but clear benchmarks
                    st.session_state.benchmarks = None
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

        # Allow negative earnings
        earnings = st.number_input(
            "Annual Earnings / Net Profit (USD)?",
             value=st.session_state.valuation_data.get('earnings', 0.0), # Retain value
             step=1000.0,
             format="%.2f", # Use float for potential losses
             key="val_earnings",
             help="Enter Net Income after expenses & taxes (can be negative)."
        )
        assets = st.number_input(
            "Total Assets value (USD)?",
            min_value=0.0, step=1000.0, format="%.2f", key="val_assets",
            value=st.session_state.valuation_data.get('assets', 0.0), # Retain value
            help="Total value of everything the company owns (e.g., cash, equipment, inventory, receivables)."
        )
        liabilities = st.number_input(
            "Total Liabilities (USD)?",
            min_value=0.0, step=1000.0, format="%.2f", key="val_liabilities",
             value=st.session_state.valuation_data.get('liabilities', 0.0), # Retain value
            help="Total of all debts and obligations the company owes (e.g., loans, accounts payable)."
        )
        # Provide default value for slider
        default_growth = st.session_state.valuation_data.get('growth', "10-25% (Moderate)")
        growth_options = ["<0% (Declining)", "0-10% (Low)", "10-25% (Moderate)", "25-50% (High)", ">50% (Very High)"]
        if default_growth not in growth_options: default_growth="10-25% (Moderate)" # Ensure default exists

        growth = st.select_slider(
             "Estimate Annual Growth Rate (Next 1-3 Years)",
             options=growth_options,
             key="val_growth", value=default_growth,
             help="Estimate revenue growth rate over the next 1-3 years."
        )

        st.markdown("---") # Divider before buttons
        col_back, col_calc = st.columns([1, 6])
        with col_back:
             if st.button("Back", key="back_step2", type="secondary"):
                 # Navigate back correctly based on whether benchmarks were shown
                 if st.session_state.benchmarks and st.session_state.benchmarks.get("count", 0) > 2:
                     st.session_state.valuation_step = 1 # Go back to benchmark display
                 else:
                     st.session_state.valuation_step = 0 # Go back to start
                 st.rerun()
        with col_calc:
             if st.button("Calculate Valuation", key="calc_val", use_container_width=True):
                # Store collected data from this step
                st.session_state.valuation_data['earnings'] = earnings
                st.session_state.valuation_data['assets'] = assets
                st.session_state.valuation_data['liabilities'] = liabilities
                st.session_state.valuation_data['growth'] = growth
                st.session_state.valuation_step = 3 # Move to results
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 3: Calculate and Display Valuation + Comparables ---
    elif st.session_state.valuation_step == 3:
        st.progress(3 / 4) # Nearing completion
        st.markdown("##### Step 4 of 4: Valuation Analysis & Comparables")

        # --- Summary Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user_company_name = st.session_state.valuation_data.get('company_name', 'Your Company')
        st.markdown(f"### Analysis Summary for: **{user_company_name}**")

        data = st.session_state.valuation_data
        benchmarks = st.session_state.benchmarks
        # Define formatters within scope
        fmt_usd_summary = lambda value: f"${safe_float(value):,.0f}"
        fmt_pct_summary = lambda value: f"{value:.1f}%" if isinstance(value, (int, float)) else "N/A"
        fmt_mult_summary = lambda value: f"{value:.2f}x" if isinstance(value, (int, float)) and value != 0 else "N/A"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Your Input Data:**")
            st.markdown(f"- Industry: `{data.get('industry', 'N/A')}`")
            st.markdown(f"- Revenue: `{fmt_usd_summary(data.get('revenue'))}`")
            st.markdown(f"- Earnings: `{fmt_usd_summary(data.get('earnings'))}`")
            st.markdown(f"- Assets: `{fmt_usd_summary(data.get('assets'))}`")
            st.markdown(f"- Liabilities: `{fmt_usd_summary(data.get('liabilities'))}`")
            st.markdown(f"- Growth Est: `{data.get('growth', 'N/A')}`")
        with col2:
             st.markdown("**Dataset Benchmarks:**")
             if benchmarks and benchmarks.get("count", 0) > 2:
                st.markdown(f"_(From {benchmarks['count']} deals)_")
                st.markdown(f"- Median Deal Val: `{fmt_usd_summary(benchmarks.get('median_deal_valuation'))}`")
                st.markdown(f"- Median Val/Revenue: `{fmt_mult_summary(benchmarks.get('median_valuation_revenue_multiple'))}`")
                st.markdown(f"- Median Val/Profit: `{fmt_mult_summary(benchmarks.get('median_valuation_profit_multiple'))}`")
             else:
                 st.markdown("_Insufficient data for specific benchmarks._")
        st.markdown("</div>", unsafe_allow_html=True) # Close Summary Card

        # --- AI Analysis Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### AI-Powered Valuation Analysis")
        st.caption("Based on your inputs and dataset benchmarks (where available).")

        # Construct prompt context string for Gemini AI
        prompt_context = f"""
Analyze the potential valuation for the company: "{user_company_name}".

User Company Data:
- Industry: {data.get('industry', 'N/A')}
- Annual Revenue: {safe_float(data.get('revenue', 0)):,.2f} USD
- Annual Earnings/Profit: {safe_float(data.get('earnings', 0)):,.2f} USD
- Total Assets: {safe_float(data.get('assets', 0)):,.2f} USD
- Total Liabilities: {safe_float(data.get('liabilities', 0)):,.2f} USD
- Estimated Growth Rate Category: {data.get('growth', 'N/A')}

Comparable Deals Dataset Benchmarks (Derived from investor pitch show data for '{data.get('industry', 'N/A')}'):
"""
        if benchmarks and benchmarks.get("count", 0) > 2:
             # Helper to format benchmarks for the prompt
             def format_bench_for_prompt(key, usd_format="${:,.0f}", mult_format="{:.2f}x", pct_format="{:.1f}%"):
                 val = benchmarks.get(key)
                 if val is None: return "N/A"
                 try:
                     # Select format based on key naming convention
                     if "valuation" in key and "multiple" not in key: return usd_format.format(val)
                     if "equity" in key: return pct_format.format(val)
                     if isinstance(val, (int, float)): return mult_format.format(val) # Default to multiple
                 except (ValueError, TypeError): return "N/A"
                 return "N/A" # Fallback

             prompt_context += f"""
- Number of Comparable Deals: {benchmarks['count']}
- Median Deal Valuation: {format_bench_for_prompt('median_deal_valuation')}
- Average Deal Valuation: {format_bench_for_prompt('avg_deal_valuation')}
- Median Val/Revenue Multiple: {format_bench_for_prompt('median_valuation_revenue_multiple')}
- Average Val/Revenue Multiple: {format_bench_for_prompt('avg_valuation_revenue_multiple')}
- Median Val/Profit Multiple: {format_bench_for_prompt('median_valuation_profit_multiple')}
- Average Val/Profit Multiple: {format_bench_for_prompt('avg_valuation_profit_multiple')}
- Median Equity Percentage Given: {format_bench_for_prompt('median_equity_percentage')}
- Average Equity Percentage Given: {format_bench_for_prompt('avg_equity_percentage')}
"""
'''
        else:
            prompt_context += "- No specific or sufficient comparable deals benchmarks available from the dataset for this industry. Rely more on standard methods and general industry knowledge, stating this limitation.\n"

        prompt_context += """
Please provide a detailed valuation analysis following the instructions in your system prompt. Focus on using the MEDIAN dataset benchmarks for calculations when available and sensible. Provide a final recommended valuation range and justification using Markdown formatting.
        """

        # --- Call Gemini AI ---
        with st.spinner("🧠 Analyzing valuation with AI... Please wait."):
            # Use the gemini_qna function defined earlier
            valuation_result = gemini_qna(prompt_context, is_valuation=True)
            st.progress(4 / 4) # Mark as complete after AI returns

        # --- Display AI Result ---
        if "Failed to get response" in valuation_result or "blocked by API" in valuation_result:
            st.error(valuation_result) # Display AI errors clearly
        else:
            st.markdown(valuation_result) # Display successful AI response

        st.markdown("</div>", unsafe_allow_html=True) # Close AI Analysis Card


        # --- Comparables Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Comparable Deal Details")
        st.caption("Contextual examples from the dataset (max 5 shown).")

        if 'comparables' in st.session_state and st.session_state.comparables:
            # Filter for deals with valid deal valuation for display
            valid_comparables = [c for c in st.session_state.comparables if get_deal_valuation(c) > 0]

            if not valid_comparables:
                 st.info(f"No comparable deals with displayable final terms found for '{data.get('industry', 'N/A')}' in the dataset.")
            else:
                comparables_to_show = valid_comparables[:5] # Show top 5 valid ones
                st.markdown(f"_Displaying details from up to {len(comparables_to_show)} comparable deals in '{data.get('industry', 'N/A')}'._")

                # Use formatters defined above for consistency
                fmt_usd = lambda value: f"${value:,.0f}" if isinstance(value, (int, float)) and value else "N/A"
                fmt_pct = lambda value: f"{value:.1f}%" if isinstance(value, (int, float)) and value is not None else "N/A"

                for i, comp_deal in enumerate(comparables_to_show):
                    comp_name = comp_deal.get("business_basics", {}).get("business_name", "Unknown Company")
                    # Use unique keys for expanders within loops
                    with st.expander(f"{comp_name}", key=f"comp_expander_{i}"):
                        try:
                            ask_amt = safe_float(comp_deal.get("pitch_metrics", {}).get("initial_ask_amount"))
                            ask_eq = safe_float(comp_deal.get("pitch_metrics", {}).get("equity_offered"))
                            ask_val = safe_float(comp_deal.get("pitch_metrics", {}).get("implied_valuation"))

                            final_terms_dict = get_displayable_final_terms(comp_deal)
                            final_amt = final_terms_dict['amount']
                            final_eq = final_terms_dict['equity']
                            final_loan = final_terms_dict['loan']
                            final_val = get_deal_valuation(comp_deal) # Recalculate

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
            # This case shouldn't be reached if steps are followed, but good failsafe
            st.info(f"Comparable deal data not available.")

        st.markdown("</div>", unsafe_allow_html=True) # Close Comparables Card

        # --- Reset Button ---
        if st.button("Start New Valuation", key="reset_val_final", use_container_width=True):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.session_state.benchmarks = None
            st.session_state.comparables = None
            # Clear caches if implemented with @st.cache_data or similar
            # get_comparable_deals.clear() # Be cautious clearing cache if functions are widely used
            # calculate_sharktank_benchmarks.clear()
            # get_unique_industries.clear()
            st.rerun()
# ===================================================
# <<< --- END: REPLACED VALUATION PAGE LOGIC --- >>>
# ===================================================


# --- 📊 BUSINESS ASSESSMENT PAGE LOGIC --- (Keep Existing)
elif page == "📊 Business Assessment":
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Engage in an AI-driven Q&A to analyze your business, similar to an investor pitch.")

    max_assessment_questions = 10 # Max questions for this assessment

    # --- Assessment Q&A Flow ---
    if not st.session_state.assessment_completed:
        st.progress(st.session_state.assessment_question_count / max_assessment_questions)
        st.markdown(f"##### Question {st.session_state.assessment_question_count + 1} of {max_assessment_questions}")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.current_assessment_question}**")

        # Unique key for text area ensures it resets visually when question changes
        response = st.text_area("Your Answer:", height=120, key=f"assess_answer_{st.session_state.assessment_question_count}")

        # Button to submit answer
        if st.button("Submit Answer & Get Next Question", key=f"submit_assess_{st.session_state.assessment_question_count}", use_container_width=True):
            if response and response.strip():
                # Store response paired with question
                st.session_state.assessment_responses[st.session_state.current_assessment_question] = response.strip()
                # Update conversation history for AI context
                st.session_state.conversation_history.append({"role": "model", "parts": [{"text": st.session_state.current_assessment_question}]})
                st.session_state.conversation_history.append({"role": "user", "parts": [{"text": response.strip()}]})

                st.session_state.assessment_question_count += 1

                # Check if assessment complete
                if st.session_state.assessment_question_count >= max_assessment_questions:
                    st.session_state.assessment_completed = True
                    st.success("Assessment Q&A complete! Generating analysis...")
                    time.sleep(1) # Brief pause
                    st.rerun()
                else:
                    # Generate next question
                    with st.spinner("🧠 Thinking of the next question..."):
                        system_prompt_next_q = """
You are an expert business analyst and investor interviewer. Based on the conversation history provided, ask the SINGLE most important and relevant next question to assess the business's viability, strategy, or financials. Do not add any conversational filler, just output the question itself. Consider aspects like market, competition, financials, team, operations, and growth plans. Prioritize areas that seem weak, unclear, or unaddressed.
                        """
                        history_for_prompt = "\n".join([f"{turn['role'].capitalize()}: {turn['parts'][0]['text']}" for turn in st.session_state.conversation_history])
                        next_q_prompt = f"{system_prompt_next_q}\n\n**Conversation History:**\n{history_for_prompt}\n\n**Next Question:**"

                        next_question_text = gemini_qna(next_q_prompt, is_valuation=False) # Not a valuation task
                        cleaned_next_q = next_question_text.strip().strip('"').strip("'").replace("Next Question:","").strip() # Clean potential AI artifacts

                        if cleaned_next_q and "?" in cleaned_next_q: # Basic check
                            st.session_state.current_assessment_question = cleaned_next_q
                        else: # Fallback
                            fallback_qs = ["What are your main revenue streams?", "Who are your top 3 competitors?", "What is your customer acquisition strategy?", "What are the biggest risks facing your business?"]
                            st.session_state.current_assessment_question = fallback_qs[st.session_state.assessment_question_count % len(fallback_qs)] # Cycle through fallbacks
                            st.warning("AI failed to generate next question clearly, using fallback.")

                    st.rerun() # Display the new question
            else:
                 st.warning("Please provide an answer before submitting.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Assessment Results Display ---
    elif st.session_state.assessment_completed:
        st.success("✅ Business Assessment Complete")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## AI-Generated Business Analysis")
        st.caption("Based on your answers during the assessment.")

        # Format conversation for the analysis prompt
        assessment_data_for_prompt = "\n\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in st.session_state.assessment_responses.items()
        ])

        # Detailed system prompt for the final analysis
        analysis_system_prompt = """
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

        analysis_prompt = f"{analysis_system_prompt}\n\n**Q&A Session Transcript:**\n{assessment_data_for_prompt}\n\n**Provide Comprehensive Analysis:**"

        with st.spinner("📊 Generating comprehensive business analysis report..."):
             analysis_result = gemini_qna(analysis_prompt, is_valuation=False) # Not a valuation task

        # Display analysis result
        if "Failed to get response" in analysis_result or "blocked by API" in analysis_result:
            st.error(analysis_result)
        else:
            st.markdown(analysis_result)

        # Reset button for assessment
        if st.button("Start New Assessment", key="reset_assess_final", use_container_width=True):
            st.session_state.assessment_step = 0
            st.session_state.conversation_history = []
            st.session_state.assessment_responses = {}
            st.session_state.assessment_question_count = 0
            st.session_state.assessment_completed = False
            st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# --- Footer --- (Keep Existing)
st.markdown("---") # Visual separator before footer
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub | Powered by Gemini AI & MongoDB | Benchmarks derived from public pitch data.
</div>
""", unsafe_allow_html=True)

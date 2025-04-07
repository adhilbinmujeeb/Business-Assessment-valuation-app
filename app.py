import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
# from scipy.spatial.distance import cosine # Removed as it's not used in the provided sections
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import statistics # Added for median calculation
import json # Added for potential use, though not strictly necessary for insertion code

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Basic Error Handling for Env Variables ---
if not MONGO_URI:
    st.error("🚨 Missing MONGO_URI environment variable. Please configure it in your .env file or environment.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("🚨 Missing GEMINI_API_KEY environment variable. Please configure it in your .env file or environment.")
    st.stop()

# --- Database and Collection Names ---
DATABASE_NAME = "business_rag"  # Or your specific DB name
LISTINGS_COLLECTION_NAME = "business_listings" # Collection with Shark Tank/Pitch Data
QUESTIONS_COLLECTION_NAME = "questions" # If used elsewhere

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Constrain width slightly for better readability on wide screens */
    }
    h1, h2, h3 {
        color: #1E3A8A; /* Dark Blue */
    }
    h1 {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #1E3A8A; /* Dark Blue */
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB; /* Brighter Blue */
    }
    .stButton button[kind="secondary"] { /* Style the back button */
         background-color: #D1D5DB; /* Light Gray */
         color: #374151; /* Dark Gray Text */
    }
     .stButton button[kind="secondary"]:hover {
         background-color: #9CA3AF; /* Medium Gray */
     }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        background-color: #F1F5F9; /* Very Light Gray */
        color: #374151; /* Dark Gray Text */
        border: 1px solid #E2E8F0;
        border-bottom: none;
        margin-bottom: -2px; /* Align with border */
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important; /* Dark Blue */
        color: white !important;
        border-color: #1E3A8A;
    }
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC; /* Lighter than Card bg */
        padding-top: 1.5rem;
    }
    .card {
        background-color: #FFFFFF; /* White card background */
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem; /* Increased spacing */
        border: 1px solid #E2E8F0; /* Lighter border */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    .metric-card {
        background-color: #EFF6FF; /* Very Light Blue */
        border-radius: 0.5rem;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #BFDBFE; /* Light Blue Border */
        height: 100%; /* Make cards in a row same height */
    }
    .metric-value {
        font-size: 1.8rem; /* Larger value */
        font-weight: bold;
        color: #1E3A8A; /* Dark Blue */
    }
    .metric-label {
        font-size: 0.9rem;
        color: #475569; /* Slightly Darker Gray */
        margin-top: 0.3rem;
    }
    .sidebar-header {
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
    /* Custom Expander Header */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E3A8A;
        background-color: #EFF6FF; /* Light blue background */
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- MongoDB Connection ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Added timeout
    db = client[DATABASE_NAME]
    listings_collection = db[LISTINGS_COLLECTION_NAME]
    # Optional: Test connection only once
    if 'mongo_connected' not in st.session_state:
        client.server_info() # Test connection
        st.session_state.mongo_connected = True
        # st.toast("🔌 Connected to MongoDB", icon="✅") # Use toast for less intrusive message
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()

# --- Gemini API Setup ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro') # Use a standard reliable model
    # Test generation once if needed, but can cause startup delays
    # if 'gemini_tested' not in st.session_state:
    #    gemini_model.generate_content("Test", generation_config=genai.types.GenerationConfig(max_output_tokens=5))
    #    st.session_state.gemini_tested = True
    #    st.toast("✨ Connected to Gemini AI", icon="🧠")
except Exception as e:
    st.error(f"Failed to configure or connect to Gemini API: {e}")
    st.stop()


# ===============================
# --- Helper Functions ---
# ===============================

def safe_float(value, default=0.0):
    """Safely converts a value to float, handling common variations."""
    if value is None or value == "not_provided" or str(value).strip() == '':
        return default
    try:
        str_value = str(value).lower()
        str_value = str_value.replace("$", "").replace(",", "").replace("usd","").strip()
        # Handle suffixes like 'k', 'm', 'b'
        if 'k' in str_value:
            str_value = str_value.replace('k', '')
            multiplier = 1e3
        elif 'million' in str_value or 'm' in str_value:
             str_value = str_value.replace('million', '').replace('m','')
             multiplier = 1e6
        elif 'billion' in str_value or 'b' in str_value:
             str_value = str_value.replace('billion','').replace('b', '')
             multiplier = 1e9
        else:
             multiplier = 1
        # Remove any remaining non-numeric characters except decimal point and potential negative sign at start
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x == '.' or (x == '-' and str_value.startswith('-')), str_value))
        if cleaned_value: # Ensure not empty after cleaning
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
        query = {
            # Match industry category using regex for flexibility and case-insensitivity
            "business_basics.industry_category": {"$regex": f".*{industry.replace('&', '&')}.*", "$options": "i"}, # More robust regex
            "deal_outcome.final_result": "deal"
        }
        # Fetch necessary fields for benchmarking and display
        projection = {
            "_id": 0, # Exclude Mongo's internal ID
            "business_basics.business_name": 1,
            "business_metrics.basic_metrics.revenue": 1,
            "business_metrics.basic_metrics.profit": 1,
            "questions.new_metrics_revealed": 1, # To extract profit if needed
            "pitch_metrics.initial_ask_amount": 1,
            "pitch_metrics.equity_offered": 1,
            "pitch_metrics.implied_valuation": 1,
            "deal_outcome.final_terms": 1, # Preferred source for final deal
            "pitch_metrics.final_terms": 1, # Fallback source
            "product_service_information.unique_selling_proposition": 1
        }
        # Adding a limit just in case of performance issues with very large results, adjust as needed
        comparables = list(_listings_collection.find(query, projection).limit(100))
        return comparables
    except Exception as e:
        st.error(f"Error querying MongoDB for comparables: {e}")
        return []


def get_profit_from_pitch(pitch_data):
    """Extracts profit figure, searching multiple potential fields."""
    # Check direct basic_metrics field
    basic_profit = pitch_data.get("business_metrics", {}).get("basic_metrics", {}).get("profit")
    if basic_profit and str(basic_profit).strip().lower() != "not_provided":
        profit_val = safe_float(basic_profit)
        # Ensure it's a meaningful number, not just 0 from safe_float default
        if profit_val != 0 or "0" in str(basic_profit):
            return profit_val

    # Search in questions->new_metrics_revealed as fallback
    questions = pitch_data.get("questions", [])
    for q in questions:
        metrics = q.get("new_metrics_revealed", [])
        for metric in metrics:
            metric_lower = str(metric).lower()
            # Look for variations of profit keywords and numeric presence
            if ("profit" in metric_lower or "earnings" in metric_lower) and any(char.isdigit() for char in metric_lower):
                 # Attempt to parse number from the metric string
                extracted_num_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or (x == '-' and metric_lower.strip().startswith('-')), metric_lower))
                profit_val = safe_float(extracted_num_str)
                if profit_val != 0 or "0" in extracted_num_str:
                    return profit_val

    return 0.0 # Default if not found


def get_deal_valuation(pitch_data):
    """Extracts agreed deal valuation robustly, prioritizing final terms."""
    deal_amount = 0.0
    deal_equity = 0.0

    # Prioritize deal_outcome terms
    final_terms = pitch_data.get("deal_outcome", {}).get("final_terms")
    # Fallback to pitch_metrics terms if deal_outcome missing (less accurate)
    if not final_terms:
        final_terms = pitch_data.get("pitch_metrics", {}).get("final_terms")

    if final_terms and isinstance(final_terms, dict):
        investment = safe_float(final_terms.get("investment_amount"))
        # Check older 'amount' key if investment_amount missing
        if investment == 0 and "amount" in final_terms:
            investment = safe_float(final_terms.get("amount"))

        deal_equity = safe_float(final_terms.get("equity"))

        # Calculate valuation if equity is non-zero
        if investment > 0 and deal_equity > 0:
            return investment / (deal_equity / 100.0)

    # Extreme fallback: use requested valuation (least accurate for *deal* benchmark)
    implied = pitch_data.get("pitch_metrics", {}).get("implied_valuation")
    if implied:
         val = safe_float(implied)
         if val > 0: return val

    return 0.0 # Default


def get_displayable_final_terms(pitch_data):
    """Extracts display-friendly final deal terms (amount, equity, loan)."""
    final_terms_data = {'amount': 0.0, 'equity': 0.0, 'loan': 0.0}
    # Prioritize deal_outcome terms
    final_terms = pitch_data.get("deal_outcome", {}).get("final_terms")
    if not final_terms: # Fallback
        final_terms = pitch_data.get("pitch_metrics", {}).get("final_terms")

    if final_terms and isinstance(final_terms, dict):
        investment = safe_float(final_terms.get("investment_amount"))
        if investment == 0 and "amount" in final_terms:
            investment = safe_float(final_terms.get("amount"))
        final_terms_data['amount'] = investment
        final_terms_data['equity'] = safe_float(final_terms.get("equity"))
        final_terms_data['loan'] = safe_float(final_terms.get("loan"))

    return final_terms_data


@st.cache_data(ttl=3600)
def calculate_sharktank_benchmarks(_comparables):
    """Calculates benchmark metrics from a list of comparable deals."""
    deal_valuations = []
    revenues = []
    profits = []
    valuation_revenue_multiples = []
    valuation_profit_multiples = []
    equity_percentages = []

    if not _comparables: # Handle empty list early
        return {"count": 0}

    for deal in _comparables:
        try:
            # --- Calculate Metrics for Each Deal ---
            # Deal Valuation
            deal_val = get_deal_valuation(deal)
            if deal_val <= 0: continue # Skip if no valid deal valuation

            # Revenue
            revenue = safe_float(deal.get("business_metrics", {}).get("basic_metrics", {}).get("revenue"))
            # Profit
            profit = get_profit_from_pitch(deal)
            # Equity Pledged
            equity_terms = get_displayable_final_terms(deal)
            equity = equity_terms.get('equity')

            # --- Append to Lists ---
            deal_valuations.append(deal_val)
            if revenue > 0:
                revenues.append(revenue)
                valuation_revenue_multiples.append(deal_val / revenue)
            if profit > 0: # Use > 0 for profit multiple calc
                 profits.append(profit)
                 valuation_profit_multiples.append(deal_val / profit)
            if equity is not None and equity > 0:
                equity_percentages.append(equity)

        except Exception as e:
            # Log or warn about deals causing errors if needed for debugging
            # st.warning(f"Skipping deal calculation due to error: {deal.get('business_basics',{}).get('business_name','Unknown')} - {e}")
            continue # Skip this deal safely

    # --- Aggregate Benchmarks ---
    # Use numpy for safe mean calculation (handles empty lists -> nan)
    # Use statistics for median (handles empty lists -> error, so check)
    benchmarks = {
        "count": len(deal_valuations), # Base count on successfully processed valuations
        "avg_deal_valuation": np.nanmean(deal_valuations) if deal_valuations else 0,
        "median_deal_valuation": statistics.median(deal_valuations) if deal_valuations else 0,
        "avg_revenue": np.nanmean(revenues) if revenues else 0,
        "median_revenue": statistics.median(revenues) if revenues else 0, # Added median revenue
        "avg_profit": np.nanmean(profits) if profits else 0,
        "median_profit": statistics.median(profits) if profits else 0, # Added median profit
        "avg_valuation_revenue_multiple": np.nanmean(valuation_revenue_multiples) if valuation_revenue_multiples else 0,
        "median_valuation_revenue_multiple": statistics.median(valuation_revenue_multiples) if valuation_revenue_multiples else 0,
        "avg_valuation_profit_multiple": np.nanmean(valuation_profit_multiples) if valuation_profit_multiples else 0,
        "median_valuation_profit_multiple": statistics.median(valuation_profit_multiples) if valuation_profit_multiples else 0,
        "avg_equity_percentage": np.nanmean(equity_percentages) if equity_percentages else 0,
        "median_equity_percentage": statistics.median(equity_percentages) if equity_percentages else 0,
    }

    # Replace potential NaN results with 0 for cleaner display
    for key in benchmarks:
        if isinstance(benchmarks[key], (float, np.number)) and np.isnan(benchmarks[key]):
             benchmarks[key] = 0

    return benchmarks


def gemini_qna(prompt, is_valuation=False):
    """Sends prompt to Gemini and handles potential errors.
       Uses different system prompts based on the task."""
    try:
        if is_valuation:
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
        else: # Assuming assessment-related prompts for other cases
            system_prompt = """
You are an expert business analyst and investor interviewer (like those on Shark Tank/Dragon's Den).
Your goal is to assess a business or provide strategic insights based on the user's input or an ongoing Q&A.
Maintain a professional, insightful, and constructively critical tone. Ask clarifying questions if needed.
Provide actionable feedback, identify strengths/weaknesses, and evaluate potential based on the information given.
Adapt your response style based on whether you are asking a question, providing analysis, or summarizing an assessment.
            """ # Use the detailed assessment system prompts defined within the Assessment page logic itself when needed.

        # Prepare content object for Gemini API
        # Handle case where prompt is already a list (for conversational history)
        if isinstance(prompt, list):
             content_payload = prompt
             # Inject system prompt if not already present as the first 'system' instruction
             # (This part might need refinement based on how assessment history is structured)
        else:
             # Default: Simple user query + system instruction
             content_payload = [
                  {'role': 'user', 'parts': [{'text': system_prompt}]}, # Use system role if API supports it
                  {'role': 'user', 'parts': [{'text': prompt}]}        # Or combine into user prompt
             ]


        response = gemini_model.generate_content(
            content_payload,
            generation_config=genai.types.GenerationConfig(
                # Adjust temperature for creativity vs. factuality if needed (e.g., 0.5 for valuation)
                # temperature=0.5 if is_valuation else 0.7
            ),
            # Add safety settings if facing blocking issues
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        return response.text
    except genai.types.BlockedPromptException as bpe:
        st.error("❌ Gemini Error: The prompt was blocked. Please rephrase or check input. Details: " + str(bpe))
        return "Response blocked by API safety filters."
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        # print(f"--- PROMPT DEBUG ---\n{prompt}\n--- END PROMPT ---") # Uncomment for detailed debugging
        return "Failed to get response from AI due to an error."


@st.cache_data(ttl=3600)
def get_unique_industries(_listings_collection):
    """Fetches unique industry categories from the listings collection."""
    try:
        pipeline = [
            # Ensure industry_category exists and is an array
            {"$match": {"business_basics.industry_category": {"$exists": True, "$ne": None, "$type": "array"}}},
            {"$unwind": "$business_basics.industry_category"}, # Unwind the array
            # Normalize: Trim whitespace and convert to title case for consistency
            {"$addFields": {"normalized_industry": {"$toTitle": {"$trim": {"input": "$business_basics.industry_category"}}}}},
            # Group by the normalized industry name
            {"$group": {"_id": "$normalized_industry"}},
            {"$sort": {"_id": 1}}, # Sort alphabetically
            # Limit the number of distinct industries if it becomes too large
            {"$limit": 500}
        ]
        industries = [doc["_id"] for doc in _listings_collection.aggregate(pipeline) if doc["_id"]] # Filter out None/empty
        if industries:
             return ["Select an Industry..."] + industries
        else:
             # Fallback if aggregation fails or returns nothing
             st.warning("Could not fetch dynamic industry list, using fallback.")
             return ["Select an Industry...", "Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Healthcare", "Food & Beverage", "Education", "Consumer Product", "Apparel", "Service", "Technology", "Other"]
    except Exception as e:
        st.error(f"Could not fetch industries from MongoDB: {e}")
        # Return fallback list on error
        return ["Select an Industry...", "Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Healthcare", "Food & Beverage", "Education", "Consumer Product", "Apparel", "Service", "Technology", "Other"]


# ===============================
# --- Sidebar Navigation ---
# ===============================
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Insights Hub")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("Choose a tool:", [
        "💰 Company Valuation",
        "📊 Business Assessment",
    ], label_visibility="collapsed") # Hide the label

    st.markdown("---")
    # Optional: Add data source info or links
    st.caption("Valuation benchmarks derived from investor pitch data.")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)


# ===============================
# --- Session State Initialization ---
# ===============================
# General state setup
if 'current_page' not in st.session_state:
    st.session_state.current_page = "💰 Company Valuation" # Default page

# Valuation page state
if 'valuation_step' not in st.session_state:
    st.session_state.valuation_step = 0
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'benchmarks' not in st.session_state:
    st.session_state.benchmarks = None
if 'comparables' not in st.session_state:
    st.session_state.comparables = None

# Assessment page state
if 'assessment_step' not in st.session_state: # Added for consistency
    st.session_state.assessment_step = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'assessment_responses' not in st.session_state: # Combined with history mostly
    st.session_state.assessment_responses = {}
if 'assessment_question_count' not in st.session_state:
     st.session_state.assessment_question_count = 0
if 'assessment_completed' not in st.session_state:
    st.session_state.assessment_completed = False
if 'current_assessment_question' not in st.session_state:
     st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."


# ===============================
# --- Page Logic ---
# ===============================

# --- Load industries once ---
if 'available_industries' not in st.session_state:
     st.session_state.available_industries = get_unique_industries(listings_collection)


# --------------------------------------------------------------
# 💰 COMPANY VALUATION PAGE
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# 📊 BUSINESS ASSESSMENT PAGE
# --------------------------------------------------------------
elif page == "📊 Business Assessment":
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Engage in an AI-driven Q&A to analyze your business, similar to an investor pitch.")

    # Define max questions for assessment
    max_assessment_questions = 10 # Reduced for brevity maybe?

    # --- Assessment Q&A Flow ---
    if not st.session_state.assessment_completed:
        st.progress(st.session_state.assessment_question_count / max_assessment_questions)
        st.markdown(f"##### Question {st.session_state.assessment_question_count + 1} of {max_assessment_questions}")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.current_assessment_question}**")

        # Use a unique key based on the question count to allow reentry
        response = st.text_area("Your Answer:", height=120, key=f"assess_q_{st.session_state.assessment_question_count}")

        if st.button("Submit Answer & Get Next Question", key="submit_assess", use_container_width=True):
            if response and response.strip():
                # Store question and answer
                st.session_state.assessment_responses[st.session_state.current_assessment_question] = response.strip()
                # Update conversation history (more structured)
                st.session_state.conversation_history.append({"role": "model", "parts": [{"text": st.session_state.current_assessment_question}]})
                st.session_state.conversation_history.append({"role": "user", "parts": [{"text": response.strip()}]})

                st.session_state.assessment_question_count += 1

                if st.session_state.assessment_question_count >= max_assessment_questions:
                    st.session_state.assessment_completed = True
                    st.success("Assessment Q&A complete! Generating analysis...")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    # Generate next question based on the conversation history
                    with st.spinner("🧠 Thinking of the next question..."):
                        # Construct specific prompt for getting the next question
                        system_prompt_next_q = """
You are an expert business analyst and investor interviewer. Based on the conversation history provided, ask the SINGLE most important and relevant next question to assess the business's viability, strategy, or financials. Do not add any conversational filler, just output the question itself. Consider aspects like market, competition, financials, team, operations, and growth plans. Prioritize areas that seem weak, unclear, or unaddressed.
                        """
                        # Create context string from history
                        history_for_prompt = "\n".join([f"{turn['role'].capitalize()}: {turn['parts'][0]['text']}" for turn in st.session_state.conversation_history])

                        next_q_prompt = f"{system_prompt_next_q}\n\n**Conversation History:**\n{history_for_prompt}\n\n**Next Question:**"

                        # Get the next question text
                        next_question_text = gemini_qna(next_q_prompt, is_valuation=False) # Use standard QnA

                        # Simple cleaning of potential artifacts if AI adds quotes etc.
                        cleaned_next_q = next_question_text.strip().strip('"').strip("'")

                        if cleaned_next_q and "?" in cleaned_next_q: # Basic check if it looks like a question
                            st.session_state.current_assessment_question = cleaned_next_q
                        else:
                            # Fallback question if AI fails
                            st.session_state.current_assessment_question = "What are your financial projections for the next year?"
                            st.warning("AI failed to generate next question, using fallback.")

                    st.rerun() # Rerun to display the new question
            else:
                 st.warning("Please enter your answer before submitting.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Assessment Results ---
    elif st.session_state.assessment_completed:
        st.success("✅ Business Assessment Complete")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## AI-Generated Business Analysis")
        st.caption("Based on your answers during the assessment.")

        # Format conversation history for final analysis
        assessment_data = "\n\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in st.session_state.assessment_responses.items()
        ])

        # Specific system prompt for the final analysis
        analysis_system_prompt = """
You are an expert business analyst providing a final assessment based on a Q&A session. Analyze the provided conversation thoroughly.

**Output Structure:**

1.  **Overall Summary:** A brief paragraph highlighting the key takeaways about the business.
2.  **Strengths:** Bullet points identifying positive aspects (e.g., clear value prop, strong traction, good margins).
3.  **Weaknesses/Risks:** Bullet points identifying areas of concern (e.g., unclear financials, high competition, scalability issues, vague strategy).
4.  **Opportunities:** Bullet points suggesting potential avenues for growth or improvement based on the discussion.
5.  **Key Questions Remaining:** Bullet points listing critical unanswered questions or areas needing much deeper investigation (vital for real assessment).
6.  **Concluding Remark:** A short final statement on the business's potential or current standing based *only* on the provided Q&A.

Be objective, specific, and use the information directly from the Q&A provided below. Do not invent information.
        """

        analysis_prompt = f"{analysis_system_prompt}\n\n**Q&A Session Transcript:**\n{assessment_data}\n\n**Provide Comprehensive Analysis:**"

        with st.spinner("📊 Generating comprehensive business analysis report..."):
             # Use the gemini_qna function
             analysis_result = gemini_qna(analysis_prompt, is_valuation=False)

        # Display analysis result
        st.markdown(analysis_result)

        if st.button("Start New Assessment", key="reset_assess", use_container_width=True):
            # Reset assessment state variables
            st.session_state.assessment_step = 0
            st.session_state.conversation_history = []
            st.session_state.assessment_responses = {}
            st.session_state.assessment_question_count = 0
            st.session_state.assessment_completed = False
            st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# --- Footer ---
# ===============================
st.markdown("---")
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub | Powered by Gemini AI & MongoDB | Benchmarks derived from public pitch data.
</div>
""", unsafe_allow_html=True)

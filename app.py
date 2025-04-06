import streamlit as st
import pymongo
from pymongo import MongoClient
from groq import Groq
import os
import json
import random
from collections import defaultdict

# --- Configuration ---
# Using Streamlit Secrets for sensitive info
MONGO_URI = st.secrets.get("MONGO_URI")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
# Fallbacks for local development if secrets aren't set (optional, remove for production)
if not MONGO_URI:
    MONGO_URI = 'mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    # st.warning("MONGO_URI not found in secrets, using hardcoded fallback (remove for production)")
if not GROQ_API_KEY:
     GROQ_API_KEY = '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' # Replace with your actual key ONLY for local test
    # st.warning("GROQ_API_KEY not found in secrets, using hardcoded fallback (remove for production or provide a valid key)")

DEFAULT_COMPLETENESS_TARGET = 0.7 # Target 70% completeness across core areas

# --- Core Categories for Completeness Tracking ---
CORE_ASSESSMENT_AREAS = [
    "Business Fundamentals", "Financial Performance", "Market Position",
    "Operations", "Team & Leadership", "Growth & Scaling", "Investment & Funding",
    "Risk Assessment", "Sustainability & Vision", "Stage Specific", "Industry Specific",
    "Function Specific", "Model Specific", "Strategy Specific", "Specialized Area",
    "Other Core", "Other Specific", "Unknown Category" # Added fallbacks
]

# Map JSON keys slightly differently for better tracking logic
QUESTION_CATEGORY_MAP = {
    "Core Business Analysis Questions": {
        "Business Fundamentals": "Business Fundamentals",
        "Financial Performance": "Financial Performance",
        "Market Position": "Market Position",
        "Operations": "Operations",
        "Team & Leadership": "Team & Leadership",
        "Growth & Scaling": "Growth & Scaling",
        "Investment & Funding": "Investment & Funding", # Keep separate for potential focus
        "Risk Assessment": "Risk Assessment",
        "Sustainability & Vision": "Sustainability & Vision"
    },
    "Specific Business Questions": {
        "By Business Stage": "Stage Specific",
        "By Business Type/Industry": "Industry Specific",
        "By Business Function": "Function Specific",
        "By Business Model": "Model Specific",
        "By Strategic Focus": "Strategy Specific",
        "Specialized Assessment Areas": "Specialized Area"
        # Add explicit mappings for sub-keys under these if needed later
    }
}

# --- MongoDB Connection ---
@st.cache_resource # Cache the connection
def connect_mongo():
    if not MONGO_URI:
        st.error("MongoDB URI is not configured. Please set it in Streamlit secrets.")
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Added timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        db = client['business_rag']
        st.success("Connected to MongoDB!")
        return db['questions']
    except pymongo.errors.ConfigurationError as e:
         st.error(f"MongoDB Configuration Error: {e}. Check your MONGO_URI format and credentials in Streamlit secrets.")
         return None
    except pymongo.errors.ServerSelectionTimeoutError as e:
        st.error(f"MongoDB Connection Timeout: {e}. Check network access, firewall rules, and if the Atlas cluster is active.")
        return None
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# --- Load Questions ---
@st.cache_data # Cache the data fetched from DB
def load_all_questions(_collection): # Pass collection as arg to make it cacheable
    # No need to check _collection here, the caller does it.
    # If this function is ever called with None, it's an issue upstream.
    try:
        # MongoDB stores data in documents; assuming one document holds the entire JSON structure
        data = _collection.find_one() # Assuming only one doc holds the question structure
        if data and "Core Business Analysis Questions" in data: # Basic check
             # Remove the MongoDB '_id' field if it exists
            data.pop('_id', None)
            # Use st.session_state cautiously inside cached functions, set flag *after* successful return
            # This flag is better set in the main logic after the call succeeds.
            # st.session_state['all_questions_loaded'] = True
            return data
        elif data:
             st.error("Question data found in MongoDB, but it's missing the expected 'Core Business Analysis Questions' key.")
             return {}
        else:
            st.error("No question data found in the MongoDB collection.")
            return {}
    except Exception as e:
        st.error(f"Error loading questions from MongoDB: {e}")
        return {}

# --- Groq LLM Interaction ---
# Note: Caching can be less useful if prompts change slightly (e.g., different context)
# @st.cache_data(ttl=3600)
def get_groq_response(prompt, model="llama3-70b-8192"):
    if not GROQ_API_KEY:
        st.error("Groq API Key is not configured. Please set it in Streamlit secrets.")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3, # Lower temperature for more factual/consistent analysis
            max_tokens=600, # Increased slightly for potentially longer reports
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error interacting with Groq API: {e}")
        return None

# --- Helper function to get mapped category ---
def get_mapped_category(q_data):
    # q_data is a tuple like ('Core Business Analysis Questions', 'Business Fundamentals', None, 'What problem...')
    # or ('Specific Business Questions', 'By Business Stage', 'Startup/Early Stage', 'Traction...')
    if not isinstance(q_data, tuple) or len(q_data) < 2:
         st.warning(f"Unexpected format for q_data in get_mapped_category: {q_data}")
         return "Unknown Category"

    primary_key = q_data[0]
    secondary_key = q_data[1]

    if primary_key == "Core Business Analysis Questions":
        return QUESTION_CATEGORY_MAP["Core Business Analysis Questions"].get(secondary_key, "Other Core")
    elif primary_key == "Specific Business Questions":
        # Get the mapped value based on the secondary key (like 'By Business Stage')
        # QUESTION_CATEGORY_MAP["Specific Business Questions"] maps 'By Business Stage' to 'Stage Specific' etc.
        return QUESTION_CATEGORY_MAP["Specific Business Questions"].get(secondary_key, "Other Specific")
    else:
         st.warning(f"Unrecognized primary key in get_mapped_category: {primary_key}")
         return "Unknown Category"


# --- Question Selection Logic ---
def select_next_question(all_questions, answered_questions, business_info, scores, completeness):
    if not all_questions or not business_info.get('stage'):
         # Removed industry_keywords check as it might not be available initially but description is
         # This should ideally be caught before calling this function if setup isn't complete
         st.error("Attempted to select question before business info setup is complete.")
         return ("System", "Error", "Internal state error: Setup not complete."), {}


    available_questions = []
    stage = business_info['stage']
    industry_keys = business_info.get('industry_keywords', []) # Use keywords for rough matching
    identified_industry_key = business_info.get('identified_industry') # Use the previously matched key if available

    # --- Flatten questions and filter ---
    flat_questions = []
    # Iterate through the main categories ('Core Business Analysis Questions', 'Specific Business Questions')
    for main_cat_name, main_cat_value in all_questions.items():
        if isinstance(main_cat_value, dict):
            # Iterate through sub-categories (e.g., 'Business Fundamentals', 'By Business Stage')
            for sub_cat_name, sub_cat_value in main_cat_value.items():
                if not isinstance(sub_cat_value, (dict, list)):
                     continue # Skip unexpected types

                # Case 1: Nested by Stage/Industry/Function/Model/Focus/Specialized
                if sub_cat_name in QUESTION_CATEGORY_MAP.get("Specific Business Questions", {}):
                    # This dict maps e.g. "By Business Stage" to "Stage Specific"
                    is_stage_match = (sub_cat_name == "By Business Stage")
                    is_industry_match = (sub_cat_name == "By Business Type/Industry")
                    # Extend this logic if needing specific matching for Function/Model/Focus/Specialized keys

                    if not isinstance(sub_cat_value, dict): continue # Expecting dict here

                    nested_keys = list(sub_cat_value.keys()) # e.g., ['Idea/Concept Stage', 'Software/SaaS']
                    matching_nested_key = None

                    if is_stage_match and stage in nested_keys:
                        matching_nested_key = stage

                    elif is_industry_match:
                        # Prioritize already identified industry match
                        if identified_industry_key and identified_industry_key in nested_keys:
                            matching_nested_key = identified_industry_key
                        else:
                            # Fallback: Simple keyword check (can be improved)
                            for ind_key in nested_keys:
                                # Match if any keyword appears IN the key name (case-insensitive)
                                if any(keyword.lower() in ind_key.lower() for keyword in industry_keys):
                                     matching_nested_key = ind_key
                                     # Store the matched key for potential reuse THIS SESSION ONLY (don't save to DB here)
                                     st.session_state.business_info['identified_industry'] = ind_key # Update session state
                                     break # Take first keyword match

                    # TODO: Add similar specific key matching logic for Function/Model/Focus/Specialized if needed


                    # If a specific key within Stage/Industry etc. was matched, add its questions
                    if matching_nested_key and isinstance(sub_cat_value.get(matching_nested_key), list):
                       question_list = sub_cat_value[matching_nested_key]
                       for q_text in question_list:
                            flat_questions.append((main_cat_name, sub_cat_name, matching_nested_key, q_text))

                # Case 2: Directly nested list (e.g., Core -> Business Fundamentals)
                elif isinstance(sub_cat_value, list):
                    question_list = sub_cat_value
                    for q_text in question_list:
                        flat_questions.append((main_cat_name, sub_cat_name, None, q_text))


    # --- Create available question list with base scores ---
    for q_data in flat_questions:
         q_text = q_data[-1] # The actual question text
         # Check if this exact question text has been asked
         if q_text not in answered_questions:
            # Store full path for context: (main_cat, sub_cat, nested_key_if_any, q_text)
            available_questions.append({"question_data": q_data, "text": q_text, "score": 5.0}) # Base score

    if not available_questions:
        st.info("Looks like we've covered all relevant questions based on the initial filtering!")
        return ("System", "End", "No more relevant questions found."), {}

    # --- Apply scoring logic ---
    category_boosts = scores.get('category_boosts', {})
    keyword_matches = scores.get('keyword_matches', []) # list of keywords found in last answer

    # Boost based on market conditions (example)
    market_condition = st.session_state.business_info.get('market_condition', 'Neutral')
    risk_boost = 0.0
    financial_boost = 0.0
    if market_condition == 'Pessimistic':
        risk_boost = 2.0
        financial_boost = 1.5 # e.g., focus more on runway, burn rate

    final_scored_questions = []
    for q_item in available_questions:
        q_data = q_item['question_data']
        q_text = q_item['text']
        current_score = q_item['score']
        mapped_category = get_mapped_category(q_data)

        # --- Apply Boosts ---
        # Category boost from Groq suggestion
        current_score += category_boosts.get(mapped_category, 0)

        # Keyword boost (ensure keywords list has actual strings)
        valid_keywords = [kw for kw in keyword_matches if isinstance(kw, str) and kw]
        if any(keyword.lower() in q_text.lower() for keyword in valid_keywords):
            current_score += 5.0 # Strong boost for keyword match

        # Completeness boost (boost categories that are lagging)
        category_completeness = completeness.get(mapped_category, 0)
        if category_completeness < DEFAULT_COMPLETENESS_TARGET:
             # Apply boost proportionally to how far below target it is
             current_score += 3.0 * max(0, (DEFAULT_COMPLETENESS_TARGET - category_completeness)) / DEFAULT_COMPLETENESS_TARGET

        # Market condition boost
        if mapped_category == "Risk Assessment":
             current_score += risk_boost
        if mapped_category == "Financial Performance":
             current_score += financial_boost

        # Stage relevance boost (Example: Early stage prioritizes Fundamentals & Market Fit)
        if stage in ["Idea/Concept Stage", "Startup/Early Stage"] and mapped_category in ["Business Fundamentals", "Market Position", "Stage Specific"]:
            current_score += 1.5
        elif stage == "Growth Stage" and mapped_category in ["Growth & Scaling", "Operations", "Financial Performance"]:
             current_score += 1.5
        # Add more stage-based priors


        # --- Apply Penalties (Example) ---
        # Penalize asking too many from the same category consecutively
        last_category = st.session_state.get('last_question_category')
        if last_category and last_category == mapped_category:
             current_score *= 0.85 # Stronger penalty for repetition

        # TODO: Complexity Level Adaptation penalty/boost based on answer quality history


        q_item['score'] = current_score
        final_scored_questions.append(q_item)


    # Add a bit of randomness to break ties and explore
    for q in final_scored_questions:
        q['score'] += random.uniform(0, 0.5)


    # Sort by score descending
    final_scored_questions.sort(key=lambda x: x['score'], reverse=True)

    # Select the top scoring question
    next_q_item = final_scored_questions[0]

    # Update last asked category for scoring logic
    st.session_state['last_question_category'] = get_mapped_category(next_q_item['question_data'])

    return next_q_item['question_data'], next_q_item

# --- Update Completeness ---
def update_completeness(completeness, category, quality_score):
    # Ensure quality score is a number
    if not isinstance(quality_score, (int, float)):
         quality_score = 3 # Default quality if parsing failed

     # Only count if answer quality is decent
    if quality_score >= 3:
         # Simple increment, adjust as needed
         # Could also make increment size dependent on quality score
         increment = 0.08 + (0.04 * (quality_score - 3)) # Base 0.08, up to 0.16 for quality 5
         completeness[category] = min(1.0, completeness.get(category, 0) + increment)
    # No change for low-quality answers (score 1 or 2)
    return completeness


# --- Parse Groq Analysis Result ---
def parse_analysis(analysis_result_text):
    analysis_data = {'summary': 'N/A', 'quality_score': 3, 'keywords': [], 'suggested_categories': []}
    if not analysis_result_text or not isinstance(analysis_result_text, str):
        return analysis_data # Return default if no text

    lines = analysis_result_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        try:
            if line.startswith("Summary:"):
                analysis_data['summary'] = line.split(":", 1)[1].strip()
            elif line.startswith("Quality Score:"):
                 score_text = line.split(":", 1)[1].strip()
                 # Handle potential non-numeric scores
                 if score_text.isdigit():
                     analysis_data['quality_score'] = int(score_text)
                 else:
                      st.warning(f"Could not parse Quality Score '{score_text}' as integer, using default 3.")
                      analysis_data['quality_score'] = 3 # Default fallback
            elif line.startswith("Keywords:"):
                kws_str = line.split(":", 1)[1].strip()
                if kws_str: # Avoid empty strings
                     # Split, strip whitespace, filter empty results
                     analysis_data['keywords'] = [kw.strip() for kw in kws_str.split(',') if kw.strip()]
            elif line.startswith("Suggested Categories:"):
                cats_str = line.split(":", 1)[1].strip()
                if cats_str:
                    potential_cats = [c.strip() for c in cats_str.split(',') if c.strip()]
                    # Filter against the known valid list
                    analysis_data['suggested_categories'] = [cat for cat in potential_cats if cat in CORE_ASSESSMENT_AREAS]
        except IndexError:
            # Handles cases where split fails (e.g., line without ':')
             st.warning(f"Skipping malformed analysis line: {line}")
        except ValueError:
            # Specifically catches issues like int('abc')
            st.warning(f"Could not parse number in analysis line: {line}")
        except Exception as e:
            # General catch-all
             st.warning(f"Error parsing analysis line '{line}': {e}")

    # Final validation for quality score range
    if not (1 <= analysis_data['quality_score'] <= 5):
         st.warning(f"Parsed quality score {analysis_data['quality_score']} out of range (1-5), resetting to 3.")
         analysis_data['quality_score'] = 3

    return analysis_data


# --- Generate Report ---
def generate_report(history, business_info):
    st.subheader("Business Assessment Report")
    st.write(f"**Business Name:** {business_info.get('name', 'N/A')}")
    st.write(f"**Declared Stage:** {business_info.get('initial_stage', 'N/A')}") # Show what they initially entered
    if business_info.get('stage') != business_info.get('initial_stage'):
         st.write(f"**Adapted Stage Focus:** {business_info.get('stage', 'N/A')}") # If adapted
    st.write(f"**Industry Context:** {business_info.get('description', 'N/A')}")
    st.write(f"**Matched Industry Questions:** {business_info.get('identified_industry', 'Not specified or matched')}")
    st.write(f"**Market Condition Assumption:** {business_info.get('market_condition', 'Neutral')}")
    st.write("---")

    conversation_summary = f"Assessment Conversation Summary for {business_info.get('name', 'the business')}:\n"
    # Filter out system messages or use full history based on need
    user_interactions = [entry for entry in history if entry.get('question_data') and entry['question_data'][0] != "System"]

    for i, entry in enumerate(user_interactions):
        q_text = entry.get('question_text', 'N/A')
        answer = entry.get('answer', 'N/A')
        analysis = entry.get('analysis', {})
        quality = analysis.get('quality_score', 'N/A')
        summary = analysis.get('summary', 'N/A')

        conversation_summary += f"\n{i+1}. Q: {q_text}\n"
        conversation_summary += f"   A: {answer}\n"
        conversation_summary += f"   Analysis (Quality: {quality}): {summary}\n"


    report_prompt = f"""
    Based on the following conversation log with a business owner ({business_info.get('name', 'N/A')}, declared stage: {business_info.get('initial_stage', 'N/A')}, description: "{business_info.get('description', 'N/A')}"), generate a concise business assessment report suitable for an early-stage investor or analyst. Assume a {business_info.get('market_condition', 'Neutral')} market condition context if relevant.

    Conversation Log Summary:
    {conversation_summary}

    Instructions for the report:
    1.  **Executive Summary:** (2-3 sentences) Briefly summarize the business's core concept, its stage, the overall quality of information provided, and a high-level first impression.
    2.  **Key Strengths Identified:** (Bulleted list, 2-4 points) List potential strengths based *strictly* on the information given in the answers. Mention supporting evidence from the log if possible.
    3.  **Potential Weaknesses & Risks:** (Bulleted list, 2-4 points) List potential weaknesses, risks, inconsistencies, or areas needing significant clarification. *Explicitly mention* if answers were vague, contradictory, or if key topics received low-quality scores. Note areas that weren't covered sufficiently.
    4.  **Topic Coverage & Information Quality:** Briefly comment on which core areas (Fundamentals, Financials, Market, Ops, Team, Growth, Risk) seem well-covered versus sparsely covered. Comment on the general consistency and specificity of the owner's responses.
    5.  **Suggested Next Steps for Due Diligence:** (Bulleted list, 2-4 actionable points) Recommend specific questions to ask or documents to request next, based directly on gaps or red flags identified in the conversation. Be specific (e.g., "Request detailed financial projections for the next 12 months", "Ask for clarification on competitor X's differentiation", "Investigate the mentioned supply chain vulnerability").
    6.  **Overall Tone:** Maintain a neutral, objective, and analytical investor tone. Avoid definitive judgments; focus on summarizing provided information and identifying areas for further scrutiny.

    Generate the report below this line:
    ---
    """

    with st.spinner("Generating Assessment Report..."):
        # Use a potentially larger model if available and needed for synthesis
        report_content = get_groq_response(report_prompt, model="llama3-70b-8192") # Ensure model capable of long context/synthesis
        if report_content:
            st.markdown(report_content.split("---")[-1].strip()) # Try to strip off the prompt part if included
        else:
            st.error("Failed to generate the report.")
            st.write("Here's the raw conversation summary used:")
            st.text_area("Conversation Summary", conversation_summary, height=300)

    # Also display completeness scores
    st.sidebar.subheader("Final Information Completeness")
    completeness_data = st.session_state.get('completeness', {})
    if completeness_data:
        for area, score in sorted(completeness_data.items()): # Sort for consistency
             if score > 0: # Only show areas with some progress
                st.sidebar.progress(score, text=f"{area}: {int(score*100)}%")
    else:
        st.sidebar.write("Completeness tracking data not available.")


# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide") # Use more screen space
    st.title("🤖 Business Analyser Bot")
    st.caption("Powered by Groq and MongoDB")

    # --- Initialize Session State ---
    # Use defaultdict for scores and completeness for easier updates
    default_scores = lambda: {'category_boosts': defaultdict(float), 'keyword_matches': []}

    if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
    if 'answered_questions' not in st.session_state: st.session_state.answered_questions = set()
    if 'current_question_data' not in st.session_state: st.session_state.current_question_data = None
    if 'current_question_text' not in st.session_state: st.session_state.current_question_text = None
    if 'business_info' not in st.session_state: st.session_state.business_info = {}
    if 'setup_complete' not in st.session_state: st.session_state.setup_complete = False
    if 'scores' not in st.session_state: st.session_state.scores = default_scores()
    if 'completeness' not in st.session_state: st.session_state.completeness = defaultdict(float)
    if 'all_questions' not in st.session_state: st.session_state.all_questions = {}
    if 'all_questions_loaded' not in st.session_state: st.session_state.all_questions_loaded = False
    # Removed analysis cache as parsing happens immediately now
    # if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}
    if 'interview_finished' not in st.session_state: st.session_state.interview_finished = False
    if 'last_question_category' not in st.session_state: st.session_state.last_question_category = None
    if 'connection_status' not in st.session_state: st.session_state.connection_status = "pending" # pending, connected, failed

    # --- MongoDB Connection Attempt (Only Once Per Session) ---
    if st.session_state.connection_status == "pending":
        question_collection = connect_mongo()
        if question_collection is not None:
             st.session_state.connection_status = "connected"
             st.session_state.question_collection_instance = question_collection # Store instance
        else:
             st.session_state.connection_status = "failed"
             st.error("Halting application due to MongoDB connection failure.")
             st.stop() # Stop execution if DB connection failed

    # --- Load Questions (Only Once After Connection) ---
    if st.session_state.connection_status == "connected" and not st.session_state.all_questions_loaded:
         with st.spinner("Loading question bank..."):
             # Retrieve stored collection instance
             q_collection = st.session_state.get('question_collection_instance')
             # CORRECTED LINE: Explicitly check for None
             if q_collection is not None:
                  st.session_state.all_questions = load_all_questions(q_collection)
                  if st.session_state.all_questions:
                      st.session_state.all_questions_loaded = True
                      # Trigger rerun after loading to proceed
                      st.experimental_rerun()
                  else:
                      st.error("Question bank appears empty or failed to load correctly. Cannot proceed.")
                      st.session_state.connection_status = "failed" # Mark as failed if loading failed
                      st.stop()
             else:
                  # Should not happen if connection_status is 'connected', but defensive check
                  st.error("Internal Error: Question collection instance not found in session state.")
                  st.session_state.connection_status = "failed"
                  st.stop()


    # --- Initial Business Setup Screen ---
    if not st.session_state.setup_complete and st.session_state.all_questions_loaded:
        st.subheader("Let's get started! Tell me about the business:")

        with st.form("setup_form"):
            name = st.text_input("Business Name:")
            description = st.text_area("Briefly describe the business and its primary industry:")

            # Dynamically get stage options if possible, otherwise use common list
            stage_options = ["Idea/Concept Stage", "Startup/Early Stage", "Growth Stage", "Mature Stage", "Turnaround/Restructuring Stage", "Other"]
            default_stage_index = 1 # Default to 'Startup/Early Stage'

            all_q_data = st.session_state.all_questions
            if isinstance(all_q_data, dict):
                 try:
                     specific_questions = all_q_data.get("Specific Business Questions", {})
                     stage_keys = list(specific_questions.get("By Business Stage", {}).keys())
                     if stage_keys:
                         stage_options = sorted(stage_keys) + ["Other"] # Ensure 'Other' is an option
                         if "Startup/Early Stage" in stage_options:
                              default_stage_index = stage_options.index("Startup/Early Stage")
                 except Exception as e:
                      st.warning(f"Could not dynamically load stage options: {e}") # Non-fatal

            stage = st.selectbox("Current Business Stage:", options=stage_options, index=default_stage_index)

            market_condition = st.radio(
                "Assume Current Market Condition:",
                options=['Optimistic', 'Neutral', 'Pessimistic'],
                index=1, # Default to Neutral
                horizontal=True
                )

            submitted = st.form_submit_button("Confirm Setup")
            if submitted:
                if name and description and stage:
                    st.session_state.business_info['name'] = name
                    st.session_state.business_info['description'] = description
                    chosen_stage = stage if stage != "Other" else "Startup/Early Stage" # Use default if Other
                    st.session_state.business_info['stage'] = chosen_stage
                    st.session_state.business_info['initial_stage'] = chosen_stage # Store initial for report
                    st.session_state.business_info['market_condition'] = market_condition
                    # Simple keyword extraction (improved slightly)
                    # Lowercase, split, filter short words/common words (optional)
                    words = [w for w in description.lower().split() if len(w) > 3]
                    # TODO: Add more sophisticated keyword/industry identification here if needed
                    st.session_state.business_info['industry_keywords'] = words
                    st.session_state.setup_complete = True
                    st.experimental_rerun() # Rerun to move past setup
                else:
                    st.warning("Please fill in all required fields (Name, Description, Stage).")

    # --- Main Interview Loop Screen ---
    elif st.session_state.setup_complete and not st.session_state.interview_finished:

        # Layout: Sidebar for progress, Main area for Q&A
        st.sidebar.subheader("Interview Progress")
        completeness_data = st.session_state.get('completeness', {})
        if completeness_data:
             avg_completeness = sum(completeness_data.values()) / len(CORE_ASSESSMENT_AREAS) if CORE_ASSESSMENT_AREAS else 0
             st.sidebar.progress(avg_completeness, text=f"Overall: {int(avg_completeness*100)}%")
             st.sidebar.divider()
             for area, score in sorted(completeness_data.items()):
                if score > 0: # Only show areas with progress
                    st.sidebar.progress(score, text=f"{area}: {int(score*100)}%")
        else:
            st.sidebar.write("Awaiting first answer...")

        # Determine the next question if none is current
        if st.session_state.current_question_text is None:
             next_q_data, next_q_item_info = select_next_question(
                 st.session_state.all_questions,
                 st.session_state.answered_questions,
                 st.session_state.business_info,
                 st.session_state.scores,
                 st.session_state.completeness
            )
             # Check for end/error conditions from selector
             if next_q_data[0] == "System":
                  if next_q_data[1] == "End":
                       st.success(next_q_data[2]) # Show message like "No more questions"
                       st.session_state.interview_finished = True
                       st.balloons()
                       st.experimental_rerun() # Go to report generation
                  elif next_q_data[1] == "Error":
                       st.error(f"System Error during question selection: {next_q_data[2]}")
                       st.stop() # Halt on error
                  # elif next_q_data[1] == "Initial": # This shouldn't be reachable here anymore
                  #      st.warning("Need initial setup first.")

             else:
                # Valid question selected
                st.session_state.current_question_data = next_q_data # Store full data tuple
                st.session_state.current_question_text = next_q_data[-1] # Just the text for display


        # --- Display Q&A Area ---
        if st.session_state.current_question_text:
             st.subheader("Question:")
             st.markdown(f"##### {st.session_state.current_question_text}")

             # Use a form for the answer submission
             with st.form(key=f"answer_form_{len(st.session_state.conversation_history)}"):
                 user_answer = st.text_area("Your Answer:", height=150, key=f"text_area_{len(st.session_state.conversation_history)}")

                 col1, col2, col3 = st.columns([2,1,1]) # Give more space to submit
                 with col1:
                     submit_button = st.form_submit_button("Submit Answer", type="primary")
                 # Finish early button outside the form
                 with col3:
                      finish_early_button_visible = len(st.session_state.conversation_history) > 3 # Only show after a few questions
                      if finish_early_button_visible:
                         finish_early = st.button("Finish Interview Early")
                      else:
                          finish_early = False


             # Handle "Finish Early" outside the form logic
             if finish_early:
                  st.session_state.interview_finished = True
                  st.warning("Finishing interview early. Report will be based on information gathered so far.")
                  st.experimental_rerun()

             # Handle Form Submission
             if submit_button:
                  if user_answer and user_answer.strip(): # Check if answer is not just whitespace
                      current_q_text = st.session_state.current_question_text
                      current_q_data = st.session_state.current_question_data
                      # Add to answered set *before* potentially failing analysis
                      st.session_state.answered_questions.add(current_q_text)

                      # Process the answer with Groq
                      # Assemble prompt parts
                      qa_pair = f"Question: '{current_q_text}'\nAnswer: \"{user_answer}\""
                      analysis_instructions = f"""
Based *only* on this answer provided to the specific question:
1.  Briefly summarize the key information points provided. (Summary: ...)
2.  Assess the answer's quality (clarity, completeness, specificity). Provide a single score from 1 (very poor) to 5 (excellent). (Quality Score: ...)
3.  Identify key topics, entities, concepts, or potential flags mentioned (e.g., competition, scaling challenge, specific financial metric, risk, IP). List up to 5 concise keywords or short phrases. (Keywords: ...)
4.  Suggest 2-3 *categories* from this exact list [{', '.join(CORE_ASSESSMENT_AREAS)}] that seem most relevant to explore next based *only* on the content of this specific answer. Do not suggest categories already implied by the question itself unless the answer adds significant new detail there. (Suggested Categories: ...)

Provide the output ONLY in the format specified (Summary: ..., Quality Score: ..., etc.). Do not add any preamble or explanation.
                      """
                      analysis_prompt = f"{qa_pair}\n\n{analysis_instructions}"

                      with st.spinner("Analysing response..."):
                          analysis_result_text = get_groq_response(analysis_prompt)

                      # Parse the result
                      analysis_data = parse_analysis(analysis_result_text)

                      # --- Update State After Successful Processing ---
                      # Store history entry
                      st.session_state.conversation_history.append({
                          'question_text': current_q_text,
                          'question_data': current_q_data, # Store the tuple for context
                          'answer': user_answer,
                          'analysis': analysis_data # Store the parsed dictionary
                      })

                      # Update scores for next selection based on parsed analysis
                      next_category_boosts = defaultdict(float)
                      for cat in analysis_data.get('suggested_categories', []):
                          next_category_boosts[cat] += 3.0 # Apply boost value
                      st.session_state.scores = {
                          'category_boosts': next_category_boosts, # Pass the defaultdict
                          'keyword_matches': analysis_data.get('keywords', [])
                      }

                       # Update completeness
                      mapped_cat = get_mapped_category(current_q_data)
                      st.session_state.completeness = update_completeness(
                           st.session_state.completeness, # Pass the defaultdict
                           mapped_cat,
                           analysis_data['quality_score']
                      )


                      # --- Stage Adaptation Check (Example) ---
                      # TODO: Implement logic here - if answers consistently mention scaling, prompt user?
                      # e.g., check keywords from analysis_data['keywords'] or summary across history

                      # Reset current question to trigger selection of the next one
                      st.session_state.current_question_data = None
                      st.session_state.current_question_text = None

                      # Check for completion based on average completeness score
                      avg_completeness = sum(st.session_state.completeness.values()) / len(CORE_ASSESSMENT_AREAS) if CORE_ASSESSMENT_AREAS else 0
                      if avg_completeness >= DEFAULT_COMPLETENESS_TARGET:
                              st.success(f"Sufficient information gathered (Average Completeness: {int(avg_completeness*100)}%). Proceeding to report.")
                              st.session_state.interview_finished = True
                              st.balloons()

                      st.experimental_rerun() # Rerun to display next question or finish/report
                  else:
                      st.warning("Please provide an answer before submitting.")


        # --- Display Conversation History (Optional Expander) ---
        if st.session_state.conversation_history:
             with st.expander("Show Conversation History"):
                  for i, entry in enumerate(reversed(st.session_state.conversation_history)): # Show latest first
                      st.markdown(f"**{len(st.session_state.conversation_history)-i}. Q: {entry['question_text']}**")
                      st.text(f"A: {entry['answer']}")
                      analysis = entry.get('analysis', {})
                      st.caption(f"Analysis: [Quality: {analysis.get('quality_score', '?')}] {analysis.get('summary', 'N/A')}")
                      st.divider()


        # --- Debug/Info panel in sidebar ---
        with st.sidebar.expander("Debug Info"):
            st.json(st.session_state.business_info, expanded=False)
            st.write("Answered Count:", len(st.session_state.answered_questions))
            # Convert defaultdict for display if needed
            scores_display = {
                 'category_boosts': dict(st.session_state.scores['category_boosts']),
                 'keyword_matches': st.session_state.scores['keyword_matches']
                 }
            st.json(scores_display, expanded=False)
            st.write("Completeness:", dict(st.session_state.completeness))
            st.write("Last Asked Category:", st.session_state.last_question_category)
            st.write("All Qs Loaded:", st.session_state.all_questions_loaded)


    # --- Report Generation Screen ---
    elif st.session_state.interview_finished:
        if not st.session_state.conversation_history:
             st.warning("No conversation history recorded. Cannot generate a report.")
        else:
             # Call report generation function
             generate_report(st.session_state.conversation_history, st.session_state.business_info)

        # Option to restart
        st.divider()
        if st.button("Start New Interview", type="primary"):
             # Clear *all* session state to ensure a clean start
             # Be careful if using st.cache_resource/data, reset might require app restart sometimes
             keys_to_clear = list(st.session_state.keys()) # Get all keys
             for key in keys_to_clear:
                 del st.session_state[key]
             # Need to force rerun to truly restart the logic flow
             st.experimental_rerun()


if __name__ == "__main__":
    # Basic check for required secrets
    if not MONGO_URI or not GROQ_API_KEY:
         st.error("Missing essential configuration (MONGO_URI or GROQ_API_KEY). Please configure Streamlit secrets.")
         st.stop()
    else:
         main()

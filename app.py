import streamlit as st
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
from datetime import datetime
from groq import Groq, APIError, RateLimitError
import os
from dotenv import load_dotenv
import time
import random

# --- Load Environment Variables ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Constants ---
# Adjust these as needed
INITIAL_FOUNDATIONAL_QUESTIONS = 3
MIN_QUESTIONS_TO_ASSESS = 8
MAX_QUESTIONS_TO_ASK = 25 # Increased max for potentially deeper dives
REQUIRED_CATEGORIES_COVERED = {"Financials", "Operations", "Marketing", "Product/Service"} # Core areas for small biz

# --- Page Config (Keep as is) ---
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Keep as is) ---
st.markdown("""
<style>
    /* ... [Your existing CSS here] ... */
</style>
""", unsafe_allow_html=True)

# --- MongoDB Connection (Keep enhanced version) ---
@st.cache_resource
def get_mongo_client():
    for attempt in range(3):
        try:
            # Increase timeout slightly if experiencing issues on initial connect
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=7000)
            client.admin.command('ismaster')
            print("MongoDB connected successfully!")
            return client
        except pymongo.errors.ConnectionFailure as e:
            print(f"MongoDB Connection Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                st.error(f"Failed to connect to MongoDB after retries: {e}. Please check your MONGO_URI and network settings.")
                st.stop()
        except Exception as e:
             st.error(f"An unexpected error occurred during MongoDB connection: {e}")
             st.stop()

client = get_mongo_client()
db = client['business_rag']
business_collection = db['business_attributes']
question_collection = db['questions'] # Ensure this collection exists and has data + new fields
listings_collection = db['business_listings']

# --- Groq API Setup ---
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Helper Functions (Keep safe_float, get_business, get_all_businesses) ---
def safe_float(value, default=0):
    # ... (keep existing implementation)
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return default

@st.cache_data(ttl=3600)
def get_business(business_name):
    _db = get_mongo_client()['business_rag']
    return _db['business_attributes'].find_one({"business_name": business_name})

@st.cache_data(ttl=3600)
def get_all_businesses(limit=2072):
    _db = get_mongo_client()['business_rag']
    return list(_db['business_attributes'].find().limit(limit))


# --- Enhanced Groq QnA Function ---
def groq_qna(prompt, system_prompt_addon=None, context=None, model="llama-3.1-8b-instant", max_tokens=1500):
    """ Calls Groq API with optional context and system prompt additions. """
    try:
        # Base system prompt (Keep your original detailed one)
        base_system_prompt = """
Expert Business Investor Interview System
System Role Definition
You are an expert business analyst and investor interviewer, combining the analytical precision of Kevin O'Leary...
[... Your full original system prompt ...]
"""
        # --- ADDITION for Small Business & Conversational Flow ---
        small_biz_addon = """
ADDITIONAL INSTRUCTIONS FOR THIS SESSION:
- Focus: Your primary goal is to assess small, often local businesses (like convenience stores, mechanical shops, local service providers, small restaurants, online artisans, etc.). Tailor your thinking and implied follow-ups to this scale.
- Question Source: You are simulating access to a database of validated investor questions (like those from Shark Tank/Dragon's Den). When asked to determine the next question or topic, you should aim to suggest a relevant area that likely corresponds to a question in that database.
- Conversational Flow: Mimic a natural conversation. If the user mentions operating for '4 years', a logical follow-up relates to performance during those years (like revenue). Your suggested next topic should reflect this type of logical progression based on the *latest* answer provided in the context.
- Adaptability: Adjust your line of questioning based on the business type inferred from the owner's answers (e.g., inventory questions for retail, scheduling/capacity for services).
"""
        final_system_prompt = base_system_prompt + "\n" + small_biz_addon
        if system_prompt_addon:
            final_system_prompt += "\n" + system_prompt_addon

        context_str = f"\n\nConversation Context So Far:\n{context}" if context else "\n\nNo prior context for this specific query."

        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": f"{context_str}\n\nUser Query/Task:\n{prompt}"}
            ],
            max_tokens=max_tokens,
            # temperature=0.5 # Slightly lower temp might make category selection more predictable
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        st.error("Groq rate limit exceeded. Please wait and try again.")
        return None
    except APIError as e:
        st.error(f"Groq API error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred calling Groq: {e}")
        return None


# --- Enhanced MongoDB Question Fetching Function ---
@st.cache_data(ttl=600)
def get_next_question_from_db(asked_ids, is_foundational=False, target_category=None):
    """
    Fetches the next question from MongoDB.
    - Prioritizes foundational questions if is_foundational is True.
    - Prioritizes target_category if provided (and not foundational phase).
    - Excludes questions already asked.
    """
    _db = get_mongo_client()['business_rag']
    query = {"_id": {"$nin": asked_ids}} # Exclude asked questions

    if is_foundational:
        query["is_foundational"] = True
        # print(f"DEBUG: Fetching foundational question. Query: {query}") # Debug
    elif target_category:
        query["category"] = target_category
        # Optional: Add logic here later to use 'tags' if you implement them
        # print(f"DEBUG: Fetching question in category '{target_category}'. Query: {query}") # Debug

    # Use aggregation pipeline with $sample for randomness within the filtered set
    pipeline = [
        {"$match": query},
        {"$sample": {"size": 1}}
    ]

    try:
        results = list(_db['questions'].aggregate(pipeline))
        if results:
            question_doc = results[0]
            # print(f"DEBUG: Found question: {question_doc['text']} (Category: {question_doc.get('category')})") # Debug
            # Ensure foundational flag is handled correctly
            if is_foundational and not question_doc.get('is_foundational'):
                # This shouldn't happen with the query, but as a safeguard
                print("WARN: Foundational query returned non-foundational doc. Fetching again without category.")
                query.pop("is_foundational", None) # Remove flag and try generic fetch
                pipeline = [{"$match": query}, {"$sample": {"size": 1}}]
                results = list(_db['questions'].aggregate(pipeline))
                if results: question_doc = results[0]
                else: return None # No more questions at all

            return question_doc
        else:
            # print(f"DEBUG: No question found for query: {query}") # Debug
            # Fallback: If a specific category yielded no results, try finding *any* unasked question
            if not is_foundational and target_category:
                 query.pop("category", None) # Remove category constraint
                 # print(f"DEBUG: Fallback - No question in '{target_category}', trying any category. Query: {query}") # Debug
                 pipeline = [{"$match": query}, {"$sample": {"size": 1}}]
                 results = list(_db['questions'].aggregate(pipeline))
                 if results:
                     # print("DEBUG: Found fallback question from any category.") # Debug
                     return results[0]

            return None # No more suitable questions found
    except Exception as e:
        st.error(f"Error fetching question from MongoDB: {e}")
        return None


# --- Sidebar Navigation (Keep as is) ---
with st.sidebar:
    # ... (sidebar code remains the same) ...
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Business Insights Hub")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("", ["💰 Company Valuation", "📊 Business Assessment"])
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)


# --- Session State Initialization (Ensure all assessment variables are present) ---
# ... (valuation state remains the same) ...
if 'valuation_data' not in st.session_state: st.session_state.valuation_data = {}
if 'valuation_step' not in st.session_state: st.session_state.valuation_step = 0

# Assessment specific state
if 'assessment_phase' not in st.session_state: st.session_state.assessment_phase = 'not_started'
if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
if 'assessment_responses' not in st.session_state: st.session_state.assessment_responses = {}
if 'asked_question_ids' not in st.session_state: st.session_state.asked_question_ids = []
if 'current_question' not in st.session_state: st.session_state.current_question = None
if 'covered_categories' not in st.session_state: st.session_state.covered_categories = set()
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None


# --- Page Implementations ---

# 1. Company Valuation (Keep as is, or use the improved version from previous response)
if "Company Valuation" in page:
    # ... [Your existing/improved Company Valuation code here] ...
    st.markdown("# 💰 Company Valuation Estimator")
    st.info("This section uses a predefined set of questions for valuation.")
    # (Include the full code for the valuation steps, calculation, and display here)


# 2. Interactive Business Assessment (REVISED LOGIC)
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Answer questions like you're talking to an investor. We'll adapt based on your small business context.")

    # --- Start Assessment ---
    if st.session_state.assessment_phase == 'not_started':
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("Ready to discuss your business? Click below to start the assessment.")
        if st.button("🚀 Start Business Assessment", use_container_width=True):
            # Reset state
            st.session_state.conversation_history = []
            st.session_state.assessment_responses = {}
            st.session_state.asked_question_ids = []
            st.session_state.current_question = None
            st.session_state.covered_categories = set()
            st.session_state.analysis_result = None

            # Fetch the first *foundational* question
            first_question = get_next_question_from_db(asked_ids=[], is_foundational=True)
            if first_question:
                st.session_state.current_question = first_question
                st.session_state.asked_question_ids.append(first_question['_id'])
                st.session_state.assessment_phase = 'asking'
                st.rerun()
            else:
                st.error("Could not fetch an initial question from the database. Please ensure 'foundational' questions exist.")
                st.session_state.assessment_phase = 'not_started'
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Asking Questions Phase ---
    elif st.session_state.assessment_phase == 'asking':
        num_asked = len(st.session_state.asked_question_ids)
        st.progress(min(1.0, num_asked / MAX_QUESTIONS_TO_ASK))
        st.markdown(f"##### Question {num_asked} (Approx. {MIN_QUESTIONS_TO_ASSESS}-{MAX_QUESTIONS_TO_ASK} total)")

        if st.session_state.current_question:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            question_doc = st.session_state.current_question
            question_text = question_doc['text']
            question_id = question_doc['_id']
            question_category = question_doc.get('category', 'General')

            st.markdown(f"**{question_text}**")
            st.caption(f"Topic Area: {question_category}")

            response = st.text_area("Your Answer", height=120, key=f"q_{question_id}") # Slightly larger text area

            if st.button("Submit Answer", use_container_width=True, key=f"submit_{question_id}"):
                if response:
                    # --- Store Response & Update State ---
                    st.session_state.assessment_responses[question_text] = response
                    st.session_state.conversation_history.append({
                        "question": question_text,
                        "answer": response,
                        "category": question_category
                    })
                    if question_category != 'General': # Track covered topics
                        st.session_state.covered_categories.add(question_category)

                    # --- Check Termination Conditions ---
                    proceed_to_report = False
                    # Condition 1: Max questions reached
                    if num_asked >= MAX_QUESTIONS_TO_ASK:
                        st.warning("Maximum question limit reached. Generating report...")
                        proceed_to_report = True
                    # Condition 2: Minimums met, check sufficiency
                    elif num_asked >= MIN_QUESTIONS_TO_ASSESS and REQUIRED_CATEGORIES_COVERED.issubset(st.session_state.covered_categories):
                        with st.spinner("AI checking if enough data collected for assessment..."):
                            conversation_context = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in st.session_state.conversation_history])
                            # Enhanced sufficiency prompt for small biz context
                            sufficiency_prompt = f"""
                            Review the conversation history provided below, which is an interview with a small business owner (e.g., shop, local service).

                            Conversation History:
                            {conversation_context}

                            Task: Based *only* on this conversation, determine if there's *sufficient* information covering the essential aspects (like what they do, target customers, basic operations, general financial health idea, marketing approach) to generate a *preliminary but meaningful* assessment for an investor evaluating this *type* of small business. We don't need every detail yet, but are the core pillars touched upon?

                            Respond with only 'YES' or 'NO'.
                            """
                            ai_sufficiency_check = groq_qna(sufficiency_prompt, model="llama-3.1-8b-instant", max_tokens=20)

                        if ai_sufficiency_check and ai_sufficiency_check.strip().upper() == "YES":
                            st.success("AI determined enough information has been gathered for a preliminary assessment.")
                            proceed_to_report = True

                    # --- Decide Next Step ---
                    if proceed_to_report:
                        st.session_state.assessment_phase = 'generating_report'
                        st.rerun()
                    else:
                        # Determine next question dynamically
                        with st.spinner("AI determining the next best question area..."):
                            conversation_context = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in st.session_state.conversation_history])
                            last_exchange = st.session_state.conversation_history[-1]

                            # Define potential categories from your DB for guidance
                            # Fetch distinct categories dynamically if possible, otherwise list common ones
                            try:
                                available_categories = question_collection.distinct("category", {"category": {"$ne": None}}) # Get actual categories from DB
                                if not available_categories: raise Exception("No categories found") # Handle empty case
                                available_categories = [c for c in available_categories if c] # Remove None/empty strings
                            except Exception as e:
                                print(f"WARN: Could not fetch distinct categories from DB ({e}), using default list.")
                                available_categories = ["Financials", "Marketing", "Operations", "Team", "Strategy", "Product/Service", "Market", "Competition", "Legal", "Background"] # Fallback list

                            # Enhanced prompt for next topic suggestion
                            next_topic_prompt = f"""
                            You are role-playing an investor interviewing a small business owner. Review the conversation history, paying close attention to the *last exchange*.

                            Conversation History:
                            {conversation_context}

                            Last Exchange:
                            Q: {last_exchange['question']}
                            A: {last_exchange['answer']}

                            Task: Based on the *entire* conversation and especially the *last answer*, what is the single MOST logical and important business topic area to ask about *next* to gain a deeper understanding for investment assessment? Consider the typical aspects of running a small business (shop, service, etc.). Choose the best fit from the following available categories.

                            Available Categories: {', '.join(available_categories)}

                            Respond with ONLY the single most relevant category name from the list.
                            """
                            suggested_category = groq_qna(next_topic_prompt, model="llama-3.1-8b-instant", max_tokens=50)

                        # Parse suggested category
                        parsed_category = None
                        if suggested_category:
                             cleaned_suggestion = suggested_category.strip().replace("Category:", "").strip()
                             # Find the best match from available categories
                             best_match_score = -1
                             for cat in available_categories:
                                 if cat.lower() == cleaned_suggestion.lower(): # Exact match preferred
                                     parsed_category = cat
                                     break
                                 # Add fuzzy matching later if needed, but exact is safer
                             if not parsed_category: # If no exact match, maybe take the first word if it looks like a category? Risky.
                                 print(f"WARN: AI suggested category '{cleaned_suggestion}' not found exactly in available list: {available_categories}. Will attempt fetch without category.")
                             # print(f"DEBUG: AI suggested category: {suggested_category} -> Parsed: {parsed_category}") # Debug

                        # Fetch the next question from DB
                        is_still_foundational = (num_asked < INITIAL_FOUNDATIONAL_QUESTIONS)
                        next_question_doc = get_next_question_from_db(
                            asked_ids=st.session_state.asked_question_ids,
                            is_foundational=is_still_foundational, # Continue foundational if needed
                            target_category=None if is_still_foundational else parsed_category # Only use target cat after foundational phase
                        )

                        if next_question_doc:
                            st.session_state.current_question = next_question_doc
                            st.session_state.asked_question_ids.append(next_question_doc['_id'])
                            # Don't rerun immediately, let the button press handle the loop
                        else:
                            st.warning("No more relevant questions found in the database based on the conversation flow. Generating report.")
                            st.session_state.assessment_phase = 'generating_report'

                        st.rerun() # Rerun to display the new question or move to report generation

                else:
                    st.warning("Please provide an answer.")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Error: Assessment phase is 'asking' but no current question is loaded. State might be inconsistent.")
            if st.button("Try Restarting Assessment"):
                st.session_state.assessment_phase = 'not_started'
                st.rerun()

    # --- Generating Report Phase ---
    elif st.session_state.assessment_phase == 'generating_report':
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Generating Business Assessment Report")
        st.info("The AI investor is analyzing your responses to create a preliminary assessment...")

        with st.spinner("🤖 Preparing assessment based on conversation..."):
            assessment_data = "\n\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in st.session_state.conversation_history])

            # Enhanced analysis prompt focusing on small business context and conversational data
            analysis_prompt = f"""
            Act as an expert business analyst specializing in evaluating **small, local businesses** (like shops, restaurants, service providers). You have just concluded an initial interview conversation with the owner.

            **Interview Conversation Transcript:**
            {assessment_data}

            **Your Task:** Generate a preliminary business assessment report based *strictly* on the information provided in the transcript.

            **Report Structure:**

            1.  **Business Snapshot:**
                *   Inferred Business Type & Stage: (e.g., "Appears to be an operating local retail store, likely 3-5 years old.")
                *   Core Offering/Value Proposition: (What problem do they solve for customers, based on their words?)
                *   Target Customer (Implied): (Who do they seem to be serving?)

            2.  **Initial Assessment (Based ONLY on Conversation):**
                *   **Apparent Strengths:** (What sounds positive or well-handled based on their answers?)
                *   **Potential Weaknesses/Concerns:** (What raises red flags, seems unclear, or sounds challenging based on their answers?)
                *   **Noted Opportunities:** (Are there hints of potential growth or improvement areas mentioned?)

            3.  **Key Information Gaps:**
                *   List the *most critical* pieces of information *missing* from this conversation that you would need for a proper investment decision for this *type* of small business. (e.g., "Detailed breakdown of monthly expenses", "Specific customer acquisition cost", "Lease agreement details", "Local competitor comparison"). Be specific.

            4.  **Investor's Next Steps:**
                *   **Overall Impression:** (Briefly state your gut feeling based *only* on the conversation - e.g., "Cautiously optimistic", "Needs significant clarification", "Intriguing but lacks financial detail").
                *   **Top 3 Follow-Up Questions:** List the 3 *most important, specific questions* you would ask next to address the biggest information gaps.

            **Important:** Stick to the transcript. Do not invent data. Acknowledge the preliminary nature of this assessment due to limited information. Maintain a professional, analytical investor tone suitable for small business evaluation.
            """
            # Use a capable model for analysis
            analysis_result = groq_qna(analysis_prompt, model="llama-3.1-70b-versatile", max_tokens=4000)
            st.session_state.analysis_result = analysis_result
            st.session_state.assessment_phase = 'displaying_report'
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


    # --- Displaying Report Phase ---
    elif st.session_state.assessment_phase == 'displaying_report':
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Preliminary Business Assessment Report")
        st.markdown("*(Based on the interview conversation)*")

        if st.session_state.analysis_result:
            st.markdown(st.session_state.analysis_result)
        else:
            st.error("Failed to generate or retrieve the assessment report. Please try starting a new assessment.")

        if st.button("🔄 Start New Assessment", use_container_width=True):
            # Reset state completely
            st.session_state.assessment_phase = 'not_started'
            st.session_state.conversation_history = []
            st.session_state.assessment_responses = {}
            st.session_state.asked_question_ids = []
            st.session_state.current_question = None
            st.session_state.covered_categories = set()
            st.session_state.analysis_result = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# --- Footer (Keep as is) ---
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub © 2025 | Powered by Groq AI |
</div>
""", unsafe_allow_html=True)

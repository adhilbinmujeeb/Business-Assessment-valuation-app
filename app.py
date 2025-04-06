import streamlit as st
import pymongo
from pymongo import MongoClient
from groq import Groq
import os
import json
import random
from collections import defaultdict

# --- Configuration ---
# Best practice: Use Streamlit secrets or environment variables for sensitive info
# For demonstration purposes, placing them here but add warnings.
# st.warning("Using hardcoded API keys. Use st.secrets or environment variables in production!")
MONGO_URI = st.secrets.get("MONGO_URI", 'mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", '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')
DEFAULT_COMPLETENESS_TARGET = 0.7 # Target 70% completeness across core areas

# --- Core Categories for Completeness Tracking ---
CORE_ASSESSMENT_AREAS = [
    "Business Fundamentals", "Financial Performance", "Market Position",
    "Operations", "Team & Leadership", "Growth & Scaling",
    "Risk Assessment", "Sustainability & Vision", "Stage Specific", "Industry Specific"
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
        "Investment & Funding": "Investment & Funding", # Could merge or keep separate
        "Risk Assessment": "Risk Assessment",
        "Sustainability & Vision": "Sustainability & Vision"
    },
    "Specific Business Questions": {
        "By Business Stage": "Stage Specific",
        "By Business Type/Industry": "Industry Specific",
        # ... add others if needed from Specialized Assessment Areas
    }
}

# --- MongoDB Connection ---
@st.cache_resource # Cache the connection
def connect_mongo():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Added timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        db = client['business_rag']
        st.success("Connected to MongoDB!")
        return db['questions']
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# --- Load Questions ---
@st.cache_data # Cache the data fetched from DB
def load_all_questions(_collection): # Pass collection as arg to make it cacheable
    if _collection is None:
        return {}
    try:
        # MongoDB stores data in documents; assuming one document holds the entire JSON structure
        # Adjust query if structure is different (e.g., one doc per category)
        data = _collection.find_one() # Assuming only one doc holds the question structure
        if data and "Core Business Analysis Questions" in data: # Basic check
             # Remove the MongoDB '_id' field if it exists
            data.pop('_id', None)
            st.session_state['all_questions_loaded'] = True
            return data
        else:
            st.error("Could not find the expected question structure in MongoDB.")
            return {}
    except Exception as e:
        st.error(f"Error loading questions from MongoDB: {e}")
        return {}

# --- Groq LLM Interaction ---
@st.cache_data(ttl=3600) # Cache Groq responses for an hour to save API calls for identical requests
def get_groq_response(prompt, model="llama3-70b-8192"):
    if not GROQ_API_KEY:
        st.error("Groq API Key not configured.")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3, # Lower temperature for more factual/consistent analysis
            max_tokens=500,
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
    # q_data is a tuple like ('Core Business Analysis Questions', 'Business Fundamentals', 'What problem...')
    primary_key = q_data[0]
    secondary_key = q_data[1]

    if primary_key == "Core Business Analysis Questions":
        return QUESTION_CATEGORY_MAP["Core Business Analysis Questions"].get(secondary_key, "Other Core")
    elif primary_key == "Specific Business Questions":
        # Check if it's Stage, Industry, or other Specific category
        if secondary_key == "By Business Stage":
            return "Stage Specific"
        elif secondary_key == "By Business Type/Industry":
            return "Industry Specific"
        # Add mappings for other specific areas if you use them
        elif secondary_key == "By Business Function":
             # You might want finer granularity here based on the function (e.g., 'Marketing', 'Finance Func')
             return "Function Specific"
        elif secondary_key == "By Business Model":
            return "Model Specific"
        elif secondary_key == "By Strategic Focus":
            return "Strategy Specific"
        elif secondary_key == "Specialized Assessment Areas":
            # Could map sub-keys like ESG, Digital Transformation etc.
             return "Specialized Area"
        else:
            return "Other Specific" # Fallback
    else:
         return "Unknown Category"


# --- Question Selection Logic ---
def select_next_question(all_questions, answered_questions, business_info, scores, completeness):
    if not all_questions or not business_info.get('stage') or not business_info.get('industry_keywords'):
        # If basic info isn't set, ask for it first (handled in main app logic)
         return ("System", "Initial", "Please provide basic business info first."), {}

    available_questions = []
    stage = business_info['stage']
    industry_keys = business_info.get('industry_keywords', []) # Use keywords for rough matching

    # Flatten questions and filter
    flat_questions = []
    # Iterate through the main categories ('Core Business Analysis Questions', 'Specific Business Questions')
    for main_cat_name, main_cat_value in all_questions.items():
        if isinstance(main_cat_value, dict):
            # Iterate through sub-categories (e.g., 'Business Fundamentals', 'By Business Stage')
            for sub_cat_name, sub_cat_value in main_cat_value.items():
                # Special handling for nested stage/industry questions
                if sub_cat_name in ["By Business Stage", "By Business Type/Industry"]:
                    # Check if the nested key matches the user's stage/industry
                    nested_keys = list(sub_cat_value.keys()) # e.g., ['Idea/Concept Stage', 'Startup/Early Stage'] or ['Software/SaaS', 'E-commerce']

                    matching_nested_key = None
                    if sub_cat_name == "By Business Stage":
                        # Direct match (case-insensitive compare might be better)
                        if stage in nested_keys:
                           matching_nested_key = stage

                    elif sub_cat_name == "By Business Type/Industry":
                        # Keyword matching or direct match for industry
                        # Simple check if provided industry name matches a key
                        identified_industry = business_info.get('identified_industry')
                        if identified_industry and identified_industry in nested_keys:
                             matching_nested_key = identified_industry
                        else:
                            # Simple keyword check as fallback (can be improved)
                            for ind_key in nested_keys:
                                if any(keyword.lower() in ind_key.lower() for keyword in industry_keys):
                                     matching_nested_key = ind_key
                                     # Store the matched key for potential reuse
                                     business_info['identified_industry'] = ind_key
                                     break # Take first match

                    if matching_nested_key and isinstance(sub_cat_value[matching_nested_key], list):
                       question_list = sub_cat_value[matching_nested_key]
                       for q_text in question_list:
                            flat_questions.append((main_cat_name, sub_cat_name, matching_nested_key, q_text))

                elif isinstance(sub_cat_value, list): # Standard case like Core -> Business Fundamentals
                    question_list = sub_cat_value
                    for q_text in question_list:
                        flat_questions.append((main_cat_name, sub_cat_name, None, q_text)) # None indicates no further nesting


    # Create available question list with base scores
    for q_data in flat_questions:
         q_text = q_data[-1] # The actual question text
         # Check if this exact question text has been asked
         if q_text not in answered_questions:
            # Store full path for context: (main_cat, sub_cat, nested_key_if_any, q_text)
            available_questions.append({"question_data": q_data, "text": q_text, "score": 5.0}) # Base score

    if not available_questions:
        return ("System", "End", "No more relevant questions found."), {}

    # Apply scoring logic based on session state 'scores'
    # 'scores' should store category boosts, keyword boosts etc. from last response analysis
    category_boosts = scores.get('category_boosts', {})
    keyword_matches = scores.get('keyword_matches', []) # list of keywords found in last answer

    final_scored_questions = []
    for q_item in available_questions:
        q_data = q_item['question_data']
        q_text = q_item['text']
        current_score = q_item['score']
        mapped_category = get_mapped_category(q_data)

        # --- Apply Boosts ---
        # Category boost from Groq suggestion
        current_score += category_boosts.get(mapped_category, 0)

        # Keyword boost
        if any(keyword.lower() in q_text.lower() for keyword in keyword_matches):
            current_score += 5.0 # Strong boost for keyword match

        # Completeness boost (boost categories that are lagging)
        category_completeness = completeness.get(mapped_category, 0)
        if category_completeness < DEFAULT_COMPLETENESS_TARGET:
             # Apply boost proportionally to how far below target it is
             current_score += 3.0 * (DEFAULT_COMPLETENESS_TARGET - category_completeness) / DEFAULT_COMPLETENESS_TARGET


        # --- Apply Penalties (Example) ---
        # Maybe slightly penalize asking too many from the same category consecutively? (More complex logic)
        last_category = st.session_state.get('last_question_category')
        if last_category and last_category == mapped_category:
             current_score *= 0.9 # Slight penalty for asking from same category


        q_item['score'] = current_score
        final_scored_questions.append(q_item)


    # Sort by score descending
    final_scored_questions.sort(key=lambda x: x['score'], reverse=True)

    # Select the top scoring question
    next_q_item = final_scored_questions[0]

    # Update last asked category for scoring logic
    st.session_state['last_question_category'] = get_mapped_category(next_q_item['question_data'])

    # Return the full question data tuple and the item for logging
    # ('Core Business Analysis Questions', 'Business Fundamentals', None, 'What problem does your business solve?')
    return next_q_item['question_data'], next_q_item

# --- Update Completeness ---
def update_completeness(completeness, category, quality_score):
     # Only count if answer quality is decent
    if quality_score >= 3:
         increment = 0.1 # Arbitrary increment, maybe tune this
         completeness[category] = min(1.0, completeness.get(category, 0) + increment)
    return completeness


# --- Generate Report ---
def generate_report(history, business_info):
    st.subheader("Business Assessment Report")
    st.write(f"**Business Name:** {business_info.get('name', 'N/A')}")
    st.write(f"**Identified Stage:** {business_info.get('stage', 'N/A')}")
    st.write(f"**Identified Industry:** {business_info.get('identified_industry', 'N/A')}") # Use the potentially matched industry
    st.write("---")

    conversation_summary = f"Assessment Conversation Summary for {business_info.get('name', 'the business')}:\n"
    for i, entry in enumerate(history):
        q_text = entry['question_text']
        answer = entry['answer']
        quality = entry.get('quality_score', 'N/A')
        conversation_summary += f"\n{i+1}. Q: {q_text}\nA: {answer} (Quality: {quality})\n"
        conversation_summary += f"Analysis: {entry.get('analysis', {}).get('summary', 'N/A')}\n"


    report_prompt = f"""
    Based on the following conversation log with a business owner ({business_info.get('name', 'N/A')}, identified stage: {business_info.get('stage', 'N/A')}, industry context: {business_info.get('industry_keywords', [])}), generate a concise business assessment report suitable for an early-stage investor or analyst.

    Conversation Log:
    {conversation_summary}

    Instructions for the report:
    1.  **Executive Summary:** Briefly summarize the business, its core value proposition, stage, and overall initial impression.
    2.  **Key Strengths:** Identify 2-4 potential strengths based *only* on the information provided.
    3.  **Potential Weaknesses/Risks:** Identify 2-4 potential weaknesses, risks, or areas needing clarification based *only* on the information provided (mention vague answers or low-quality responses if applicable).
    4.  **Key Areas Discussed:** Briefly list the main topics covered (e.g., Market Position, Financials, Team).
    5.  **Suggested Next Steps/Areas for Deeper Dive:** Recommend 2-3 specific areas or questions that would require further investigation based on the conversation.
    6.  **Overall Tone:** Maintain a neutral, analytical tone. Avoid making definitive judgments; focus on summarizing the provided information and identifying gaps or flags.

    Generate the report:
    """

    with st.spinner("Generating Assessment Report..."):
        report_content = get_groq_response(report_prompt)
        if report_content:
            st.markdown(report_content)
        else:
            st.error("Failed to generate the report.")
            st.write("Here's the raw conversation summary:")
            st.text_area("Conversation Summary", conversation_summary, height=300)

    # Also display completeness scores
    st.sidebar.subheader("Information Completeness")
    for area, score in st.session_state.completeness.items():
         st.sidebar.progress(score, text=f"{area}: {int(score*100)}%")


# --- Streamlit App ---
def main():
    st.title("🤖 Business Analyser Bot")
    st.caption("Powered by Groq and MongoDB")

    # --- Initialize Session State ---
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'answered_questions' not in st.session_state:
        # Store the text of questions asked to avoid repetition
        st.session_state.answered_questions = set()
    if 'current_question_data' not in st.session_state:
         st.session_state.current_question_data = None
         st.session_state.current_question_text = None
    if 'business_info' not in st.session_state:
         # name, description, stage, industry_keywords, identified_industry (from match)
        st.session_state.business_info = {}
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False
    if 'scores' not in st.session_state:
        # Store boosts, etc., derived from the *last* answer analysis
        st.session_state.scores = {'category_boosts': {}, 'keyword_matches': []}
    if 'completeness' not in st.session_state:
         # Track coverage across core areas
        st.session_state.completeness = {area: 0.0 for area in CORE_ASSESSMENT_AREAS}
    if 'all_questions' not in st.session_state:
        st.session_state.all_questions = {}
        st.session_state.all_questions_loaded = False
    if 'analysis_cache' not in st.session_state:
         # Simple cache for Groq analysis of last answer
        st.session_state.analysis_cache = {}
    if 'interview_finished' not in st.session_state:
        st.session_state.interview_finished = False
    if 'last_question_category' not in st.session_state:
        st.session_state.last_question_category = None


    # --- MongoDB Connection & Data Load ---
    question_collection = connect_mongo()
    if question_collection and not st.session_state.all_questions_loaded:
         with st.spinner("Loading question bank..."):
            st.session_state.all_questions = load_all_questions(question_collection)
            if not st.session_state.all_questions:
                 st.error("Question bank appears empty or failed to load correctly. Cannot proceed.")
                 st.stop() # Halt execution if no questions


    # --- Initial Business Setup ---
    if not st.session_state.setup_complete and st.session_state.all_questions_loaded:
        st.subheader("Let's get started! Tell me about the business:")
        name = st.text_input("Business Name:")
        description = st.text_area("Briefly describe the business and its industry:")
        # Dynamically get stage options if possible, otherwise use common list
        stage_options = ["Idea/Concept Stage", "Startup/Early Stage", "Growth Stage", "Mature Stage", "Turnaround/Restructuring Stage", "Other"]
        # Attempt to load stages dynamically from data structure
        try:
            specific_questions = st.session_state.all_questions.get("Specific Business Questions", {})
            stage_keys = list(specific_questions.get("By Business Stage", {}).keys())
            if stage_keys:
                stage_options = stage_keys + ["Other"] # Ensure 'Other' is an option
        except Exception:
             pass # Stick with default if loading fails
        stage = st.selectbox("Current Business Stage:", options=stage_options)

        if st.button("Confirm Setup"):
            if name and description and stage:
                st.session_state.business_info['name'] = name
                st.session_state.business_info['description'] = description
                st.session_state.business_info['stage'] = stage if stage != "Other" else "Startup/Early Stage" # Default if Other
                 # Simple keyword extraction (can be improved with NLP)
                st.session_state.business_info['industry_keywords'] = [kw for kw in description.lower().split() if len(kw) > 3] # Basic keyword list
                st.session_state.setup_complete = True
                st.rerun() # Rerun to move past setup
            else:
                st.warning("Please fill in all fields.")

    # --- Main Interview Loop ---
    elif st.session_state.setup_complete and not st.session_state.interview_finished:
        st.sidebar.subheader("Interview Progress")
        # Display completeness in sidebar
        for area, score in st.session_state.completeness.items():
             st.sidebar.progress(score, text=f"{area}: {int(score*100)}%")


        # Determine the next question if none is current
        if st.session_state.current_question_text is None:
             # Pass current scores and completeness to selection logic
             next_q_data, next_q_item = select_next_question(
                 st.session_state.all_questions,
                 st.session_state.answered_questions,
                 st.session_state.business_info,
                 st.session_state.scores,
                 st.session_state.completeness
            )
             # Check for end condition from selector
             if next_q_data[1] == "End":
                  st.session_state.interview_finished = True
                  st.balloons()
                  st.rerun() # Go to report generation

             elif next_q_data[1] == "Initial": # Should not happen here, but safety check
                  st.warning("Need initial setup first.")
             else:
                st.session_state.current_question_data = next_q_data # Store full data tuple
                st.session_state.current_question_text = next_q_data[-1] # Just the text for display

        # Display the current question and get answer
        if st.session_state.current_question_text:
            st.subheader("Question:")
            st.markdown(f"**{st.session_state.current_question_text}**")
            user_answer = st.text_area("Your Answer:", key=f"answer_{len(st.session_state.conversation_history)}") # Unique key forces widget refresh


            col1, col2 = st.columns([1, 1])
            with col1:
                 submit_button = st.button("Submit Answer", type="primary")
            with col2:
                 finish_early = st.button("Finish Interview Early")

            if finish_early:
                 st.session_state.interview_finished = True
                 st.warning("Finishing interview early. Report will be based on current information.")
                 st.rerun()


            if submit_button:
                 if user_answer:
                     current_q_text = st.session_state.current_question_text
                     current_q_data = st.session_state.current_question_data
                     # Add to answered set *before* potentially failing analysis
                     st.session_state.answered_questions.add(current_q_text)

                     # Process the answer with Groq
                     analysis_prompt = f"""
                     Analyze the following answer provided to the question: '{current_q_text}'

                     Answer:
                     "{user_answer}"

                     Based *only* on this answer to this question:
                     1.  Briefly summarize the key information points provided. (Summary: ...)
                     2.  Assess the quality (clarity, completeness, specificity). Provide a single score from 1 (very poor) to 5 (excellent). (Quality Score: ...)
                     3.  Identify key topics or potential flags mentioned (e.g., competition, scaling, financial metric, risk). List up to 5 keywords or short phrases. (Keywords: ...)
                     4.  Suggest 2-3 *categories* from this list [{', '.join(CORE_ASSESSMENT_AREAS)}] that seem most relevant to explore next based on the answer content. (Suggested Categories: ...)

                     Provide the output in the specified format.
                     """
                     with st.spinner("Analysing response..."):
                         analysis_result = get_groq_response(analysis_prompt)

                     analysis_data = {'summary': 'N/A', 'quality_score': 3, 'keywords': [], 'suggested_categories': []} # Defaults
                     if analysis_result:
                        st.session_state.analysis_cache = analysis_result # Cache for display if needed

                        try: # Attempt to parse Groq response (this needs robust parsing)
                           # Extremely basic parsing, likely needs regex or more structure
                           lines = analysis_result.split('\n')
                           for line in lines:
                               if line.startswith("Summary:"): analysis_data['summary'] = line.split(":", 1)[1].strip()
                               elif line.startswith("Quality Score:"): analysis_data['quality_score'] = int(line.split(":", 1)[1].strip())
                               elif line.startswith("Keywords:"): analysis_data['keywords'] = [kw.strip() for kw in line.split(":", 1)[1].split(',')]
                               elif line.startswith("Suggested Categories:"):
                                   cats_str = line.split(":", 1)[1].strip()
                                   # Filter to ensure suggested cats are valid
                                   potential_cats = [c.strip() for c in cats_str.split(',')]
                                   analysis_data['suggested_categories'] = [cat for cat in potential_cats if cat in CORE_ASSESSMENT_AREAS]

                        except Exception as e:
                           st.warning(f"Could not fully parse analysis response: {e}. Using defaults.")
                           st.text(analysis_result) # Show raw if parse failed


                     # --- Update State ---
                     # Store history entry
                     st.session_state.conversation_history.append({
                         'question_text': current_q_text,
                         'question_data': current_q_data, # Store the tuple for context
                         'answer': user_answer,
                         'analysis': analysis_data
                     })

                     # Update scores for next selection
                     category_boosts = defaultdict(float)
                     for cat in analysis_data.get('suggested_categories', []):
                         category_boosts[cat] += 3.0 # Apply boost value
                     st.session_state.scores = {
                         'category_boosts': dict(category_boosts), # Convert back to regular dict
                         'keyword_matches': analysis_data.get('keywords', [])
                     }


                      # Update completeness
                     mapped_cat = get_mapped_category(current_q_data)
                     st.session_state.completeness = update_completeness(
                          st.session_state.completeness,
                          mapped_cat,
                          analysis_data['quality_score']
                     )


                     # Reset current question to trigger selection of the next one
                     st.session_state.current_question_data = None
                     st.session_state.current_question_text = None
                     st.session_state.analysis_cache = {} # Clear cache

                     # Optional: Check if completeness target is met across enough categories
                     completed_areas = sum(1 for score in st.session_state.completeness.values() if score >= DEFAULT_COMPLETENESS_TARGET)
                     # Example end condition: at least 5 areas complete
                     if completed_areas >= 5: # Or calculate overall average completeness
                        avg_completeness = sum(st.session_state.completeness.values()) / len(CORE_ASSESSMENT_AREAS)
                        if avg_completeness >= DEFAULT_COMPLETENESS_TARGET :
                             st.success("Sufficient information gathered.")
                             st.session_state.interview_finished = True
                             st.balloons()

                     st.rerun() # Rerun to display next question or finish
                 else:
                     st.warning("Please provide an answer before submitting.")

        # Debug/Info panel (optional)
        with st.sidebar.expander("Debug Info"):
            st.write("Current Business Info:", st.session_state.business_info)
            st.write("Answered Questions Count:", len(st.session_state.answered_questions))
            st.write("Scores for Next Question:", st.session_state.scores)
            # st.write("Last Analysis Cache:", st.session_state.analysis_cache)
            st.write("Last Asked Q Category:", st.session_state.last_question_category)


    # --- Report Generation ---
    elif st.session_state.interview_finished:
        if not st.session_state.conversation_history:
             st.warning("No conversation history to generate a report from.")
        else:
             generate_report(st.session_state.conversation_history, st.session_state.business_info)

        if st.button("Start New Interview"):
             # Clear relevant session state parts
             keys_to_clear = [
                 'conversation_history', 'answered_questions', 'current_question_data',
                 'current_question_text', 'business_info', 'setup_complete',
                 'scores', 'completeness', 'analysis_cache', 'interview_finished',
                 'last_question_category'
             ]
             for key in keys_to_clear:
                 if key in st.session_state:
                     del st.session_state[key]
             st.rerun()


if __name__ == "__main__":
    main()

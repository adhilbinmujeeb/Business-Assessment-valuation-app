import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    st.error("Failed to initialize AI model. Please check your API key.")

# MongoDB connection
try:
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['your_database']
    questions_collection = db['investor_questions']
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    st.error("Failed to connect to database. Please check your connection settings.")

def get_initial_context() -> str:
    """Returns the initial context for the AI model."""
    return """You are an experienced venture capitalist and investor. Your goal is to thoroughly 
    understand the business presented to you and make informed decisions. Ask questions strategically 
    to build a complete picture of the business opportunity. Be professional but conversational."""

def validate_business_info(info: Dict) -> bool:
    """Validates the business information."""
    required_fields = ['name', 'industry', 'stage']
    return all(field in info and info[field] for field in required_fields)

def get_business_basics() -> Dict:
    """Gets basic business information from the user."""
    st.header("Let's start with the basics of your business")
    
    col1, col2 = st.columns(2)
    
    with col1:
        business_name = st.text_input("What is your business name?", 
                                    help="Enter your company's legal name")
        industry = st.selectbox("What industry are you in?", 
                              ["Technology", "Healthcare", "Retail", "Manufacturing", "Services", "Other"],
                              help="Select the primary industry your business operates in")
    
    with col2:
        stage = st.selectbox("What stage is your business in?",
                           ["Idea Stage", "Pre-Seed", "Seed", "Series A", "Growth", "Mature"],
                           help="Select your current business development stage")
        funding_needed = st.number_input("How much funding are you seeking? (in USD)",
                                       min_value=0,
                                       help="Enter the amount of funding you're seeking")
    
    return {
        "name": business_name,
        "industry": industry,
        "stage": stage,
        "funding_needed": funding_needed
    }

def evaluate_responses(qa_history: List[Dict]) -> Dict:
    """Evaluates the Q&A history to determine if enough information has been gathered."""
    evaluation_prompt = f"""
    Based on these Q&A interactions:
    {json.dumps(qa_history, indent=2)}
    
    Please evaluate:
    1. Do we have enough information to generate a comprehensive investment report?
    2. What key areas still need more information?
    3. How many more questions are needed (provide a number)?
    4. What is the confidence level in the current information (0-100)?
    
    Respond in JSON format with keys: 
    - 'enough_info' (boolean)
    - 'missing_areas' (list)
    - 'questions_needed' (int)
    - 'confidence_level' (int)
    """
    
    try:
        response = model.generate_content(evaluation_prompt)
        eval_result = json.loads(response.text)
        return eval_result
    except Exception as e:
        logger.error(f"Failed to evaluate responses: {e}")
        return {
            "enough_info": False,
            "missing_areas": ["general"],
            "questions_needed": 3,
            "confidence_level": 0
        }

def get_relevant_questions(business_info: Dict, previous_responses: List[Dict]) -> List[Dict]:
    """Gets relevant questions based on business info and previous responses."""
    try:
        context = f"""
        Business Name: {business_info['name']}
        Industry: {business_info['industry']}
        Stage: {business_info['stage']}
        Funding Needed: ${business_info['funding_needed']:,.2f}
        Previous Responses: {json.dumps(previous_responses, indent=2)}
        
        Based on this information, what are the most critical areas we need to explore next?
        Focus on gathering information about: {', '.join(evaluate_responses(previous_responses)['missing_areas'])}
        """
        
        response = model.generate_content(context)
        category = response.text
        
        eval_result = evaluate_responses(previous_responses)
        questions = questions_collection.find({
            "category": {"$regex": category, "$options": "i"},
            "stage": business_info['stage']
        }).limit(eval_result['questions_needed'])
        
        return list(questions)
    except Exception as e:
        logger.error(f"Failed to get relevant questions: {e}")
        return []

def generate_report(business_info: Dict, qa_history: List[Dict]) -> str:
    """Generates a comprehensive investment report."""
    report_prompt = f"""
    Based on the following information about {business_info['name']}, create a comprehensive 
    investment analysis report:
    
    Business Details:
    - Industry: {business_info['industry']}
    - Stage: {business_info['stage']}
    - Funding Needed: ${business_info['funding_needed']:,.2f}
    
    Q&A Summary:
    {json.dumps(qa_history, indent=2)}
    
    Please provide a detailed report including:
    1. Executive Summary
    2. Business Model Analysis
    3. Market Opportunity
    4. Risk Assessment
    5. Growth Potential
    6. Financial Projections
    7. Team Assessment
    8. Recommendations for both investors and business owners
    
    Format the report in markdown with appropriate headers and bullet points.
    """
    
    try:
        response = model.generate_content(report_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return "Failed to generate report. Please try again."

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="AI Investor Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("AI Investor Assistant")
    st.write("Let me help evaluate your business proposition")
    
    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'basics'
        st.session_state.qa_history = []
        st.session_state.business_info = None
    
    # Business Basics Stage
    if st.session_state.stage == 'basics':
        st.session_state.business_info = get_business_basics()
        if st.button("Start Assessment", type="primary"):
            if validate_business_info(st.session_state.business_info):
                st.session_state.stage = 'questioning'
                st.experimental_rerun()
            else:
                st.error("Please fill in all required fields.")
    
    # Questioning Stage
    elif st.session_state.stage == 'questioning':
        st.write(f"Evaluating {st.session_state.business_info['name']}")
        
        eval_result = evaluate_responses(st.session_state.qa_history)
        
        # Progress bar
        progress = min(100, (len(st.session_state.qa_history) / 
                           (len(st.session_state.qa_history) + eval_result['questions_needed'])) * 100)
        st.progress(progress)
        
        if eval_result['enough_info']:
            st.success("We have gathered sufficient information to generate a detailed report!")
            if st.button("Generate Report", type="primary"):
                st.session_state.stage = 'report'
                st.experimental_rerun()
        else:
            st.info(f"We need more information about: {', '.join(eval_result['missing_areas'])}")
            st.write(f"Confidence Level: {eval_result['confidence_level']}%")
            
            relevant_questions = get_relevant_questions(
                st.session_state.business_info, 
                st.session_state.qa_history
            )
            
            for question in relevant_questions:
                with st.expander(f"Q: {question['question']}"):
                    response = st.text_area("Your Answer:", key=f"q_{question['_id']}")
                    if response:
                        st.session_state.qa_history.append({
                            "question": question['question'],
                            "answer": response,
                            "timestamp": datetime.now().isoformat()
                        })
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Continue Assessment", type="primary"):
                    st.experimental_rerun()
            with col2:
                if st.button("Start Over", type="secondary"):
                    st.session_state.clear()
                    st.experimental_rerun()
    
    # Report Stage
    elif st.session_state.stage == 'report':
        with st.spinner("Generating report..."):
            report = generate_report(st.session_state.business_info, st.session_state.qa_history)
        
        st.markdown("## Investment Analysis Report")
        st.markdown(report)
        
        # Add download button for the report
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"investment_report_{st.session_state.business_info['name']}.md",
            mime="text/markdown"
        )
        
        if st.button("Start New Assessment", type="primary"):
            st.session_state.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    main()

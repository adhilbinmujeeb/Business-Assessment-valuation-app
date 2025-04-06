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
        seeking_funding = st.checkbox("Are you seeking investment?", value=False)
        funding_needed = None
        if seeking_funding:
            funding_needed = st.number_input("How much funding are you seeking? (in USD)",
                                           min_value=0,
                                           help="Enter the amount of funding you're seeking")
    
    return {
        "name": business_name,
        "industry": industry,
        "stage": stage,
        "funding_needed": funding_needed,
        "seeking_funding": seeking_funding
    }

def evaluate_responses(qa_history: List[Dict]) -> Dict:
    """Evaluates the Q&A history to determine if enough information has been gathered."""
    evaluation_prompt = f"""
    You are an experienced business analyst evaluating a Q&A session between an investor and entrepreneur.
    
    Q&A History:
    {json.dumps(qa_history, indent=2)}
    
    Please evaluate:
    1. Do we have enough information to generate a comprehensive investment report?
    2. What key areas still need more information?
    3. How many more questions are needed (provide a number)?
    4. What is the confidence level in the current information (0-100)?
    5. What specific aspects need more detail?
    
    Respond in JSON format with keys: 
    - 'enough_info' (boolean)
    - 'missing_areas' (list)
    - 'questions_needed' (int)
    - 'confidence_level' (int)
    - 'needs_more_detail' (list of specific aspects)
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
            "confidence_level": 0,
            "needs_more_detail": ["business model", "market opportunity"]
        }

def get_relevant_questions(business_info: Dict, previous_responses: List[Dict]) -> List[Dict]:
    """Gets relevant questions based on business info and previous responses."""
    # Default questions that will be used if API calls fail
    default_questions = [
        {
            "question": f"Tell me about your business model. How do you make money?",
            "category": "Business Model",
            "follow_up": "Could you elaborate on your main revenue streams?"
        },
        {
            "question": f"What problem does your business solve in the {business_info['industry']} industry?",
            "category": "Market Problem",
            "follow_up": "How is your solution different from existing alternatives?"
        },
        {
            "question": f"Who are your target customers and how do you reach them?",
            "category": "Customer Acquisition",
            "follow_up": "What's your customer acquisition cost and lifetime value?"
        }
    ]

    try:
        # Create a context for Gemini to generate personalized questions
        context = """
        Generate 3-5 specific business evaluation questions for a {industry} company at the {stage} stage.
        Focus on key business metrics, market opportunity, and growth potential.
        
        Format: Return only a JSON array of question objects with the following structure:
        [
            {
                "question": "The question text",
                "category": "Question category",
                "follow_up": "Follow-up question"
            }
        ]
        """.format(
            industry=business_info['industry'],
            stage=business_info['stage']
        )
        
        response = model.generate_content(context)
        if not response or not response.text:
            logger.warning("Empty response from Gemini API")
            return default_questions

        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            if not response_text.startswith('['):
                response_text = response_text[response_text.find('['):]
            if not response_text.endswith(']'):
                response_text = response_text[:response_text.rfind(']')+1]
            
            questions = json.loads(response_text)
            if not isinstance(questions, list) or len(questions) == 0:
                logger.warning("Invalid questions format from Gemini API")
                return default_questions

            # Validate each question has required fields
            valid_questions = []
            for q in questions:
                if isinstance(q, dict) and 'question' in q and 'category' in q:
                    valid_questions.append({
                        'question': q['question'],
                        'category': q['category'],
                        'follow_up': q.get('follow_up', 'Could you elaborate on that?')
                    })

            if len(valid_questions) < 3:
                logger.warning(f"Not enough valid questions ({len(valid_questions)}), using defaults")
                return default_questions

            return valid_questions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated questions: {e}")
            return default_questions

    except Exception as e:
        logger.error(f"Failed to get relevant questions: {e}")
        return default_questions

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
        st.session_state.current_questions = []
        st.session_state.answered_questions = set()
    
    # Business Basics Stage
    if st.session_state.stage == 'basics':
        st.session_state.business_info = get_business_basics()
        if st.button("Start Assessment", type="primary"):
            if validate_business_info(st.session_state.business_info):
                st.session_state.stage = 'questioning'
                # Generate initial questions
                st.session_state.current_questions = get_relevant_questions(
                    st.session_state.business_info,
                    st.session_state.qa_history
                )
                st.rerun()
            else:
                st.error("Please fill in all required fields.")
    
    # Questioning Stage
    elif st.session_state.stage == 'questioning':
        st.write(f"Evaluating {st.session_state.business_info['name']}")
        
        # Always ensure we have questions to display
        if not st.session_state.current_questions:
            st.session_state.current_questions = get_relevant_questions(
                st.session_state.business_info,
                st.session_state.qa_history
            )
        
        # Display current progress
        eval_result = evaluate_responses(st.session_state.qa_history)
        total_questions = max(5, len(st.session_state.qa_history) + eval_result['questions_needed'])
        progress = min(100, max(0, (len(st.session_state.qa_history) / total_questions) * 100))
        st.progress(int(progress))
        
        # Display questions
        st.write("Let's start with some key questions about your business:")
        
        for idx, question in enumerate(st.session_state.current_questions):
            if question['question'] not in st.session_state.answered_questions:
                with st.expander(f"Q: {question['question']}", expanded=True):
                    response = st.text_area(
                        "Your Answer:",
                        key=f"q_{idx}_{hash(question['question'])}"
                    )
                    
                    if st.button("Submit Answer", key=f"submit_{idx}"):
                        if response:
                            st.session_state.qa_history.append({
                                "question": question['question'],
                                "answer": response,
                                "category": question['category'],
                                "timestamp": datetime.now().isoformat()
                            })
                            st.session_state.answered_questions.add(question['question'])
                            
                            # Check if answer needs follow-up
                            follow_up = st.checkbox(
                                "Would you like to provide more detail?",
                                key=f"fu_{idx}_{hash(question['question'])}"
                            )
                            
                            if follow_up:
                                follow_up_response = st.text_area(
                                    "Additional Details:",
                                    key=f"f_{idx}_{hash(question['question'])}"
                                )
                                if follow_up_response:
                                    st.session_state.qa_history.append({
                                        "question": question['follow_up'],
                                        "answer": follow_up_response,
                                        "category": question['category'],
                                        "timestamp": datetime.now().isoformat()
                                    })
                            st.rerun()
        
        # Show progress information
        st.info(f"We need more information about: {', '.join(eval_result['missing_areas'])}")
        st.write(f"Confidence Level: {eval_result['confidence_level']}%")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue Assessment", type="primary"):
                # Generate new questions for the next round
                st.session_state.current_questions = get_relevant_questions(
                    st.session_state.business_info,
                    st.session_state.qa_history
                )
                st.rerun()
        with col2:
            if st.button("Start Over", type="secondary"):
                st.session_state.clear()
                st.rerun()
                
        # Check if we have enough information for a report
        if eval_result['enough_info']:
            st.success("We have gathered sufficient information to generate a detailed report!")
            if st.button("Generate Report", type="primary"):
                st.session_state.stage = 'report'
                st.rerun()
    
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
            st.rerun()

if __name__ == "__main__":
    main()

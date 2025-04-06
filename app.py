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
    # Count questions by category
    categories = {
        'General': 0,
        'Business Model': 0,
        'Industry': 0
    }
    
    for qa in qa_history:
        category = qa['category']
        if category == 'General':
            categories['General'] += 1
        elif category in ['Business Model', 'Business Metrics', 'Business Operations']:
            categories['Business Model'] += 1
        elif category in ['Market Problem', 'Competition', 'Industry Trends']:
            categories['Industry'] += 1
    
    # Define minimum requirements
    min_general = 3
    min_business = 3
    min_industry = 3
    
    missing_areas = []
    if categories['General'] < min_general:
        missing_areas.append('general')
    if categories['Business Model'] < min_business:
        missing_areas.append('business model')
    if categories['Industry'] < min_industry:
        missing_areas.append('industry specifics')
    
    # Calculate confidence level based on completeness
    total_questions_needed = min_general + min_business + min_industry
    total_questions_answered = sum(categories.values())
    confidence_level = min(100, int((total_questions_answered / total_questions_needed) * 100))
    
    return {
        "enough_info": len(missing_areas) == 0,
        "missing_areas": missing_areas,
        "questions_needed": max(0, total_questions_needed - total_questions_answered),
        "confidence_level": confidence_level,
        "needs_more_detail": missing_areas
    }

def get_relevant_questions(business_info: Dict, previous_responses: List[Dict]) -> List[Dict]:
    """Gets relevant questions based on business info and previous responses."""
    # Default questions organized by category
    default_general_questions = [
        {
            "question": "Could you give me an overview of your business and what inspired you to start it?",
            "category": "General",
            "follow_up": "What is your long-term vision for the company?"
        },
        {
            "question": "How long have you been operating and what major milestones have you achieved?",
            "category": "General",
            "follow_up": "What are your next major milestones?"
        },
        {
            "question": "Tell me about your team and their backgrounds.",
            "category": "General",
            "follow_up": "What key roles are you looking to fill in the next 12 months?"
        }
    ]
    
    default_business_questions = [
        {
            "question": f"Tell me about your business model. How do you make money?",
            "category": "Business Model",
            "follow_up": "Could you elaborate on your main revenue streams?"
        },
        {
            "question": "What are your current key performance indicators (KPIs)?",
            "category": "Business Metrics",
            "follow_up": "How have these metrics changed over the past year?"
        },
        {
            "question": "What is your customer acquisition strategy?",
            "category": "Business Operations",
            "follow_up": "What's your customer acquisition cost and lifetime value?"
        }
    ]
    
    default_industry_questions = [
        {
            "question": f"What problem does your business solve in the {business_info['industry']} industry?",
            "category": "Market Problem",
            "follow_up": "How is your solution different from existing alternatives?"
        },
        {
            "question": f"Who are your main competitors in the {business_info['industry']} space?",
            "category": "Competition",
            "follow_up": "What gives you a competitive advantage?"
        },
        {
            "question": f"What are the key trends affecting the {business_info['industry']} industry?",
            "category": "Industry Trends",
            "follow_up": "How are you positioned to capitalize on these trends?"
        }
    ]

    if not model:
        logger.error("Gemini model not initialized")
        return default_general_questions

    try:
        # Analyze previous responses to determine what type of questions to ask next
        categories_asked = set(qa['category'] for qa in previous_responses)
        general_questions_asked = len([qa for qa in previous_responses if qa['category'] == 'General'])
        
        # Determine which type of questions to ask based on previous responses
        if general_questions_asked < 3:
            question_type = "general"
            default_questions = default_general_questions
            context = f"""
            You are an experienced investor evaluating a new business opportunity.
            Generate 3 general questions to understand the basic foundation of the business.
            Focus on understanding the overall business, team, and vision.
            """
        elif 'Business Model' not in categories_asked or 'Business Metrics' not in categories_asked:
            question_type = "business"
            default_questions = default_business_questions
            context = f"""
            You are an experienced investor evaluating a {business_info['stage']} stage company.
            Generate 3 specific questions about their business model, metrics, and operations.
            Focus on understanding how the business works and its performance.
            """
        else:
            question_type = "industry"
            default_questions = default_industry_questions
            context = f"""
            You are an experienced investor evaluating a company in the {business_info['industry']} industry.
            Generate 3 specific questions about their market position, competition, and industry trends.
            Focus on understanding their competitive advantage and market opportunity.
            """
        
        context += """
        Format your response as a JSON array like this:
        [
            {
                "question": "Your question here?",
                "category": "Question Category",
                "follow_up": "Follow-up question here?"
            }
        ]
        Only return the JSON array, no other text.
        """
        
        response = model.generate_content(context)
        if not response or not response.text:
            logger.warning(f"Empty response from Gemini API, using default {question_type} questions")
            return default_questions

        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start == -1 or end == 0:
                logger.warning(f"No JSON array found in response, using default {question_type} questions")
                return default_questions
                
            json_text = response_text[start:end]
            questions = json.loads(json_text)
            
            if not isinstance(questions, list):
                logger.warning(f"Response is not a list, using default {question_type} questions")
                return default_questions

            # Validate each question has required fields
            valid_questions = []
            for q in questions:
                if isinstance(q, dict) and 'question' in q and 'category' in q:
                    # Set the category based on the current question type
                    if question_type == "general":
                        q['category'] = "General"
                    valid_questions.append({
                        'question': q['question'],
                        'category': q['category'],
                        'follow_up': q.get('follow_up', 'Could you elaborate on that?')
                    })

            # If we don't have enough valid questions, add some defaults
            while len(valid_questions) < 3:
                for q in default_questions:
                    if len(valid_questions) < 3 and q not in valid_questions:
                        valid_questions.append(q)

            return valid_questions[:3]  # Return exactly 3 questions

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
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
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
        
        with col2:
            # Show progress information
            st.info(f"Areas needing more info:\n{', '.join(eval_result['missing_areas'])}")
            st.write(f"Confidence Level: {eval_result['confidence_level']}%")
            
            # Navigation buttons
            if st.button("Continue Assessment", type="primary"):
                # Generate new questions for the next round
                st.session_state.current_questions = get_relevant_questions(
                    st.session_state.business_info,
                    st.session_state.qa_history
                )
                st.rerun()
            
            if st.button("Start Over", type="secondary"):
                st.session_state.clear()
                st.rerun()
            
            # Check if we have enough information for a report
            if eval_result['enough_info']:
                st.success("Ready to generate report!")
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

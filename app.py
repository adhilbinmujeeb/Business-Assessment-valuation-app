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
                next_question = groq_qna(next_question_prompt).strip()
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

import streamlit as st
import torch
from transformers import pipeline
import re
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

class TaskExtractor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_pipeline = pipeline("question-answering",
            model="distilbert-base-uncased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1)

    def extract_with_qa(self, text: str) -> Dict:
        context = text

        # Ask specific questions to extract information
        questions = [
            "What task needs to be completed?",
            "What action should be taken?",
            "When should this be done?",
            "What time is mentioned?",
            "What is the deadline?"
        ]

        answers = {}
        for question in questions:
            try:
                result = self.qa_pipeline(question=question, context=context)
                if result['score'] > 0.1:  # Confidence threshold
                    answers[question] = result['answer']
            except:
                continue
        
        task_answers = [answers.get(q) for q in questions[:2] if answers.get(q)]
        task = task_answers[0] if task_answers else self._extract_task_with_rules(text)

        # Extract time and date
        time_answers = [answers.get(q) for q in questions[2:] if answers.get(q)]
        time_info = time_answers[0] if time_answers else None

        # Parse time and date from extracted information
        extracted_time = self._extract_time_from_text(time_info or text)
        extracted_date = self._extract_date_from_text(time_info or text)

        return {
            "task": task,
            "time": extracted_time,
            "date": extracted_date
        }

    def _extract_task_with_rules(self, text: str) -> str:
        # Remove time and date references
        cleaned = re.sub(r'at\s+\d{1,2}:?\d{0,2}\s*(am|pm)?', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d{1,2}:?\d{0,2}\s*(am|pm)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'tomorrow|today|next\s+week|monday|tuesday|wednesday|thursday|friday|saturday|sunday',
                        '', cleaned, flags=re.IGNORECASE)

        # Remove task indicators
        cleaned = re.sub(r'^(need to|have to|must|should|remember to|don\'t forget to)\s+',
                        '', cleaned, flags=re.IGNORECASE)

        # Clean up
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'^[,\s]+|[,\s]+$', '', cleaned)

        return cleaned

    def _extract_time_from_text(self, text: str) -> Optional[str]:
        """Extract time from text"""
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(am|pm)',
            r'(\d{1,2})\s*(am|pm)',
            r'at\s+(\d{1,2}):(\d{2})',
            r'at\s+(\d{1,2})\s*(am|pm)'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).replace('at ', '').strip()
        return None

    def _extract_date_from_text(self, text: str) -> str:
        """Extract and calculate date from text"""
        today = datetime.now()

        date_patterns = {
            r'tomorrow': 1,
            r'today': 0,
            r'next week': 7,
            r'monday': self._days_until_weekday(0),
            r'tuesday': self._days_until_weekday(1),
            r'wednesday': self._days_until_weekday(2),
            r'thursday': self._days_until_weekday(3),
            r'friday': self._days_until_weekday(4),
            r'saturday': self._days_until_weekday(5),
            r'sunday': self._days_until_weekday(6)
        }

        for pattern, offset in date_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if callable(offset):
                    target_date = today + timedelta(days=offset())
                else:
                    target_date = today + timedelta(days=offset)
                return target_date.strftime('%d/%m/%Y')

        return today.strftime('%d/%m/%Y')

    def _days_until_weekday(self, target_weekday):
        """Calculate days until target weekday"""
        def calculate():
            today = datetime.now().weekday()
            days_ahead = target_weekday - today
            if days_ahead <= 0:
                days_ahead += 7
            return days_ahead
        return calculate

    def extract_task(self, text: str, method="qa") -> Dict:
        return self.extract_with_qa(text)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Task Extractor",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Task Extractor")
    st.markdown("Extract tasks, times, and dates from natural language text using AI")
    
    # Initialize the extractor with caching
    @st.cache_resource
    def load_extractor():
        return TaskExtractor()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses AI to extract:
        - **Tasks**: What needs to be done
        - **Time**: When to do it
        - **Date**: Which day to do it
        
        **Example inputs:**
        - "Need to call mom at 3pm tomorrow"
        - "Meeting with client on Friday at 2:30pm"
        - "Remember to buy groceries today"
        """)
        
        st.header("⚙️ Settings")
        show_raw_output = st.checkbox("Show raw extraction details", value=False)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Text")
        user_input = st.text_area(
            "Enter your task description:",
            placeholder="e.g., Need to finish the report by Friday at 5pm",
            height=150
        )
        
        extract_button = st.button("🔍 Extract Task", type="primary")
        
        # Example buttons
        st.markdown("**Quick examples:**")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("📞 Call example"):
                st.session_state.example_text = "Need to call the dentist tomorrow at 2pm"
        
        with example_col2:
            if st.button("📧 Email example"):
                st.session_state.example_text = "Send project update email on Friday morning"
        
        with example_col3:
            if st.button("🛒 Shopping example"):
                st.session_state.example_text = "Buy groceries after work today at 6:30pm"
        
        # Use example text if selected
        if 'example_text' in st.session_state:
            user_input = st.session_state.example_text
            del st.session_state.example_text
    
    with col2:
        st.header("Quick Stats")
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.metric("Words", word_count)
            st.metric("Characters", char_count)
    
    # Process the input
    if extract_button and user_input:
        with st.spinner("Extracting task information..."):
            try:
                extractor = load_extractor()
                result = extractor.extract_task(user_input)
                
                # Display results
                st.header("📋 Extraction Results")
                
                # Main results in cards
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown("### 🎯 Task")
                    task = result.get('task', 'Not found')
                    st.success(task if task else "No task detected")
                
                with result_col2:
                    st.markdown("### ⏰ Time")
                    time = result.get('time', 'Not specified')
                    if time:
                        st.info(time)
                    else:
                        st.warning("No time specified")
                
                with result_col3:
                    st.markdown("### 📅 Date")
                    date = result.get('date', 'Not specified')
                    if date:
                        st.info(date)
                    else:
                        st.warning("No date specified")
                
                # Formatted output
                st.header("📝 Formatted Task")
                formatted_task = f"**Task:** {result.get('task', 'Not specified')}\n"
                if result.get('date'):
                    formatted_task += f"**Date:** {result.get('date')}\n"
                if result.get('time'):
                    formatted_task += f"**Time:** {result.get('time')}\n"
                
                st.markdown(formatted_task)
                
                # Copy to clipboard functionality
                st.code(formatted_task, language=None)
                
                # Raw output (if enabled)
                if show_raw_output:
                    st.header("🔧 Raw Output")
                    st.json(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("**Possible solutions:**")
                st.markdown("- Check your internet connection")
                st.markdown("- Try a simpler input text")
                st.markdown("- Refresh the page and try again")
    
    elif extract_button and not user_input:
        st.warning("Please enter some text to extract tasks from!")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Powered by Hugging Face Transformers")

if __name__ == "__main__":
    main()
import streamlit as st

def main():
    st.title("🧪 Feedback Test")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()
    
    # Add a test message
    if st.button("Add Test Message"):
        import pandas as pd
        test_data = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        
        st.session_state.messages.append({
            "role": "user",
            "content": "Test question"
        })
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Test response",
            "data": test_data,
            "sql_query": "SELECT * FROM test"
        })
    
    # Display messages with debug
    for i, message in enumerate(st.session_state.messages):
        message_id = f"msg_{i}"
        
        st.write(f"**Message {i} ({message['role']}):**")
        st.write(message["content"])
        
        if "data" in message:
            st.dataframe(message["data"])
        
        # Debug info
        st.write(f"🔧 DEBUG - Message {i}:")
        st.write(f"- Role: {message['role']}")
        st.write(f"- Has data: {'data' in message}")
        st.write(f"- Message ID: {message_id}")
        st.write(f"- In feedback_given: {message_id in st.session_state.feedback_given}")
        
        # Simple feedback test
        if (message["role"] == "assistant" and 
            "data" in message and
            message_id not in st.session_state.feedback_given):
            
            st.write("🎯 SHOULD SHOW FEEDBACK WIDGET!")
            
            rating = st.slider(f"Rate message {i}", 1, 5, 3, key=f"rating_{message_id}")
            
            if st.button(f"Submit Feedback {i}", key=f"submit_{message_id}"):
                st.success("Feedback submitted!")
                st.session_state.feedback_given.add(message_id)
                st.rerun()

if __name__ == "__main__":
    main()
import streamlit as st
from langchainchatbot_backend import PDFDocumentProcessor, CustomRAGProcessor, ChatLog

def main():
    st.title('Nurse-City Health Education Chatbot')
    st.sidebar.title("Settings")
    directory = st.sidebar.text_input("Enter the directory of PDF files:", "path/to/pdf_directory")
    storage_directory = st.sidebar.text_input("Enter the directory for storing processed files:", "path/to/storage")
    model_name = st.sidebar.text_input("Enter the model name:", "emilyalsentzer/Bio_ClinicalBERT")

    # Initialize chat log
    chat_log = ChatLog()

    process_button = st.sidebar.button("Process PDFs")
    if process_button:
        processor = PDFDocumentProcessor(directory, storage_directory)
        processor.extract_text()
        st.success("PDFs processed and stored successfully.")

    # Toggle for displaying chat history
    show_history = st.sidebar.checkbox("Show Chat History")

    # Display chat history if toggled on
    if show_history:
        st.subheader("Chat History")
        history = chat_log.display_history()
        if history:
            chat_pairs = history.split("\n\n")
            for pair in chat_pairs:
                if pair.startswith("User:"):
                    st.text(pair)
                elif pair.startswith("Bot:"):
                    st.text(pair)
        else:
            st.warning("No chat history available.")

    query = st.text_input("You:", "")
    if st.button("Send"):
        if query:
            rag = CustomRAGProcessor(model_name, storage_directory)
            answer = rag.generate_answer(query)
            st.subheader("Bot:")

            # Display bot message in a chat bubble
            with st.container():
                st.write(answer)

            # Add user query and bot answer to chat log
            chat_log.add_message(query, answer)
        else:
            st.warning("Please enter a question to generate an answer.")

if __name__ == "__main__":
    main()

# def main():
#     st.title('Artificial Intelligence Chatbot')
#     st.sidebar.title("Settings")
#     directory = st.sidebar.text_input("Enter the directory of PDF files:", "path/to/pdf_directory")
#     storage_directory = st.sidebar.text_input("Enter the directory for storing processed files:", "path/to/storage")
#     model_name = st.sidebar.text_input("Enter the model name:", "gpt2")
    
#     process_button = st.sidebar.button("Process PDFs")
#     if process_button:
#         processor = PDFDocumentProcessor(directory, storage_directory)
#         processor.extract_text()
#         st.success("PDFs processed and stored successfully.")
    
#     query = st.text_input("Enter your question:")
#     generate_button = st.button("Generate Answer")
#     if generate_button:
#         if query:
#             rag = CustomRAGProcessor(model_name, storage_directory)
#             answer = rag.generate_answer(query)
#             st.write("Answer:", answer)
#         else:
#             st.write("Please enter a question to generate an answer.")

# if __name__ == "__main__":
#     main()



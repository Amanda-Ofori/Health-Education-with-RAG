from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Assuming the custom RAG processor and other classes have been imported correctly
from langchainchatbot_backend import CustomRAGProcessor, ChatLog  # Placeholder for your actual import statement

# Initialize the chat processor
model_name = 'emilyalsentzer/Bio_ClinicalBERT'  # Example model name
storage_directory = 'processed_documents'  # Adjust accordingly
processor = CustomRAGProcessor(model_name, storage_directory)
chat_log = ChatLog()

@app.route("/")
def home():
    # Add your quotes here
    quotes = [
        "Ladies, it is allright to say NO TO SEX when you don't want it",
        "A vitamin C a day keeps the cold, and sore throats away",
        # Add more quotes as needed
    ]
    return render_template("home.html", quotes=quotes)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_message = request.form["message"]
        context_keywords = []  # Dynamically determine or manage your context
        bot_response = processor.generate_answer(user_message, context_keywords)
        chat_log.add_message(user_message, bot_response)
        return redirect(url_for('chat'))
    history = chat_log.display_history()
    return render_template("chat.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)

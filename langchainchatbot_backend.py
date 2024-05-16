import os
import fitz  # PyMuPDF
import re
import random
import torch
import glob
import numpy as np
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

class ChatLog:
    def __init__(self, log_file="chat_history.txt"):
        self.log_file = log_file

    def add_message(self, user_message, bot_response):
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(f"User: {user_message}\n")
            file.write(f"Bot: {bot_response}\n\n")

    def display_history(self):
        try:
            with open(self.log_file, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return "No chat history available."

class PDFDocumentProcessor:
    def __init__(self, directory, storage_directory):
        self.directory = directory
        self.storage_directory = storage_directory
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.ensure_directory_exists(self.storage_directory)

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def extract_text(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.pdf'):
                path = os.path.join(self.directory, filename)
                with fitz.open(path) as doc:
                    text = " ".join(page.get_text() for page in doc)
                tokenized_text = self.text_splitter.split_text(text)
                self.store_document(tokenized_text, filename)

    def store_document(self, document, filename):
        file_path = os.path.join(self.storage_directory, filename.replace('.pdf', '.txt'))
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in document:
                file.write("%s\n" % item)

class SemanticSearch:
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze()

    def cosine_similarity(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)

class CustomRAGProcessor:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def __init__(self, model_name, storage_directory):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.storage_directory = storage_directory
        self.tfidf_vectorizer = TfidfVectorizer()
        self.semantic_search = SemanticSearch(model_name)
        self.answer_cache = {}

    def deduplicate_text(self, text):
        sentences = text.split('. ')
        seen_sentences = set()
        unique_sentences = []
        for sentence in sentences:
            cleaned_sentence = re.sub(r'\s+', ' ', sentence.strip())
            if cleaned_sentence and cleaned_sentence not in seen_sentences:
                unique_sentences.append(cleaned_sentence)
                seen_sentences.add(cleaned_sentence)
        return '. '.join(unique_sentences)

    def retrieve_documents(self, query, context_keywords=None):
        if context_keywords is None:
            context_keywords = []

        documents = []
        for file_path in glob.glob(os.path.join(self.storage_directory, '*.txt')):
            with open(file_path, 'r', encoding='utf-8') as file:
                document = file.read()
                if any(keyword in document for keyword in context_keywords):
                    documents.append(document)

        if not documents:
            print("No documents matched the context keywords.")
            return ""

        if documents:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            query_vector = self.tfidf_vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            relevant_docs_indices = np.argsort(cosine_similarities)[-3:]  # Top 3 documents
            
            # Re-rank with ClinicalBERT
            bert_embeddings = [self.semantic_search.encode(doc) for doc in documents]
            query_embedding = self.semantic_search.encode(query)
            scores = [self.semantic_search.cosine_similarity(query_embedding, doc_emb) for doc_emb in bert_embeddings]
            top_docs_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            selected_documents = ' '.join([documents[i] for i in top_docs_indices])

            # Debug: Print selected documents
            print("Selected Documents:", selected_documents)

            return self.deduplicate_text(selected_documents)
        return ""

    def generate_answer(self, query, context_keywords=None, max_length=300):
        if query in self.answer_cache and self.answer_cache[query][1] == context_keywords:
            return self.answer_cache[query][0]

        if context_keywords is None:
            context_keywords = []

        combined_context = self.retrieve_documents(query, context_keywords)
        
        # Debug: Check the retrieved context
        print("Retrieved context:", combined_context)
        
        if not combined_context:
            return "I couldn't find relevant information in the documents."

        max_context_length = max_length - len(query) - 50
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length].rsplit(' ', 1)[0]

        # Add special tokens [SEP] for BERT
        input_text = f"{query} [SEP] {combined_context}"
        inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="pt")
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Debug: Check the input ids and attention mask
        print("Input IDs:", input_ids)
        print("Attention Mask:", attention_mask)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Debug: Check the answer start and end indices
        print("Answer Start:", answer_start)
        print("Answer End:", answer_end)

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

        # Post-processing to remove repeated sentences
        answer = self.remove_repetitive_sentences(answer)

        self.answer_cache[query] = (answer, context_keywords)
        return answer

    def complete_last_sentence(self, text):
        if '.' in text:
            completed_text = text.rsplit('.', 1)[0] + '.'
        else:
            completed_text = text
        return completed_text

    def remove_repetitive_sentences(self, text):
        sentences = text.split('. ')
        seen_sentences = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence)
        return '. '.join(unique_sentences)

## EVALUATION
class EvaluationMetrics:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    def compute_precision_recall_f1(self, retrieved_docs, relevant_docs, k):
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / k
        recall = true_positives / len(relevant_set)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1

    def compute_exact_match(self, generated_answer, ground_truth):
        return int(generated_answer.strip().lower() == ground_truth.strip().lower())

    def compute_f1_score(self, generated_answer, ground_truth):
        generated_tokens = self.tokenizer.tokenize(generated_answer)
        ground_truth_tokens = self.tokenizer.tokenize(ground_truth)
        common = set(generated_tokens) & set(ground_truth_tokens)
        if len(common) == 0:
            return 0
        precision = len(common) / len(generated_tokens)
        recall = len(common) / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

def load_ground_truth_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def evaluate_rag_system(rag_system, data, k=3):
    metrics = EvaluationMetrics()
    total_precision, total_recall, total_f1, total_em, total_f1_score = 0, 0, 0, 0, 0
    num_samples = len(data)

    for sample in data:
        query = sample['query']
        ground_truth_answer = sample['answer']
        relevant_docs = sample['relevant_docs']

        # Retrieve documents
        retrieved_docs = rag_system.retrieve_documents(query, context_keywords=None)
        retrieved_docs = [doc.strip() for doc in retrieved_docs.split('. ') if doc.strip()]
        relevant_docs = [doc.strip() for doc in relevant_docs if doc.strip()]

        # Compute retrieval metrics
        precision, recall, f1 = metrics.compute_precision_recall_f1(retrieved_docs, relevant_docs, k)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        # Generate answer
        generated_answer = rag_system.generate_answer(query)

        # Compute generation metrics
        em = metrics.compute_exact_match(generated_answer, ground_truth_answer)
        f1_score = metrics.compute_f1_score(generated_answer, ground_truth_answer)
        total_em += em
        total_f1_score += f1_score

    # Average the metrics over all samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples
    avg_em = total_em / num_samples
    avg_f1_score = total_f1_score / num_samples

    return avg_precision, avg_recall, avg_f1, avg_em, avg_f1_score

# Load ground truth data
data = load_ground_truth_data('ground_truth_data')

# Initialize the RAG system (assuming the class CustomRAGProcessor is already defined)
rag_system = CustomRAGProcessor(model_name='emilyalsentzer/Bio_ClinicalBERT', storage_directory='path/to/storage')

# Evaluate the RAG system
avg_precision, avg_recall, avg_f1, avg_em, avg_f1_score = evaluate_rag_system(rag_system, data)

print(f'Precision@{3}: {avg_precision:.4f}')
print(f'Recall@{3}: {avg_recall:.4f}')
print(f'F1 Score (Retrieval)@{3}: {avg_f1:.4f}')
print(f'Exact Match: {avg_em:.4f}')
print(f'F1 Score (Answer): {avg_f1_score:.4f}')
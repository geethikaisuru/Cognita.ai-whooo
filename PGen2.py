import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from gensim import corpora
from gensim.models import LdaModel
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import requests
import json

print("Successfully imported all libraries")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    print(f"Extracted text sample: {text[:200]}...")  # Print a sample of the extracted text
    return text

def preprocess_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove numbering and special characters
    text = re.sub(r'\d+\.|\(|\)', '', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        preprocessed_sentences.append(' '.join(tokens))
    
    return preprocessed_sentences

def analyze_questions(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    analyzed_questions = []
    for sent in doc.sents:
        if sent.text.strip().endswith('?'):
            entities = [ent.text for ent in sent.ents]
            pos_tags = [token.pos_ for token in sent]
            analyzed_questions.append({
                'text': sent.text,
                'entities': entities,
                'pos_tags': pos_tags
            })
    
    return analyzed_questions

def identify_topics(preprocessed_text):
    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary([text.split() for text in preprocessed_text])
    
    # Create a corpus
    corpus = [dictionary.doc2bow(text.split()) for text in preprocessed_text]
    
    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100)
    
    topics = lda_model.print_topics()
    return topics

def format_question_paper(questions):
    paper = "Model Paper Generated locally.\n ----------------------------------\n"
    for i, question in enumerate(questions, 1):
        paper += f"{i}. {question}\n\n"
    return paper

def save_as_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Split the text into lines
    lines = text.split('\n')
    
    y = height - 40  # Start near the top of the page
    for line in lines:
        # If we're near the bottom of the page, start a new page
        if y < 40:
            c.showPage()
            y = height - 40
        
        c.drawString(40, y, line)
        y -= 15  # Move down for the next line
    
    c.save()

def generate_questions(context, num_questions):
    generated_questions = []
    for i in range(num_questions):
        start_idx = i * 100 % len(context)
        prompt = f"Generate a question based on the following context: {context[start_idx:start_idx+200]}\n\nQuestion:"
        
        response = requests.post('http://localhost:11434/api/generate', 
                                 json={
                                     "model": "phi3:mini",
                                     "prompt": prompt,
                                     "stream": False
                                 })
        
        if response.status_code == 200:
            question = response.json()['response'].strip()
            generated_questions.append(question)
        else:
            print(f"Error generating question: {response.status_code}")
    
    print(f"Generated {len(generated_questions)} questions")
    return generated_questions


def filter_and_rank_questions(generated_questions, original_questions, topics):
    filtered_questions = []
    for question in generated_questions:
        processed_question = post_process_question(question)
        if len(processed_question.split()) >= 5 and processed_question not in filtered_questions:
            filtered_questions.append(processed_question)
    
    # Limit to 20 questions
    filtered_questions = filtered_questions[:20]
    
    print(f"Filtered to {len(filtered_questions)} questions")
    return filtered_questions

def post_process_question(question):
    # Remove any text before the actual question
    question = re.sub(r'^.*?([A-Z])', r'\1', question, flags=re.DOTALL)
    
    # Capitalize the first letter
    question = question.capitalize()
    
    # Ensure the question ends with a question mark
    if not question.endswith('?'):
        question += '?'
    
    return question

print("Completed functions compiling")

def main(pdf_files):
    print("Extracting text from PDFs...")
    text = extract_text_from_pdfs(pdf_files)
    print(f"Extracted {len(text)} characters of text")
    
    print("Preprocessing text...")
    preprocessed_text = preprocess_text(text)
    print(f"Preprocessed into {len(preprocessed_text)} sentences")
    
    print("Analyzing questions...")
    analyzed_questions = analyze_questions(text)
    print(f"Analyzed {len(analyzed_questions)} questions")
    
    print("Identifying topics...")
    topics = identify_topics(preprocessed_text)
    print(f"Identified {len(topics)} topics")
    
    print("Generating questions...")
    generated_questions = generate_questions(' '.join(preprocessed_text), num_questions=50)
    
    print("Filtering and ranking questions...")
    filtered_questions = filter_and_rank_questions(generated_questions, analyzed_questions, topics)
    
    print("Formatting question paper...")
    final_paper = format_question_paper(filtered_questions)
    
    print("Saving question paper as PDF...")
    save_as_pdf(final_paper, "model_paperLocalOllama2.pdf")
    
    return final_paper

pdf_files = ['./PGenInputs/Papers/DBTutorial1.pdf', './PGenInputs/Papers/DBTutorial2.pdf', './PGenInputs/Papers/DBTutorial3.pdf']
model_paper = main(pdf_files)
print("Model paper generation complete. Output saved as 'model_paper.pdf'.")
print("\nGenerated Model Paper:")
print(model_paper)
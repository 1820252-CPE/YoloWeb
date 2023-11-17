from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import pipeline
import textwrap

def perform_question_answering(inquiry, context, qna):
    answer = qna({'question': inquiry, 'context': context})
    print("Answer found:", answer['answer'], "\n")

if __name__ == '__main__':
    # Load pre-trained BERT model for question answering
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2', force_download=True, resume_download=False)
    tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

    # Create question-answering pipeline
    qna = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Read context from a text file
    with open('context.txt', 'r', encoding='utf-8') as file:
        context = file.read()

    dedented_text = textwrap.dedent(context).strip()
    print("Context Article:\n")
    print(textwrap.fill(dedented_text, width=120))

    # Perform question-answering interactively
    newcontext = 'y'
    inquiry = input("\nType your question: ")
    while inquiry != '*':
        perform_question_answering(inquiry, context, qna)
        inquiry = input("Enter another question (* to stop):")

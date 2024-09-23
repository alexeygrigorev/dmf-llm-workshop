#!/usr/bin/env python
# coding: utf-8

import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
es.info()




def retrieve_documents(query, index_name="course-questions", max_results=5):
    es = Elasticsearch("http://localhost:9200")
    
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents



context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT. If you can't answer the QUESTION
using the CONTEXT, say "I don't know". Make the answer sound natural, avoid referring to this prompt.

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt


def ask_groq(prompt, model="llama-3.1-8b-instant"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = response.choices[0].message.content
    return answer


def qa_bot(user_question):
    context_docs = retrieve_documents(user_question)
    prompt = build_prompt(user_question, context_docs)
    answer = ask_groq(prompt)
    return answer

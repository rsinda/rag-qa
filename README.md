# RAG-QA

## Problem Statement
A multi-country B2B retail platform that serves structured content — Terms of Service, FAQs, Promotional Banners, Product Announcements, and Static Pages — across multiple countries and languages. The current system relies on exact-match lookups (type + country + language), forcing customers to manually search through long documents or contact support when they have specific questions. To improve this system we add a natural-language Q&A layer on top of that structred content: given a question, a country, and a language, it retrieves the most relevant content chunks that belong exclusively to that country, synthesize a grounded answer using an LLM, and returns verifiable citations that point back to the exact source documents — guaranteeing that no customer ever receives an answer grounded in another country's rules, even if those rules are semantically very similar.


## Architecture Diagram

![Langraph](screenshots/03_langgraph_graph.png)


## Prerequisites
1. Python 3.11
2. Docker
3. API KEY for LLM
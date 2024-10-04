import argparse
from pprint import pprint

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from openai import OpenAI
from sklearn.metrics import classification_report
import os
from dotenv import load_dotenv, find_dotenv

from data.data_spec import get_data_spec
from get_train_test_splits import get_train_test_splits
from utils.validata_args import validate_args

model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

load_dotenv(find_dotenv())


def find_similar_with_different_labels(query_text, retriever):
    """
    Finds k most similar documents with different labels (among retrieved)
    """
    similar_docs = retriever.get_relevant_documents(query_text)[1:]
    similar_texts = [(relevant_doc.metadata['english_MT']) for relevant_doc in similar_docs]
    similar_labels = [relevant_doc.metadata['label'] for relevant_doc in similar_docs]

    # Get unique labels from retrieved documents
    unique_labels = set(similar_labels)
    k = len(unique_labels)
    seen_labels = set()
    result = []
    for i, (text, label) in enumerate(zip(similar_texts, similar_labels)):
        if label in unique_labels and label not in seen_labels:
            result.append((text, label))
            seen_labels.add(label)
            if len(result) == k:  # Stop once we have k unique pairs
                break

    return result


def get_chatgpt_response(client, text):
    responce = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": text}],
    )

    return responce.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--k", type=str, default=7)
    parser.add_argument("--embedding_method", type=str, default="intfloat/multilingual-e5-base")
    args = parser.parse_args()
    args = validate_args(args)

    data_spec = get_data_spec()

    data_splits = get_train_test_splits(args["task"], data_spec)
    subset_data = data_splits["train"]
    query_data = data_splits["test"]

    embeddings = HuggingFaceEmbeddings(
        model_name=args["embedding_method"],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    documents = [
        Document(
            page_content=d['Text'], metadata={"label": d[args["task"]], "english_MT": d["English MT"], "id": id}
        ) for id, d in subset_data.iterrows()
    ]

    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": args["k"]})

    relevant_docs_with_labels = []

    query_list = query_data["Text"].tolist()
    query_list_english = query_data["English MT"].tolist()

    for query in query_list:
        output = find_similar_with_different_labels(query, retriever)
        relevant_docs_with_labels.append(output)

    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    definitions_list = [
        ('unbiased',
         'This label is applied when the text presents information neutrally without favoring any particular side.'),
        ('biased against palestine',
         'This label is used when referring to the perception that media outlets demonstrate a tendency to favor one '
         'side or present Palestinians in a negative light, often through selective reporting or framing'),
        ('biased against israel',
         'Bias against Israel in media refers to the perception that media outlets demonstrate a tendency to favor '
         'one side or present Israel in a negative light, often through selective reporting or framing'),
        ('unclear',
         'This label is applied when the text does not clearly indicate its stance or exhibits ambiguity in its '
         'presentation, leaving the reader uncertain about the intended message.'),
        ('not applicable',
         'This label is applied when the text does not relate to the bias annotations or the conflict between Israel '
         'and Palestine.'),
        ('biased against others',
         'Bias against others in media refers to the perception that media outlets demonstrate a tendency to favor '
         'certain groups or individuals while presenting others in a negative light'),
        ('biased against both palestine and israel',
         'Bias against both Palestine and Israel in media refers to the perception that media outlets display a '
         'propensity to favor one side or portray both sides in a negative light')
    ]

    # Create an empty list to store the responses
    responses = []
    prompts = []

    # Iterate over the list and generate the prompt for each string
    for query, examples in zip(query_list_english, relevant_docs_with_labels):

        # Create a list to store the example prompts
        example_prompts = []

        # Iterate over the examples for the current query
        for example in examples:
            example_text = example[0]  # Get the text of the example
            example_class = example[1]  # Get the class of the example

            # Find the definition for the current example class
            definition = next(
                (definition for definition in definitions_list if example_class.upper() in definition[0].upper()), None)

            # Create the prompt for each example with the definition
            example_prompt = f"{example_class}: {definition[1]}\nExample: {example_text}\n"

            example_prompts.append(example_prompt)

        # Combine all the example prompts into a single string
        examples_texts = "\n".join(example_prompts)

        # Create the complete prompt for the current query
        prompt = f'''
        Your task is to classify the following text, delimited with triple backticks, into one of the following categories with just the class name: {', '.join(set(example[1].upper() for example in examples))}.
        The text is a news article that has been published during the Israel-Palestine conflict. Your task is to classify the news to identify potential biases in media narratives.

        Here you can see th definition and examples of the classes:

        {examples_texts}

        Here is the news text to classify:
        ```{query}```

        just give back the class name from these classes: {', '.join(set(example[1].upper() for example in examples))}

        '''
        prompts.append(prompt)
        response = get_chatgpt_response(client, prompt)
        # Append the answer to the responses list
        responses.append(response)

    responses = [item.lower() for item in responses]
    responses = [string.replace('.', '') for string in responses]

    labels = query_data[args["task"]].tolist()
    labels = [item.lower() for item in labels]

    count = 0
    for i in range(len(responses)):
        if responses[i] == labels[i]:
            count += 1

    print(f"Number of correct predictions: {count / len(responses)}\n")

    print(
        f"The classification report for {args['task']} task with {args['embedding_method']} "
        f"embedding method:\n"
    )

    pprint(classification_report(labels, responses))


if __name__ == '__main__':
    main()

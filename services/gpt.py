api_key = "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"
import openai

openai.api_key = api_key


def get_ticker_from_filename(filename):
    system_msg = "You are a helpful assistant for a search engine."
    # Define the user message
    user_msg = (
        "Figure out what stock ticker is mentioned in this filename "
        + filename
        + ". Please just return the ticker. Example response: 'AAPL'"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    ai_message = response["choices"][0]["message"]["content"]
    return ai_message


def get_company_name_from_input(input):
    # Define the system message

    system_msg = "You are a helpful assistant for a search engine."
    # Define the user message
    user_msg = (
        "Figure out what companies are mentioned in this input "
        + input
        + ". Please return it as a array of strings. Just return the array."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    print("response: ", response)
    ai_message = response["choices"][0]["message"]["content"]

    return ai_message

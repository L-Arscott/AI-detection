import openai
import time
import pandas as pd

#from parse_xml import obtain_q_and_a_table_from_xml_file

def gpt_answer_from_prompt(prompt: str, max_tokens=1000):
    # Fetches chatGPT's answer to prompt as string
    with open("my_key.txt") as f:
        openai.api_key = f.readline()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    return response["choices"][0]["message"]["content"]


def gpt_answers_from_df_column_with_delay(question_column, delay=30):
    # Given the column of a dataframe, fetches chatGPT's answer to each entry
    # Delay option to avoid overloading server
    gpt_answers = []
    n_questions = len(question_column)

    for i, question in enumerate(question_column):
        gpt_answer = gpt_answer_from_prompt(question)
        gpt_answers.append(gpt_answer)

        print(f'{i+1}/{n_questions}')  # Progress

        time.sleep(delay)

    return gpt_answers

##
#obtain_q_and_a_table_from_xml_file('Posts.xml')  # Start from scratch
q_a_df_with_gpt = pd.read_csv('SE_with_GPT_0_to_100.csv')  # Continue

##
# Used to gradually fill our table, progress not lost if error occurs
for index, row in q_a_df_with_gpt.iterrows():
    if not type(row['GPT answer']) == str:
        q_a_df_with_gpt.at[index, 'GPT answer'] = gpt_answer_from_prompt(row['Title_body_combined'])

        print(f'did {index}')

        # Pause for 20 seconds before the next iteration
        time.sleep(20)

##
# Save our file
q_a_df_with_gpt.to_csv('SE_with_GPT_0_to_234.csv', index=False)

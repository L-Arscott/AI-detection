import xml.etree.ElementTree as ET
import pandas as pd
import re

def _remove_angle_bracket_content(string):
    return re.sub("\<.*?\>", "", string)

def obtain_q_and_a_table_from_xml_file(xml_file, n_rows=10000):
    # Reads in Stack Exchange posts in .xml format
    # Outputs a table containing id, question title, question body, answer
    # Table also has a title + question body combined column which will be fed to ChatGPT

    # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize empty lists to store data
    questions_data = []
    answers_data = []

    # Iterate through the first n_rows elements and extract data
    for i, row_element in enumerate(root.findall(".//row")):
        if i >= n_rows:
            break

        post_type_id = int(row_element.get("PostTypeId"))
        row_id = int(row_element.get("Id"))
        body = _remove_angle_bracket_content(row_element.get("Body"))

        if post_type_id == 1:
            accepted_answer_id = row_element.get("AcceptedAnswerId")
            title = row_element.get("Title")
            questions_data.append([row_id, title, body, accepted_answer_id])
        elif post_type_id == 2:
            answers_data.append([row_id, body])

    # Create DataFrames for questions and answers
    questions_df = pd.DataFrame(questions_data, columns=["ID", "Title", "Body", "AcceptedAnswerId"])
    answers_df = pd.DataFrame(answers_data, columns=["ID", "Body"])

    # Merge
    questions_df["AcceptedAnswerId"] = questions_df["AcceptedAnswerId"].astype(pd.Int64Dtype())
    questions_with_answers_df = pd.merge(
        questions_df,
        answers_df,
        how="inner",
        left_on="AcceptedAnswerId",
        right_on="ID",
        suffixes=("_question", "_answer")
    )

    # Keep only necessary colummns
    questions_with_answers_df = questions_with_answers_df[["ID_question", "Title", "Body_question", "Body_answer"]]

    # Rename columns
    questions_with_answers_df = questions_with_answers_df.rename(
        columns={"ID_question": "ID", "Body_question": "Body", "Body_answer": "Accepted answer"}
    )

    # Create title_body_combined column
    questions_with_answers_df['Title_body_combined'] = questions_with_answers_df.apply(
        lambda row: row['Title'] + '\n' + row['Body'], axis=1)

    return questions_with_answers_df

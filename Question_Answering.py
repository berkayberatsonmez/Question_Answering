import torch

from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example through the model.
    outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]),
                    # The segment IDs to differentiate question from answer_text
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)

bert_abstract = "If we go straight after leaving the living room, we will go to the kitchen. After exiting the hall, if we go straight to the left, the first room on our right is the bathroom. After exiting the living room, if we go straight to the left, the first room on our left is the bedroom. If we go straight after leaving the kitchen, we will go to the Hall. After exiting the kitchen, if we go straight to the right, the first room to our right is the Bathroom. After exiting the kitchen, if we go straight to the right, the first room on our left is the Bedroom. If we go straight after exiting the bathroom, we'll go to the Bedroom. After exiting the bathroom, if we head straight to the left, the first room to our right is the Hall. After exiting the bathroom, if we go straight to the left, the first room on our left is the Kitchen. If we go straight after leaving the bedroom, we'll go to the bathroom. After exiting the bedroom, if we go straight to the right, the first room to our right is the living room. After exiting the bedroom, if we go straight to the right, the first room on our left is the kitchen."

#print(wrapper.fill(bert_abstract))

question = "How do I get from the Living room to the Bathroom?"

print(answer_question(question, bert_abstract))

question = "How do I get from the Kitchen to the Living room?"

print(answer_question(question, bert_abstract))

question = "How do I get from the Bathroom to the Bedroom?"

print(answer_question(question, bert_abstract))

question = "How do I get from the Bedroom to the Living Room?"

print(answer_question(question, bert_abstract))
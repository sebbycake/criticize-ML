from simpletransformers.t5 import T5Model

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "eval_batch_size": 16,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "use_multiprocessing": False,
    # "silent": True,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

model = T5Model("outputs/best_model", args=model_args)


query = "ask_question: " + """

Singapore has started planning for the possibility that Covid-19 may become endemic here, said Finance Minister Lawrence Wong.

This could mean Singaporeans will need to get booster jabs from time to time, he noted. In the coming months, better treatments could also be developed for the disease, making it less of something to fear, he added at a virtual press conference.

But even then, the country may have to take basic precautions - for example, with regard to ventilation systems and buildings - in order to minimise the risk of infection.

"When will it happen? I really can't say," Wong said, in response to a reporter's question on when the virus will be considered endemic here.

"But we are indeed planning for a plausible scenario down the road where scientists around the world... come to the conclusion that it's not going to be possible to eradicate this virus - it's never going to go away, and we then have to learn to live with it."

The minister was speaking at a press conference to announce extra help for individuals and businesses impacted by the tightened measures on social interaction. The S$800 million package of support measures will be debated at the next Parliament sitting in July.

At the virtual event, Wong was asked how Singaporeans might go about their daily lives in the coming years, given that it seems difficult to picture the current restrictions on mask wearing and social gatherings lasting for a long time.

"I can't even predict what's going to happen next month," he replied.

"So I don't know that it's so easy to tell you what's going to happen years down the road because the situation is really very uncertain."

He did, however, note that the current strict measures are working to help Singapore curb the spread of the virus.

"Therefore, we do not think there is a need for further tightening in our overall posture," Mr Wong said, adding that a fuller update will be given at the next press conference by the multi-ministerial task force tackling Covid-19 on Monday
"""

preds = model.predict([query])

print(preds)

import multiprocessing as mp
import json
import logging
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

with open(r"train.json", "r") as read_file:
  train = json.load(read_file)

with open(r"test.json", "r") as read_file:
    test = json.load(read_file)


def simple_func():
    model_type="bert"
    model_name= "bert-base-cased"
    if model_type == "bert":
        model_name = "bert-base-cased"

    elif model_type == "roberta":
        model_name = "roberta-base"

    elif model_type == "distilbert":
        model_name = "distilbert-base-cased"

    elif model_type == "distilroberta":
        model_type = "roberta"
        model_name = "distilroberta-base"

    elif model_type == "electra-base":
        model_type = "electra"
        model_name = "google/electra-base-discriminator"

    elif model_type == "electra-small":
        model_type = "electra"
        model_name = "google/electra-small-discriminator"

    elif model_type == "xlnet":
        model_name = "xlnet-base-cased"
        

    # Configure the model 
    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.n_best_size=3
    model_args.num_train_epochs=5

    ### Advanced Methodology
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_cached_eval_features": True,
        "output_dir": f"outputs/{model_type}",
        "best_model_dir": f"outputs/{model_type}/best_model",
        "evaluate_during_training": True,
        "max_seq_length": 128,
        "num_train_epochs": 10,
        "evaluate_during_training_steps": 1000,
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "n_best_size":3,
        # "use_early_stopping": True,
        # "early_stopping_metric": "mcc",
        # "n_gpu": 2,
        # "manual_seed": 4,
        # "use_multiprocessing": False,
        "train_batch_size": 128,
        "eval_batch_size": 64,
        # "config": {
        #     "output_hidden_states": True
        # }
    }

    model = QuestionAnsweringModel(model_type,model_name, args=train_args,use_cuda= False)

    ### Remove output folder
    #!rm -rf outputs

    # Train the model
    model.train_model(train, eval_data=test)

    # Evaluate the model
    result, texts = model.eval_model(test)

    # Make predictions with the model
    
    to_predict = [
        {
            "context": "Mutfaktan çıktıktan sonra dümdüz ilerlerisek salona gideriz. Mutfaktan çıktıktan sonra, sağa doğru dümdüz ilerlerisek, sağımızdaki ilk oda Banyodur. Mutfaktan çıktıktan sonra, sağa doğru dümdüz ilerlerisek, solumuzdaki ilk oda Yatak odasıdır.",
            "qas": [
                {
                    "question": "Mutfaktan salona nasıl giderim?",
                    "id": "0",
                }
            ]
        }
    ]
    answers, probabilities = model.predict(to_predict)
    print(answers)

if __name__ == '__main__':
    p = mp.Process(target=simple_func)
    p.start()
    print('Waiting for simple func to end')
    p.join()
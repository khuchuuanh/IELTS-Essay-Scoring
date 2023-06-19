import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer
from utils import timing
import torch

class IeltsONNXPredictor:
    def __init__(self, model_path_coherence, model_path_task,model_path_lexical, model_path_grammar, model_name = "bert-base-uncased"):
        self.ort_session_task = ort.InferenceSession(model_path_task)
        self.ort_session_coherence = ort.InferenceSession(model_path_coherence)
        self.ort_session_lexical = ort.InferenceSession(model_path_lexical)
        self.ort_session_grammar = ort.InferenceSession(model_path_grammar)
        self.label_task = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,8, 8.5, 9]
        self.label_coherence = [4, 5, 6, 7, 8, 9]
        self.label_lexical = [4, 5, 6, 7, 8, 9]
        self.label_grammar = [4, 5, 6,7 ,8, 9]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    @timing
    def predict(self, essay, input):
        encoder_essay = self.tokenizer(
            essay,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoder_task = self.tokenizer(
            input,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ort_inputs_task = {'input_ids':encoder_task['input_ids'], 'attention_mask':encoder_task['attention_mask']} 
        ort_inputs_task = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_task.items()}
        ort_outs_task = self.ort_session_task.run(None, ort_inputs_task)
        scores_task = softmax(ort_outs_task[0])[0]
        pred_task = self.label_task[np.argmax(scores_task)]

        ort_inputs_coh = {'input_ids':encoder_essay['input_ids'], 'attention_mask':encoder_essay['attention_mask']} 
        ort_inputs_coh = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_coh.items()}
        ort_outs_coh = self.ort_session_coherence.run(None, ort_inputs_coh)
        scores_coh = softmax(ort_outs_coh[0])[0]
        pred_coh = self.label_coherence[np.argmax(scores_coh)]

        ort_inputs_grammar = {'input_ids':encoder_essay['input_ids'], 'attention_mask':encoder_essay['attention_mask']} 
        ort_inputs_grammar = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_grammar.items()}
        ort_outs_grammar = self.ort_session_grammar.run(None, ort_inputs_grammar)
        scores_grammar = softmax(ort_outs_grammar[0])[0]
        pred_grammar = self.label_coherence[np.argmax(scores_grammar)]

        ort_inputs_lexical = {'input_ids':encoder_essay['input_ids'], 'attention_mask':encoder_essay['attention_mask']} 
        ort_inputs_lexical = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_lexical.items()}
        ort_outs_lexical = self.ort_session_lexical.run(None, ort_inputs_lexical)
        scores_lexical = softmax(ort_outs_lexical[0])[0]
        pred_lexical = self.label_coherence[np.argmax(scores_lexical)]

        return  pred_coh, pred_lexical,pred_grammar,  pred_task

if __name__ == "__main__":
    question = "travelling in group with a tour guide is the best way to travel. do you agree or disagree with this statement give reason for your answer and include any relevant examples from your own knowledge or experience."
    essay = "there is no doubt on these days that travel agencies becoming more popular. the question is, booking with a tour guide would be better in this essay, i am going to discuss this view and draw my one conclusion. in terms of the positive aspects of group traveling with guidance that the main reason given to support this claim, is exposure to a variety of people from different cultures. to illustrate, it would be a good chance to discover and learn new things. the other prominent reason is safety. in other words, the tour guide will know all the risky places. thus booking with travel agencies has advantages. however, there are some disadvantages of this thought. firstly, committing to specific activities. in other words, sometimes cannot change the schedule, so ending up going to places that are not interesting on. secondly, some tour guides take a big group with each other, thus may create noise and destructive vibes during the trip. hence, a tour guide is not the better way to discover the city. in conclusion, although, the positive side of taking a guided when going to a new city, it has a negative aspect. therefore i believe that the best way to discover new things and make the best trip is when making a deep search and reading a review of other people."
    input = 'CLS '+ question + ' SEP ' + essay + ' SEP'

    model_path_coherence = 'save_model/model_coherence.onnx'
    model_path_task = 'save_model/model_task.onnx'
    model_path_lexical = 'save_model/model_lexical.onnx'
    model_path_grammar = 'save_model/model_grammar.onnx'
    predictor = IeltsONNXPredictor(model_path_coherence, model_path_task, model_path_lexical, model_path_grammar)
    print(predictor.predict(essay, input))

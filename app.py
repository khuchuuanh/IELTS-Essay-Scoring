from fastapi import FastAPI
from pydantic import BaseModel
from inference_onnx import IeltsONNXPredictor

app = FastAPI(title="IELTS Scoring App")

model_path_coherence = 'save_model/model_coherence.onnx'
model_path_task = 'save_model/model_task.onnx'
model_path_lexical = 'save_model/model_lexical.onnx'
model_path_grammar = 'save_model/model_grammar.onnx'
predictor = IeltsONNXPredictor(model_path_coherence, model_path_task, model_path_lexical, model_path_grammar)

@app.get("/")

class TextData(BaseModel):
    text: str
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.post("/predict")
async def predict(question: TextData, essay: TextData):
    input = 'CLS '+ question.text + ' SEP ' + essay.text + ' SEP'
    pred_coh, pred_lexical,pred_grammar,  pred_task = predictor.predict(essay.text, input)
    return {"predicted_coherence": pred_coh, "predicted_lexical": pred_lexical, "predicted_grammar": pred_grammar, "predicted_task": pred_task}
from flask import Flask, request, jsonify
from transformers import BertTokenizer
# , BertForSequenceClassification
import torch

app = Flask(__name__)

# pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    #model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()
    
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)




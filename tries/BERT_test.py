from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载 FinBERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# 示例
text = "The company's earnings exceeded expectations."
sentiment = predict_sentiment(text)
print(f"Predicted sentiment: {sentiment}")

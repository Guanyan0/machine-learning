from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text = "使用 BERT 进行自然语言处理。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(encoded_input)
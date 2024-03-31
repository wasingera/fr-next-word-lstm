import torch

import config
from model import FrenchSentimentAnalysis
from vocab import Vocab

vocab = Vocab(max_words=config.max_words, min_freq=config.min_freq,
              special_tokens=config.special_tokens, default_token=config.unk_token)
vocab.build_vocab_from_file(config.train_file, config.vocab_file)

model = FrenchSentimentAnalysis(len(vocab), config.embedding_dim, config.hidden_dim).to(config.device)
model.load_state_dict(torch.load(config.model_file))

def predict_next_word(text, model, vocab):
    model.eval()
    with torch.no_grad():
        tokens = vocab.numericalize_text(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(config.device)
        output = model(tokens)
        prediction = int(torch.argmax(output, dim=1).item())
        predicted_word = vocab.idx2word[prediction]
        predicted_prob = torch.softmax(output, dim=1).squeeze()[prediction].item()
        return predicted_word, predicted_prob

text = "Il"
for i in range(0, 50):
    word, prob = predict_next_word(text, model, vocab)
    # print(f"Predicted next word: {word}")
    # print(f"Confidence: {100*prob:.2f}%")
    text += " " + word
print(text)

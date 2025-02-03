import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pyrai import Engine
from pyrai.config import Config
from pyrai.backends import PyTorchBackend
from pyrai.metrics import Accuracy, MetricCollection

class TextClassifier(nn.Module):
    """LSTM-based text classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths):
        # text: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))
        
        # Pack sequence
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM
        output, (hidden, _) = self.lstm(packed)
        
        # Concat final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # Classification
        return self.fc(hidden)

def yield_tokens(data_iter, tokenizer):
    """Yield tokens from dataset."""
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch, vocab, tokenizer, device):
    """Collate batch of examples."""
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(_label - 1)  # AG_NEWS labels are 1-based
        processed_text = torch.tensor(
            [vocab[token] for token in tokenizer(_text)],
            dtype=torch.long
        )
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    
    labels = torch.tensor(label_list, dtype=torch.long)
    padded_text = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    lengths = torch.tensor(lengths)
    
    return padded_text.to(device), lengths, labels.to(device)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load AG_NEWS dataset
    train_iter = AG_NEWS(split='train')
    test_iter = AG_NEWS(split='test')
    
    # Setup tokenizer and vocabulary
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer),
        min_freq=5,
        specials=['<unk>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    # Create data loaders
    train_iter = AG_NEWS(split='train')
    test_iter = AG_NEWS(split='test')
    
    # Create engine
    config = Config({
        "training": {
            "batch_size": 64,
            "epochs": 5,
            "learning_rate": 0.001,
            "optimizer": {
                "name": "adam",
                "weight_decay": 0.0001
            },
            "scheduler": {
                "name": "reduce_on_plateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 1
            }
        },
        "model": {
            "type": "pytorch",
            "device": str(device),
            "save_dir": "models/ag_news"
        }
    })
    
    engine = Engine(
        backend=PyTorchBackend(),
        config=config
    )
    
    # Create model and metrics
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=4
    )
    
    metrics = MetricCollection({
        "accuracy": Accuracy()
    })
    
    # Train model
    history = engine.train(
        model=model,
        train_data=train_iter,
        val_data=test_iter,
        metrics=metrics,
        collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer, device)
    )
    
    print("\nTraining history:")
    for metric, values in history.items():
        print(f"{metric}: {values[-1]:.4f}")
    
    # Deploy model
    deployment = engine.deploy(
        model=model,
        model_id="ag_news",
        version="v1"
    )
    
    # Test predictions
    test_iter = AG_NEWS(split='test')
    total_correct = 0
    total_samples = 0
    
    for label, text in test_iter:
        # Prepare input
        tokens = tokenizer(text)
        text_tensor = torch.tensor(
            [vocab[token] for token in tokens],
            dtype=torch.long
        ).unsqueeze(0).to(device)
        lengths = torch.tensor([len(tokens)])
        
        # Make prediction
        prediction = deployment.predict((text_tensor, lengths))
        predicted_label = prediction.argmax(dim=1).item()
        
        total_correct += (predicted_label == (label - 1))
        total_samples += 1
        
        if total_samples >= 1000:  # Test on subset
            break
    
    accuracy = total_correct / total_samples
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Stop deployment
    deployment.stop()

if __name__ == "__main__":
    main() 
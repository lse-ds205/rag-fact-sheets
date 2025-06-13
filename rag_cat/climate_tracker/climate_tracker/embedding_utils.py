from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class BAAIEmbedder:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval() # Set model to evaluation mode

    def encode_batch(self, sentences, batch_size=32):
        """Encodes a list of sentences in batches."""
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # Move inputs to the same device as the model if GPU is available
            if self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        if not all_embeddings:
            return [] # Should not happen if sentences is not empty, but good practice
        return torch.cat([torch.from_numpy(e) for e in all_embeddings], dim=0).numpy() # Concatenate batches

    def mean_pooling(self, token_embeddings, attention_mask):
        """Performs mean pooling on token embeddings using the attention mask."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) # Avoid division by zero
        return sum_embeddings / sum_mask 
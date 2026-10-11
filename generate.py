import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_tokens: int,
        context: TensorType[int],
        context_length: int,
        tokenizer,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        device = next(model.parameters()).device
        context = context.to(device)
        was_training = model.training
        model.eval()
        generated_token_ids: list[int] = []

        try:
            with torch.no_grad():
                for _ in range(new_tokens):
                    # 1. Crop context to max length the model can handle
                    if context.shape[1] > context_length:
                        context_cond = context[:, -context_length:]
                    else:
                        context_cond = context

                    # 2. Forward pass -> extract last token logits
                    logits = model(context_cond)             # (1, T, vocab_size)
                    last_logits = logits[:, -1, :]           # (1, vocab_size)

                    # 3. Apply Temperature Scaling
                    if temperature > 0:
                        last_logits = last_logits / max(temperature, 1e-5)

                    # 4. Apply Top-K Filtering
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                        last_logits[last_logits < v[:, [-1]]] = -float("Inf")

                    # 5. Convert to Probabilities and Sample
                    probs = F.softmax(last_logits, dim=-1)
                    if temperature == 0:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                    next_token_id = next_token.item()

                    # 6. Append token to context tracking list
                    context = torch.cat((context, next_token), dim=-1)
                    generated_token_ids.append(next_token_id)
        finally:
            model.train(was_training)

        return tokenizer.decode(generated_token_ids)
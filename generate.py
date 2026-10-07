import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for _ in range(new_chars):
            # Crop context to max length the model can handle
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            # Forward pass -> logits for every position
            logits = model(context)                          # (1, T, vocab_size)
            last_logits = logits[:, -1, :]                   # (1, vocab_size)
            probs = nn.functional.softmax(last_logits, dim=-1)

            # Sample next token and reset RNG for reproducibility
            next_token = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)

            # Append token to context and decode
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        return ''.join(result)
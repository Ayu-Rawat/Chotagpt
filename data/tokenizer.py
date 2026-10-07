from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            # Count adjacent pair frequencies
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # Find most frequent pair (tiebreak: lexicographically smallest)
            best_count = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            best = candidates[0]

            merges.append([best[0], best[1]])

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges
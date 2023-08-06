import torch
r = list(range(6))
permutations = [r[i:] + r[:i] for i in range(6)]
print(permutations)

permutation = torch.tensor(permutations[1], dtype=torch.long)
inv_permutation = torch.sort(permutation)[1]
print(inv_permutation)

print(list(reversed(range(5))))

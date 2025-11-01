# Question2
## Suppose that you are given the following facts:
1. ∀x∀y∀z gt(x, y) ∧ gt(y, z) → gt(x, z)
2. ∀a∀b succ(a, b) → gt(a, b)
3. ∀x gt(x, x)

## To prove gt(5,2) with resolution, there is an attempt shown as below.
- ∀x gt(x,x)

| Step | Clause                          | Resolution Source       | Substitution |
|------|---------------------------------|-------------------------|--------------|
| 1    | $\neg \text{gt}(5, 2)$          | Initial clause          | -            |
| 2    | $\neg \text{gt}(x, y) \lor \neg \text{gt}(y, z) \lor \text{gt}(x, z)$ | Initial clause (transitivity axiom) | - |
| 3    | $\neg \text{gt}(5, y) \lor \neg \text{gt}(y, 2)$ | Resolve Step 1 and Step 2 | $x/5, z/2$   |
| 4    | $\neg \text{succ}(a, b) \lor \text{gt}(a, b)$ | Initial clause (successor-greater-than axiom) | - |
| 5    | $\neg \text{gt}(5, y) \lor \neg \text{succ}(y, 2)$ | Resolve Step 3 and Step 4 | $y/a, 2/b$   |
| 6    | $\neg \text{gt}(x, y) \lor \neg \text{gt}(y, z) \lor \text{gt}(x, z)$ | Initial clause (transitivity axiom, reused) | - |
| 7    | $\neg \text{gt}(5, y) \lor \neg \text{gt}(y, y) \lor \neg \text{succ}(y, 2)$ | Resolve Step 5 and Step 6 | $x/5, z/y$   |
| 8    | $\square$ (Empty Clause, denoted as $\text{T}$) | Resolve within Step 7 (since $\neg \text{gt}(y, y)$ is always false for natural numbers) | Self-resolution |

## 2.1 What kind of errors exist(s) in the above proof?
## 2.2 To avoid such errors, do you need additional procedures? What kind of procedures? Can you re-write the correct proof?
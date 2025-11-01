# Solution to Question 2

## 2.1 Errors in the Proof

There are several critical errors in the provided proof attempt:

### Error 1: Inconsistency with Axiom 3
- **Axiom 3 states**: ∀x gt(x, x) (meaning everything is greater than itself)
- **Step 8 claims**: ¬gt(y, y) is "always false for natural numbers"
- **Problem**: These statements contradict each other. If Axiom 3 is true, then gt(y, y) is TRUE, making ¬gt(y, y) FALSE, not the clause itself false.
- **Root Cause**: Axiom 3 itself is likely incorrect. The "greater than" relation should be **irreflexive**, meaning ∀x ¬gt(x, x), not ∀x gt(x, x).

### Error 2: Invalid "Self-Resolution" in Step 8
- Resolution requires two clauses with complementary literals to produce a resolvent
- Step 7 contains ¬gt(y, y), but there's no complementary literal gt(y, y) to resolve with
- Simply having a false literal in a clause doesn't lead to the empty clause
- The concept of "self-resolution" as used here is not a valid inference rule in resolution theorem proving

### Error 3: Missing Ground Facts
- The proof attempts to prove gt(5, 2) but provides no facts about the specific numbers 2, 3, 4, 5
- We need concrete successor facts like succ(3, 2), succ(4, 3), succ(5, 4) to build the chain of reasoning
- Without these facts, we cannot establish the relationship between 5 and 2

### Error 4: Questionable Resolution in Step 7
- The resolution step combining Step 5 and Step 6 is not clearly justified
- The substitution appears to unify different variables to the same name, which can lead to incorrect results
- Proper variable renaming and unification should be applied

## 2.2 Required Procedures and Correct Proof

### Additional Procedures Needed:

1. **Proper Unification**: Ensure variables are renamed to avoid conflicts before resolution
2. **Factoring**: Apply factoring when a clause can be unified with itself
3. **Ground Facts**: Add concrete facts about the successor relation for numbers
4. **Axiom Correction**: Fix the reflexivity axiom

### Corrected Proof:

**Corrected Axioms:**
1. ∀x∀y∀z (gt(x, y) ∧ gt(y, z) → gt(x, z)) - Transitivity
2. ∀a∀b (succ(a, b) → gt(a, b)) - Successor implies greater-than
3. ∀x ¬gt(x, x) - Irreflexivity (CORRECTED)

**Additional Ground Facts:**
4. succ(3, 2)
5. succ(4, 3)
6. succ(5, 4)

**CNF Clauses:**
- C1: ¬gt(x, y) ∨ ¬gt(y, z) ∨ gt(x, z) (from Axiom 1)
- C2: ¬succ(a, b) ∨ gt(a, b) (from Axiom 2)
- C3: ¬gt(x, x) (from Axiom 3)
- C4: succ(3, 2) (ground fact)
- C5: succ(4, 3) (ground fact)
- C6: succ(5, 4) (ground fact)

**Resolution Proof:**

| Step | Clause | Resolution Source | Substitution |
|------|--------|-------------------|--------------|
| 1 | ¬gt(5, 2) | Negation of goal | - |
| 2 | ¬gt(x, y) ∨ ¬gt(y, z) ∨ gt(x, z) | C1 | - |
| 3 | ¬succ(a, b) ∨ gt(a, b) | C2 | - |
| 4 | gt(3, 2) | Resolve C4 and C2 | a/3, b/2 |
| 5 | gt(4, 3) | Resolve C5 and C2 | a/4, b/3 |
| 6 | gt(5, 4) | Resolve C6 and C2 | a/5, b/4 |
| 7 | ¬gt(4, y) ∨ gt(5, y) | Resolve Step 6 and C1 | x/5, y/4, z/y |
| 8 | gt(5, 3) | Resolve Step 7 and Step 5 | y/3 |
| 9 | ¬gt(3, z) ∨ gt(5, z) | Resolve Step 8 and C1 | x/5, y/3, z/z |
| 10 | gt(5, 2) | Resolve Step 9 and Step 4 | z/2 |
| 11 | □ (Empty Clause) | Resolve Step 10 and Step 1 | - |

**Alternative Shorter Proof Using Transitivity Chain:**

| Step | Clause | Resolution Source | Substitution |
|------|--------|-------------------|--------------|
| 1 | ¬gt(5, 2) | Negation of goal | - |
| 2 | gt(3, 2) | From succ(3,2) and C2 | a/3, b/2 |
| 3 | gt(5, 4) | From succ(5,4) and C2 | a/5, b/4 |
| 4 | ¬gt(5, y) ∨ ¬gt(y, 2) | Resolve Step 1 and C1 | x/5, z/2 |
| 5 | gt(4, 3) | From succ(4,3) and C2 | a/4, b/3 |
| 6 | gt(4, 2) | Resolve Step 5, Step 2, and C1 | x/4, y/3, z/2 |
| 7 | gt(5, 2) | Resolve Step 3, Step 6, and C1 | x/5, y/4, z/2 |
| 8 | □ (Empty Clause) | Resolve Step 7 and Step 1 | - |

### Why Not Directly Add succ(5, 2)?

**Question**: 为什么不直接添加 succ(5, 2) 这个基本事实，这样会更快？

**Answer**: 这样做是**语义错误**的，原因如下：

1. **succ(a, b) 的语义**: succ(a, b) 表示 "a 是 b 的**直接后继**（immediate successor）"
   - 例如：succ(3, 2) 表示 3 = 2 + 1
   - succ(5, 2) 则表示 5 = 2 + 1，这显然是**错误**的

2. **违反后继关系的定义**:
   - 在自然数中，每个数只有一个直接后继
   - 2 的直接后继是 3，不是 5
   - 如果添加 succ(5, 2)，就破坏了后继关系的数学意义

3. **正确的建模方式**:
   - 应该添加**后继链**: succ(3,2), succ(4,3), succ(5,4)
   - 然后通过**传递性公理** (Axiom 1) 推导出 gt(5, 2)
   - 这才是正确的逻辑推理过程

4. **类比说明**:
   - 如果要证明"祖父-孙子"关系
   - 不能直接说 A 是 C 的父亲（这是错误的）
   - 而应该说：A 是 B 的父亲，B 是 C 的父亲，通过传递性得出 A 是 C 的祖父

**总结**: 虽然直接添加 succ(5,2) 可以让证明更快，但这会导致**知识库在语义上不一致**，违背了 successor 关系的数学定义。正确的做法是保持基本事实的正确性，通过逻辑推理得出结论。

### Summary of Required Procedures:
1. **Standardize variables apart**: Rename variables in clauses to avoid conflicts
2. **Apply correct unification**: Match literals properly with most general unifier (MGU)
3. **Add necessary ground facts**: Include domain-specific facts about the problem
4. **Correct axioms**: Ensure axioms accurately reflect the intended relations
5. **Apply resolution systematically**: Use only valid resolution and factoring steps
6. **Maintain semantic consistency**: Ensure all ground facts are semantically correct and consistent with domain knowledge

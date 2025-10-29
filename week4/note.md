## Hint Points
- entailment
- resolution
- backward chaining


### Logic
- Syntax
- Semantic
#### Entailment
Entailment means that a sentence follows from another: \(K B \vDash\) α . Knowledge base \(KB\) entails the sentence α iff α is true in all worlds where \(K B\) is true.
#### Model


#### Inference Methods
- Enumeration | 枚举
For each model, check if what is true in \(K B\) has to be true in α .
    - Space complexity: \(O(n)\) for n symbols.
    - Time complexity: \(O(2^{n})\)
- Forward and Backward Chaining | 正反向链接
    - Data Driver VS Goal Driven
- Resolution | 归结

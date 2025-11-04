# Question 3 Solution: Variable Elimination in Bayesian Networks

## Problem Setup

We need to compute P(X₃|X₄) for two different Bayesian network structures with different variable elimination orders.

### Network Structures

**Left Diagram:**
- Edges: X₁→X₃, X₁→X₂, X₃→X₂, X₂→X₄, X₂→X₅
- Joint distribution: P(X₁, X₂, X₃, X₄, X₅) = P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)

**Right Diagram:**
- Edges: Same as left + X₆→X₅
- Joint distribution: P(X₁, X₂, X₃, X₄, X₅, X₆) = P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₆) P(X₅|X₂, X₆)

### Computing P(X₃|X₄)

By Bayes' rule:
$P(X₃|X₄) = P(X₃, X₄) / P(X₄) = P(X₃, X₄) / Σ_{X₃} P(X₃, X₄)$

So we need to compute the unnormalized joint P(X₃, X₄) by eliminating all other variables.

---

## Case 1: Left Figure, Elimination Order X₅, X₁, X₂

**Initial expression:**
$P(X₃, X₄) = Σ_{X₅} Σ_{X₁} Σ_{X₂} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)$

**Initial factors:**
- f₀¹ = P(X₁)
- f₀² = P(X₃|X₁)
- f₀³ = P(X₂|X₁, X₃)
- f₀⁴ = P(X₄|X₂)
- f₀⁵ = P(X₅|X₂)

### Step 1: Eliminate X₅

**Factors involving X₅:** P(X₅|X₂)

**New factor:**
$f₁(X₂) = Σ_{X₅} P(X₅|X₂) = 1$

**Reasoning:** Summing over all values of X₅ given X₂ equals 1 (total probability).

**Remaining factors:** P(X₁), P(X₃|X₁), P(X₂|X₁, X₃), P(X₄|X₂)

### Step 2: Eliminate X₁

**Factors involving X₁:** P(X₁), P(X₃|X₁), P(X₂|X₁, X₃)

**New factor:**
$
f₂(X₂, X₃) = Σ_{X₁} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃)
            = Σ_{X₁} P(X₁, X₃) P(X₂|X₁, X₃)
$

**Reasoning:** This marginalizes out X₁, creating a new factor over X₂ and X₃. This factor has size O(2²) = 4 entries for Boolean variables.

**Remaining expression:**
$
P(X₃, X₄) = Σ_{X₂} f₂(X₂, X₃) P(X₄|X₂)
$

### Step 3: Eliminate X₂

**Factors involving X₂:** f₂(X₂, X₃), P(X₄|X₂)

**New factor:**
$
f₃(X₃, X₄) = Σ_{X₂} f₂(X₂, X₃) P(X₄|X₂)
            = P(X₃, X₄)
$

**Reasoning:** This gives us the final unnormalized joint distribution over X₃ and X₄.

**Final computation:**
$
P(X₃|X₄) = f₃(X₃, X₄) / Σ_{X₃} f₃(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2²) = 4 (from f₂ and f₃)

---

## Case 2: Left Figure, Elimination Order X₂, X₁, X₅

**Initial expression:**
$
P(X₃, X₄) = Σ_{X₂} Σ_{X₁} Σ_{X₅} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)
$

### Step 1: Eliminate X₂

**Factors involving X₂:** P(X₂|X₁, X₃), P(X₄|X₂), P(X₅|X₂)

**New factor:**
$
f₁(X₁, X₃, X₄, X₅) = Σ_{X₂} P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)
$

**Reasoning:** Eliminating X₂ first creates a large factor over 4 variables. This is computationally expensive: O(2⁴) = 16 entries. Eliminating X₂ early is inefficient because it connects multiple variables.

**Remaining expression:**
$
P(X₃, X₄) = Σ_{X₁} Σ_{X₅} P(X₁) P(X₃|X₁) f₁(X₁, X₃, X₄, X₅)
$

### Step 2: Eliminate X₁

**Factors involving X₁:** P(X₁), P(X₃|X₁), f₁(X₁, X₃, X₄, X₅)

**New factor:**
$
f₂(X₃, X₄, X₅) = Σ_{X₁} P(X₁) P(X₃|X₁) f₁(X₁, X₃, X₄, X₅)
$

**Reasoning:** This reduces the factor from 4 variables to 3 variables. Size: O(2³) = 8 entries.

**Remaining expression:**
$
P(X₃, X₄) = Σ_{X₅} f₂(X₃, X₄, X₅)
$

### Step 3: Eliminate X₅

**Factors involving X₅:** f₂(X₃, X₄, X₅)

**New factor:**
$
f₃(X₃, X₄) = Σ_{X₅} f₂(X₃, X₄, X₅)
            = P(X₃, X₄)
$

**Final computation:**
$
P(X₃|X₄) = f₃(X₃, X₄) / Σ_{X₃} f₃(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2⁴) = 16 (from f₁). This is worse than Case 1, showing that elimination order matters!

---

## Case 3: Right Figure, Elimination Order X₅, X₁, X₂

**Initial expression:**
$
P(X₃, X₄) = Σ_{X₆} Σ_{X₅} Σ_{X₁} Σ_{X₂} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₆) P(X₅|X₂, X₆)
$

**Initial factors:**
- P(X₁)
- P(X₃|X₁)
- P(X₂|X₁, X₃)
- P(X₄|X₂)
- P(X₆)
- P(X₅|X₂, X₆)

**Key observation:** X₆ is now a root node (no parents) rather than a leaf node. It influences X₅ but is independent of all other variables.

### Step 1: Eliminate X₅

**Factors involving X₅:** P(X₅|X₂, X₆)

**New factor:**
$
f₁(X₂, X₆) = Σ_{X₅} P(X₅|X₂, X₆) = 1
$

**Reasoning:** Summing over all values of X₅ given X₂ and X₆ equals 1 (total probability). This creates a constant factor that doesn't affect the computation.

**Remaining factors:** P(X₁), P(X₃|X₁), P(X₂|X₁, X₃), P(X₄|X₂), P(X₆)

### Step 2: Eliminate X₁

**Factors involving X₁:** P(X₁), P(X₃|X₁), P(X₂|X₁, X₃)

**New factor:**
$
f₂(X₂, X₃) = Σ_{X₁} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃)
$

**Reasoning:** Same as Case 1. This marginalizes out X₁, creating a factor over X₂ and X₃.

**Remaining factors:** f₂(X₂, X₃), P(X₄|X₂), P(X₆)

**Size:** O(2²) = 4 entries

### Step 3: Eliminate X₂

**Factors involving X₂:** f₂(X₂, X₃), P(X₄|X₂)

**New factor:**
$
f₃(X₃, X₄) = Σ_{X₂} f₂(X₂, X₃) P(X₄|X₂)
$

**Remaining factors:** f₃(X₃, X₄), P(X₆)

**Size:** O(2²) = 4 entries

### Step 4 (Implicit): Eliminate X₆

**Factors involving X₆:** P(X₆)

**Final result:**
$
P(X₃, X₄) = f₃(X₃, X₄) · Σ_{X₆} P(X₆) = f₃(X₃, X₄) · 1 = f₃(X₃, X₄)
$

**Final computation:**
$
P(X₃|X₄) = f₃(X₃, X₄) / Σ_{X₃} f₃(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2²) = 4 (from f₂ and f₃)

**Note:** X₆ is a root node independent of X₃ and X₄. Since it doesn't appear in any factors involving the query variables after eliminating X₅, X₁, and X₂, it factors out as a constant and doesn't affect the conditional probability P(X₃|X₄).

---

## Case 4: Right Figure, Elimination Order X₂, X₁, X₅

**Initial expression:**
$
P(X₃, X₄) = Σ_{X₆} Σ_{X₂} Σ_{X₁} Σ_{X₅} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₆) P(X₅|X₂, X₆)
$

**Initial factors:**
- P(X₁)
- P(X₃|X₁)
- P(X₂|X₁, X₃)
- P(X₄|X₂)
- P(X₆)
- P(X₅|X₂, X₆)

### Step 1: Eliminate X₂

**Factors involving X₂:** P(X₂|X₁, X₃), P(X₄|X₂), P(X₅|X₂, X₆)

**New factor:**
$
f₁(X₁, X₃, X₄, X₅, X₆) = Σ_{X₂} P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂, X₆)
$

**Reasoning:** Eliminating X₂ first now creates a **5-variable factor** because X₆ is a parent of X₅. This is significantly worse than Case 2 where we only had a 4-variable factor!

**Size:** O(2⁵) = **32 entries** (much larger than Case 2!)

**Remaining factors:** P(X₁), P(X₃|X₁), f₁(X₁, X₃, X₄, X₅, X₆), P(X₆)

### Step 2: Eliminate X₁

**Factors involving X₁:** P(X₁), P(X₃|X₁), f₁(X₁, X₃, X₄, X₅, X₆)

**New factor:**
$
f₂(X₃, X₄, X₅, X₆) = Σ_{X₁} P(X₁) P(X₃|X₁) f₁(X₁, X₃, X₄, X₅, X₆)
$

**Reasoning:** This reduces the factor from 5 variables to 4 variables.

**Size:** O(2⁴) = 16 entries

**Remaining factors:** f₂(X₃, X₄, X₅, X₆), P(X₆)

### Step 3: Eliminate X₅

**Factors involving X₅:** f₂(X₃, X₄, X₅, X₆)

**New factor:**
$
f₃(X₃, X₄, X₆) = Σ_{X₅} f₂(X₃, X₄, X₅, X₆)
$

**Size:** O(2³) = 8 entries

**Remaining factors:** f₃(X₃, X₄, X₆), P(X₆)

### Step 4 (Implicit): Eliminate X₆

**Final result:**
$
P(X₃, X₄) = Σ_{X₆} f₃(X₃, X₄, X₆) P(X₆)
$

**Final computation:**
$
P(X₃|X₄) = P(X₃, X₄) / Σ_{X₃} P(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2⁵) = **32 (from f₁)**

**Key observation:** The addition of X₆ as a parent of X₅ (rather than a child) makes this elimination order even worse! The maximum factor size increased from 16 (Case 2) to 32 (Case 4).

---

## Key Insights

1. **Elimination order matters significantly:**
   - **Left figure (Cases 1 & 2):**
     - Order (X₅, X₁, X₂): max factor size = 4
     - Order (X₂, X₁, X₅): max factor size = 16
   - **Right figure (Cases 3 & 4):**
     - Order (X₅, X₁, X₂): max factor size = 4
     - Order (X₂, X₁, X₅): max factor size = **32**
   - The (X₅, X₁, X₂) order is much more efficient!

2. **Why does order matter?**
   - Eliminating X₂ early is bad because it's a "hub" node connecting X₁, X₃, X₄, and X₅
   - When X₂ is eliminated first, it creates large intermediate factors
   - Eliminating variables with fewer connections first generally reduces intermediate factor sizes

3. **Impact of network structure:**
   - **Left figure:** X₅ is a leaf node with only X₂ as parent
   - **Right figure:** X₅ has two parents (X₂ and X₆), making it more connected
   - In Case 4, eliminating X₂ creates a 5-variable factor (vs. 4-variable in Case 2) because P(X₅|X₂, X₆) involves both X₂ and X₆
   - This shows that adding edges can significantly worsen bad elimination orders

4. **Root nodes vs. leaf nodes:**
   - **Root nodes** (no parents, like X₆ in right figure): Can be left until the end since they're independent and factor out as constants
   - **Leaf nodes** (no children): Can be efficiently eliminated early
   - In Case 3, X₆ doesn't impact complexity because it's independent of the query variables after X₅ is eliminated

5. **Optimal strategy:**
   - Eliminate variables in an order that minimizes the size of intermediate factors
   - Avoid eliminating "hub" nodes early
   - Consider the dependency structure: variables with many parents/children create larger factors when eliminated
   - For this specific problem, eliminating in the order X₅ → X₁ → X₂ is optimal

6. **Comparison summary:**
   - **Best case:** Case 1 and Case 3 with max factor size of 4
   - **Moderate case:** Case 2 with max factor size of 16
   - **Worst case:** Case 4 with max factor size of 32
   - Case 4 demonstrates how poor elimination order combined with additional dependencies can dramatically increase computational cost

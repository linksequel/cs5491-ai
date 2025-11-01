# Question 3 Solution: Variable Elimination in Bayesian Networks

## Problem Setup

We need to compute P(X₃|X₄) for two different Bayesian network structures with different variable elimination orders.

### Network Structures

**Left Diagram:**
- Edges: X₁→X₃, X₁→X₂, X₃→X₂, X₂→X₄, X₂→X₅
- Joint distribution: P(X₁, X₂, X₃, X₄, X₅) = P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)

**Right Diagram:**
- Edges: Same as left + X₅→X₆
- Joint distribution: P(X₁, X₂, X₃, X₄, X₅, X₆) = P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂) P(X₆|X₅)

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
P(X₃, X₄) = Σ_{X₆} Σ_{X₅} Σ_{X₁} Σ_{X₂} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂) P(X₆|X₅)
$

### Step 0: Eliminate X₆ (preliminary)

**Reasoning:** X₆ is a leaf node (no children) and only appears in one factor P(X₆|X₅). It's efficient to eliminate it first.

**New factor:**
$
Σ_{X₆} P(X₆|X₅) = 1
$

After this, we have the same factors as the left diagram, so the process is identical to Case 1.

### Step 1: Eliminate X₅

**Factors involving X₅:** P(X₅|X₂)

**New factor:**
$
f₁(X₂) = Σ_{X₅} P(X₅|X₂) = 1
$

### Step 2: Eliminate X₁

**New factor:**
$
f₂(X₂, X₃) = Σ_{X₁} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃)
$

### Step 3: Eliminate X₂

**New factor:**
$
f₃(X₃, X₄) = Σ_{X₂} f₂(X₂, X₃) P(X₄|X₂) = P(X₃, X₄)
$

**Final computation:**
$
P(X₃|X₄) = f₃(X₃, X₄) / Σ_{X₃} f₃(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2²) = 4

**Note:** The addition of X₆ doesn't affect the computation since it's a leaf node that can be eliminated immediately.

---

## Case 4: Right Figure, Elimination Order X₂, X₁, X₅

**Initial expression:**
$
P(X₃, X₄) = Σ_{X₆} Σ_{X₂} Σ_{X₁} Σ_{X₅} P(X₁) P(X₃|X₁) P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂) P(X₆|X₅)
$

### Step 0: Eliminate X₆ (preliminary)

**New factor:**
$
Σ_{X₆} P(X₆|X₅) = 1
$

After this, we proceed with the given elimination order (same as Case 2).

### Step 1: Eliminate X₂

**Factors involving X₂:** P(X₂|X₁, X₃), P(X₄|X₂), P(X₅|X₂)

**New factor:**
$
f₁(X₁, X₃, X₄, X₅) = Σ_{X₂} P(X₂|X₁, X₃) P(X₄|X₂) P(X₅|X₂)
$

**Size:** O(2⁴) = 16 entries

### Step 2: Eliminate X₁

**New factor:**
$
f₂(X₃, X₄, X₅) = Σ_{X₁} P(X₁) P(X₃|X₁) f₁(X₁, X₃, X₄, X₅)
$

**Size:** O(2³) = 8 entries

### Step 3: Eliminate X₅

**New factor:**
$
f₃(X₃, X₄) = Σ_{X₅} f₂(X₃, X₄, X₅) = P(X₃, X₄)
$

**Final computation:**
$
P(X₃|X₄) = f₃(X₃, X₄) / Σ_{X₃} f₃(X₃, X₄)
$

**Complexity:** Maximum factor size = O(2⁴) = 16 (from f₁)

---

## Key Insights

1. **Elimination order matters:**
   - Orders (X₅, X₁, X₂) produce max factor size of 4
   - Orders (X₂, X₁, X₅) produce max factor size of 16
   - The first order is more efficient!

2. **Why does order matter?**
   - Eliminating X₂ early is bad because it's a "hub" node connecting X₁, X₃, X₄, and X₅
   - Eliminating leaf nodes (X₅, X₆) first is efficient
   - Eliminating nodes with fewer connections first generally reduces intermediate factor sizes

3. **Leaf nodes:**
   - In the right diagram, X₆ is a leaf and can be eliminated immediately without increasing complexity
   - This is why Cases 3 and 4 have the same complexity as Cases 1 and 2 respectively

4. **Optimal strategy:**
   - Eliminate variables in an order that minimizes the size of intermediate factors
   - Leaf nodes and variables with fewer connections should typically be eliminated first

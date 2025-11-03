# 📖 Theory Notes: Gradient Descent & Optimization

> **CS231n Lecture 3:** Loss Functions & Optimization  
> **Author:** Hamza  
> **Last Updated:** November 2024

---

## 🎯 Table of Contents

1. [The Optimization Problem](#1-the-optimization-problem)
2. [Gradient Descent: Mathematical Foundation](#2-gradient-descent-mathematical-foundation)
3. [The Three Variants: BGD, SGD, MBGD](#3-the-three-variants)
4. [Learning Rate Dynamics](#4-learning-rate-dynamics)
5. [Convergence Analysis](#5-convergence-analysis)
6. [Common Pitfalls & Solutions](#6-common-pitfalls--solutions)
7. [Connection to Modern Optimizers](#7-connection-to-modern-optimizers)

---

## 1. The Optimization Problem

### **Problem Formulation**

Given:
- Training dataset: `D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}`
- Model: `f(x; W)` parameterized by weights `W`
- Loss function: `L(W)` that measures prediction error

**Goal:** Find weights `W*` that minimize average loss

```
W* = argmin_W L(W)

Where: L(W) = (1/N) Σᵢ L(f(xᵢ; W), yᵢ)
```

### **Why is This Hard?**

1. **Non-convex loss surface:** Neural networks have many local minima
2. **High dimensionality:** Modern networks have millions of parameters
3. **Computational cost:** Computing exact gradient uses entire dataset

### **The Solution: Gradient Descent**

Instead of finding the global minimum directly, **iteratively move downhill**.

---

## 2. Gradient Descent: Mathematical Foundation

### **The Core Idea**

**Intuition:** If you're blindfolded on a mountain, walk downhill to reach the valley.

**Mathematical formulation:**

The gradient `∇L(W)` points in the direction of **steepest increase**. To **decrease** loss, move in the **opposite direction**.

### **Update Rule**

```
W_{t+1} = W_t - α · ∇L(W_t)

Where:
  W_t     = weights at step t
  α       = learning rate (step size)
  ∇L(W_t) = gradient of loss at current weights
```

### **Why Does This Work?**

**First-order Taylor approximation:**

```
L(W + Δ) ≈ L(W) + ∇L(W)ᵀ · Δ
```

If we choose `Δ = -α · ∇L(W)`:

```
L(W - α∇L) ≈ L(W) - α · ||∇L(W)||²
                        ↑
                   Always positive!
```

Therefore: `L(W - α∇L) < L(W)` for small enough `α`

**Conclusion:** Moving in the negative gradient direction **guarantees** loss decrease (locally).

---

### **Gradient Computation**

For a loss function `L(W)`, the gradient is a vector of partial derivatives:

```
∇L(W) = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
```

**Example: MSE Loss**

```
L(W) = (1/N) Σᵢ (xᵢᵀW - yᵢ)²

∇L(W) = (2/N) Σᵢ (xᵢᵀW - yᵢ) · xᵢ
      = (2/N) Xᵀ(XW - y)
```

**Chain rule in action:**
```
dL/dW = dL/dŷ · dŷ/dW
      = 2(ŷ - y) · x
```

---

## 3. The Three Variants

The key difference: **how many examples to use when computing gradient?**

### **A. Batch Gradient Descent (BGD)**

**Definition:** Use **all N training examples** to compute gradient

```python
for epoch in range(num_epochs):
    gradient = 0
    for i in range(N):  # ALL examples
        gradient += compute_gradient(x[i], y[i], W)
    gradient /= N
    W = W - learning_rate * gradient
```

**Mathematical form:**
```
∇L(W) = (1/N) Σᵢ₌₁ᴺ ∇Lᵢ(W)
```

**Pros:**
- ✅ Exact gradient (no estimation error)
- ✅ Smooth, deterministic convergence
- ✅ Guaranteed convergence for convex functions

**Cons:**
- ❌ Slow: Must process all N examples before one update
- ❌ Memory intensive: Need to load entire dataset
- ❌ Stuck in local minima (no exploration)

**When to use:** Small datasets (N < 10,000), convex problems

---

### **B. Stochastic Gradient Descent (SGD)**

**Definition:** Use **one random example** to compute gradient

```python
for epoch in range(num_epochs):
    shuffle(data)
    for i in range(N):  # ONE example at a time
        gradient = compute_gradient(x[i], y[i], W)
        W = W - learning_rate * gradient
```

**Mathematical form:**
```
∇L(W) ≈ ∇Lᵢ(W)  (gradient from random example i)
```

**Key property: Unbiased estimator**
```
E[∇Lᵢ] = (1/N) Σᵢ ∇Lᵢ = ∇L  (true gradient)
```

**Pros:**
- ✅ Fast updates: One example → one update
- ✅ Memory efficient
- ✅ Can escape local minima (noise helps exploration)
- ✅ Online learning: Can update as new data arrives

**Cons:**
- ❌ Noisy convergence (loss bounces around)
- ❌ Harder to parallelize
- ❌ May not converge to exact minimum

**When to use:** Large datasets (N > 100,000), online learning

---

### **C. Mini-Batch Gradient Descent (MBGD)**

**Definition:** Use **small batch** of examples (e.g., 32, 64, 128)

```python
batch_size = 64
for epoch in range(num_epochs):
    shuffle(data)
    for batch in get_batches(data, batch_size):
        gradient = (1/batch_size) * Σ compute_gradient(x, y, W)
        W = W - learning_rate * gradient
```

**Mathematical form:**
```
∇L(W) ≈ (1/B) Σⱼ₌₁ᴮ ∇Lⱼ(W)  (average over B examples)
```

**Variance reduction:**
```
Var(mini-batch gradient) = Var(SGD) / √B
```

**Pros:**
- ✅ **Best of both worlds**
- ✅ Less noisy than SGD, faster than BGD
- ✅ GPU-friendly (vectorized operations)
- ✅ Still explores (some noise remains)

**Cons:**
- ❌ Batch size is a hyperparameter to tune

**When to use:** **Always!** This is the industry standard.

**Typical batch sizes:**
- Small models: 32-64
- Standard: 128-256
- Large models/GPUs: 512-1024

---

### **Visual Comparison**

```
Loss Landscape:

BGD:   Smooth path
       ─────────────────→ min

SGD:   Noisy zigzag
       ─╱╲╱╲╱╲╱╲╱╲─→ ~ min

MBGD:  Moderately smooth
       ──╱╲─╱╲──→ min
```

---

## 4. Learning Rate Dynamics

### **The Most Critical Hyperparameter**

Learning rate `α` controls step size. **Getting this right is crucial.**

### **Problems with Fixed Learning Rate**

**Too large (`α = 1.0`):**
```
Step 0: W = [1.0, 2.0], Loss = 10.0
Step 1: W = [5.0, -3.0], Loss = 50.0  ← Overshot!
Step 2: W = [-10.0, 15.0], Loss = 200.0  ← Diverging!
Step 3: Loss = NaN  💥
```

**Too small (`α = 0.0001`):**
```
Step 0: Loss = 10.0
Step 100: Loss = 9.95  ← Barely moved
Step 1000: Loss = 9.5  ← Still far from minimum
```

**Just right (`α = 0.01`):**
```
Step 0: Loss = 10.0
Step 10: Loss = 5.0
Step 30: Loss = 1.0
Step 50: Loss = 0.1  ✓
```

---

### **Learning Rate Schedules**

**Problem:** Ideal learning rate changes during training!

- **Early:** Large steps explore landscape
- **Late:** Small steps fine-tune around minimum

**Solution:** **Learning rate decay**

#### **1. Step Decay**

```
lr(t) = lr₀ × γ^floor(t/drop_every)

Example: lr₀=0.1, γ=0.5, drop_every=10
  Epoch 0-9:   lr = 0.1
  Epoch 10-19: lr = 0.05
  Epoch 20-29: lr = 0.025
```

#### **2. Exponential Decay**

```
lr(t) = lr₀ × e^(-λt)

Smooth, continuous decay
```

#### **3. Polynomial Decay**

```
lr(t) = lr₀ × (1 - t/T)^p

Where T = total steps, p = power (usually 0.5 or 1)
```

#### **4. Cosine Annealing**

```
lr(t) = lr_min + 0.5(lr₀ - lr_min)(1 + cos(πt/T))

Smooth curve from lr₀ to lr_min
Used in modern transformers
```

---

### **Warmup**

**Problem:** Large LR at start can destabilize training

**Solution:** Gradually increase LR

```
if epoch < warmup_epochs:
    lr = lr_max × (epoch / warmup_epochs)
else:
    lr = lr_max × decay_schedule(epoch - warmup_epochs)
```

**Used in:** BERT, GPT, Vision Transformers

---

## 5. Convergence Analysis

### **When Does GD Converge?**

**Theorem (Convex Case):**

If `L(W)` is convex and ` smooth, GD with `α < 2/L` converges to global minimum.

**Rate:** `O(1/T)` after T steps

**Reality:** Neural networks are **non-convex**, so this doesn't apply!

---

### **Non-Convex Case**

GD converges to a **critical point** where `∇L = 0`:
- Could be local minimum
- Could be saddle point
- Could be global minimum (lucky!)

**Key insight:** SGD noise helps escape saddle points!

---

### **Convergence Criteria**

**How to know when to stop?**

#### **1. Gradient Norm**
```
if ||∇L(W)|| < ε:
    converged = True
```

#### **2. Loss Change**
```
if |L(W_t) - L(W_{t-1})| < ε:
    converged = True
```

#### **3. Parameter Change**
```
if ||W_t - W_{t-1}|| < ε:
    converged = True
```

#### **4. Validation Loss (Practical)**
```
if val_loss not improved for k epochs:
    early_stop = True
```

---

## 6. Common Pitfalls & Solutions

### **Problem 1: Exploding Gradients**

**Symptoms:**
```
Epoch 1: Loss = 2.5
Epoch 2: Loss = 5.0
Epoch 3: Loss = 50.0
Epoch 4: Loss = NaN  💥
```

**Causes:**
- Learning rate too large
- Unstable initialization
- Deep networks (gradient multiplication)

**Solutions:**
```python
# 1. Gradient clipping
if grad_norm > threshold:
    gradient *= threshold / grad_norm

# 2. Reduce learning rate
learning_rate /= 10

# 3. Better initialization (Xavier, He)
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

---

### **Problem 2: Vanishing Gradients**

**Symptoms:**
```
Epoch 1: Loss = 2.5
Epoch 100: Loss = 2.49
Epoch 1000: Loss = 2.48  ← No progress!
```

**Causes:**
- Learning rate too small
- Poor activation functions (sigmoid squashes)
- Deep networks

**Solutions:**
- Increase learning rate
- Use ReLU instead of sigmoid
- Batch normalization
- Residual connections (ResNets)

---

### **Problem 3: Oscillation**

**Symptoms:**
```
Loss: 2.5 → 1.5 → 2.0 → 1.3 → 1.8 → ...
      ↑ Bouncing around minimum
```

**Cause:** Learning rate too large for fine-tuning

**Solution:** Learning rate decay

---

### **Problem 4: Slow Convergence**

**Symptoms:**
```
Epoch 1: Loss = 2.500
Epoch 100: Loss = 2.499
          ↑ Tiny progress
```

**Causes:**
- Learning rate too small
- Poor conditioning (some directions have small gradients)

**Solutions:**
- Increase learning rate
- Use momentum
- Use adaptive optimizers (Adam)

---

## 7. Connection to Modern Optimizers

Vanilla GD is the foundation. Modern optimizers add:

### **Momentum**

**Problem:** GD oscillates in valleys

**Solution:** Add "velocity" term

```
v_{t+1} = β·v_t + ∇L(W_t)
W_{t+1} = W_t - α·v_{t+1}
```

**Effect:** Smooths updates, accelerates convergence

---

### **RMSprop**

**Problem:** One learning rate for all parameters

**Solution:** Adapt learning rate per parameter

```
E[g²]_t = β·E[g²]_{t-1} + (1-β)·g_t²
W_{t+1} = W_t - α/(√(E[g²]_t) + ε) · g_t
```

**Effect:** Large gradients → small steps; small gradients → large steps

---

### **Adam** (Most Popular)

**Combines momentum + RMSprop:**

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t        (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²       (adaptive LR)

m̂_t = m_t / (1 - β₁ᵗ)  (bias correction)
v̂_t = v_t / (1 - β₂ᵗ)

W_{t+1} = W_t - α · m̂_t / (√v̂_t + ε)
```

**Default hyperparameters:**
- `α = 0.001`
- `β₁ = 0.9`
- `β₂ = 0.999`

**Why Adam is popular:**
- ✅ Works well out-of-the-box
- ✅ Adaptive learning rates
- ✅ Momentum smoothing
- ✅ Used in 90% of papers

---

## 📊 Summary Table

| Method | Batch Size | Speed | Convergence | GPU-Friendly | Use Case |
|--------|-----------|-------|-------------|--------------|----------|
| BGD | N (all) | Slow | Smooth | No | Small data, convex |
| SGD | 1 | Fast | Noisy | No | Online learning |
| MBGD | 32-256 | Medium | Balanced | **Yes** | **Production** |
| +Momentum | 32-256 | Fast | Smooth | Yes | CNNs |
| +Adam | 32-256 | Fast | Smooth | Yes | **Default choice** |

---

## ✅ Key Takeaways

1. **Gradient descent is iterative hill descent**
   - Move in direction opposite to gradient
   - Guaranteed to decrease loss locally

2. **Three variants, one winner**
   - BGD: Exact but slow
   - SGD: Fast but noisy
   - **MBGD: Best balance** ← Industry standard

3. **Learning rate is critical**
   - Too large: diverges
   - Too small: slow
   - **Use decay schedules**

4. **Modern optimizers extend vanilla GD**
   - Momentum: Smooths updates
   - RMSprop: Adapts per parameter
   - **Adam: Combines both** ← Most popular

5. **Always monitor training**
   - Watch for NaN (exploding gradients)
   - Check convergence (early stopping)
   - Validate on held-out data

---

## 📚 Further Reading

1. **Foundational Papers:**
   - Robbins & Monro (1951): *A Stochastic Approximation Method*
   - Polyak (1964): *Some methods of speeding up convergence of iterative methods*
   - Kingma & Ba (2014): *Adam: A Method for Stochastic Optimization*

2. **Modern Surveys:**
   - Ruder (2016): *An overview of gradient descent optimization algorithms*
   - Smith (2017): *Don't Decay the Learning Rate, Increase the Batch Size*

3. **CS231n Resources:**
   - [Optimization Notes](https://cs231n.github.io/optimization-1/)
   - [Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)

---

**End of Theory Notes** 📚
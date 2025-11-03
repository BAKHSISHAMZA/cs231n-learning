# 📖 Theory Notes: k-Nearest Neighbors Classifier

> **CS231n Lecture 2:** Image Classification  
> **Author:** Hamza  
> **Last Updated:** November 2024

---

## 🎯 Table of Contents

1. [The Image Classification Problem](#the-image-classification-problem)
2. [k-Nearest Neighbors Algorithm](#k-nearest-neighbors-algorithm)
3. [Distance Metrics](#distance-metrics)
4. [Hyperparameter Selection](#hyperparameter-selection)
5. [Why kNN Fails for Images](#why-knn-fails-for-images)
6. [Mathematical Deep Dive](#mathematical-deep-dive)

---

## 1. The Image Classification Problem

### **Problem Formulation**

Given:
- **Input:** An image (represented as a grid of pixel values)
- **Output:** A label from a fixed set of categories

**Mathematically:**
```
f: ℝ^(H×W×C) → {1, 2, ..., K}

Where:
  H = image height
  W = image width
  C = number of channels (3 for RGB)
  K = number of classes
```

### **Semantic Gap**

The challenge: Computers see images as arrays of numbers, but we want them to understand **semantic content**.

**Example:**
```
Human sees: "A cat sitting on a couch"
Computer sees: [[[123, 45, 67], [124, 46, 68], ...], ...]
```

### **Key Challenges**

| Challenge | Description | Example |
|-----------|-------------|---------|
| **Viewpoint variation** | Same object from different angles | Front-facing cat vs side-view cat |
| **Illumination** | Lighting affects pixel values drastically | Cat in sunlight vs shadow |
| **Deformation** | Objects can change shape | Cat sitting vs cat stretching |
| **Occlusion** | Parts may be hidden | Cat behind a plant |
| **Background clutter** | Objects blend with surroundings | Camouflaged cat |
| **Intra-class variation** | Same class, different appearances | Persian cat vs Siamese cat |

---

## 2. k-Nearest Neighbors Algorithm

### **Core Intuition**

> "You are the average of your k nearest neighbors."

If most of your neighbors belong to class A, you probably belong to class A too.

### **Algorithm Steps**

#### **Training Phase**
```
Input: Training set (X_train, y_train)
Output: Stored data

Simply memorize all training examples.
Time: O(1)
```

#### **Prediction Phase**
```
Input: Test example x_test
Output: Predicted label y_pred

1. Compute distance from x_test to all training examples
2. Find k nearest neighbors
3. Take majority vote among their labels
4. Return most common label

Time: O(N × D)
  N = number of training examples
  D = dimensionality
```

### **Pseudocode**

```python
class kNN:
    def train(X_train, y_train):
        self.X = X_train
        self.y = y_train
    
    def predict(x_test, k):
        distances = compute_distance(x_test, self.X)
        k_indices = get_k_smallest(distances, k)
        k_labels = self.y[k_indices]
        return most_common(k_labels)
```

### **Time Complexity Analysis**

| Phase | Complexity | Why? |
|-------|-----------|------|
| **Training** | O(1) | Just store data in memory |
| **Prediction** | O(N × D) | Compare test example to all N training examples |

**Problem for production:**
- We want: Fast at test time (real-time prediction)
- kNN gives: Fast training, **slow testing** ❌

---

## 3. Distance Metrics

Distance metrics define "similarity" between data points.

### **L1 Distance (Manhattan Distance)**

**Formula:**
```
d_L1(I₁, I₂) = Σ_p |I₁[p] - I₂[p]|
```

**Intuition:** Sum of absolute differences across all pixels.

**Visual Analogy:**
```
Manhattan grid: You can only move along streets
Distance = total blocks traveled (horizontal + vertical)
```

**Example:**
```python
I₁ = [10, 20, 30, 40]
I₂ = [12, 18, 33, 38]

d_L1 = |10-12| + |20-18| + |30-33| + |40-38|
     = 2 + 2 + 3 + 2
     = 9
```

**Properties:**
- ✓ Fast to compute
- ✓ Coordinate-dependent (rotating image changes distance)
- ✓ Robust to outliers (doesn't square differences)

---

### **L2 Distance (Euclidean Distance)**

**Formula:**
```
d_L2(I₁, I₂) = √(Σ_p (I₁[p] - I₂[p])²)
```

**Intuition:** "As the crow flies" - straight-line distance.

**Example:**
```python
I₁ = [10, 20, 30, 40]
I₂ = [12, 18, 33, 38]

d_L2 = √((10-12)² + (20-18)² + (30-33)² + (40-38)²)
     = √(4 + 4 + 9 + 4)
     = √21
     ≈ 4.58
```

**Properties:**
- ✓ Natural geometric interpretation
- ✓ Rotation-invariant in the right coordinate system
- ✗ Sensitive to outliers (squares large differences)

---

### **Visual Comparison**

```
L1 distance:           L2 distance:
  ┌─────┐               ╱─────╲
  │     │              ╱       ╲
  │  ●──┼──●         ●           ●
  │     │              ╲       ╱
  └─────┘               ╲─────╱
  
 (Diamond shape)      (Circle shape)
```

Points at distance d=1:
- **L1:** Form a diamond (Manhattan ball)
- **L2:** Form a circle (Euclidean ball)

---

### **RGB Images**

For color images with 3 channels (R, G, B):

```python
# Image shape: (H, W, 3)
# Flatten to 1D: (H × W × 3,)

# L1 distance
d_L1 = Σ_x Σ_y (|R₁[x,y] - R₂[x,y]| + 
                 |G₁[x,y] - G₂[x,y]| + 
                 |B₁[x,y] - B₂[x,y]|)

# L2 distance
d_L2 = √(Σ_x Σ_y ((R₁[x,y] - R₂[x,y])² + 
                   (G₁[x,y] - G₂[x,y])² + 
                   (B₁[x,y] - B₂[x,y])²))
```

---

## 4. Hyperparameter Selection

### **What are Hyperparameters?**

**Hyperparameters:** Choices we make **before** training (not learned from data).

For kNN:
- `k`: Number of neighbors to consider
- Distance metric: L1, L2, cosine, etc.

### **The Golden Rule**

> **NEVER tune hyperparameters on test data!**

**Why?**
- Test set must simulate "never-before-seen" data
- If you optimize for test performance, you're **cheating**
- Real-world performance will be worse

---

### **Train/Validation/Test Split**

**Correct procedure:**

```
Full Dataset
    │
    ├── 80% Training Set
    │   └── Train model parameters
    │
    ├── 10% Validation Set
    │   └── Choose hyperparameters (k, distance metric)
    │
    └── 10% Test Set
        └── Final evaluation (run ONCE)
```

**Workflow:**

```python
# 1. Split data
X_train, X_val, X_test = split_data(X)

# 2. Try different hyperparameters on VALIDATION set
best_k = None
best_accuracy = 0

for k in [1, 3, 5, 10, 20]:
    model = kNN(k=k)
    model.train(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# 3. Train final model with best_k
final_model = kNN(k=best_k)
final_model.train(X_train, y_train)

# 4. Evaluate on TEST set (ONCE!)
test_accuracy = final_model.score(X_test, y_test)
print(f"Final test accuracy: {test_accuracy}")
```

---

### **Cross-Validation**

For small datasets, use **k-fold cross-validation**:

```
Fold 1: [Val | Train | Train | Train | Train]
Fold 2: [Train | Val | Train | Train | Train]
Fold 3: [Train | Train | Val | Train | Train]
Fold 4: [Train | Train | Train | Val | Train]
Fold 5: [Train | Train | Train | Train | Val]

Average validation performance across all folds
```

**Benefit:** Use all data for both training and validation.

---

## 5. Why kNN Fails for Images

### **Problem 1: Computational Cost**

- **Test time:** O(N × D) per image
  - CIFAR-10: N = 50,000, D = 32×32×3 = 3,072
  - **1 image = 150M operations** 😱
  
- **Unacceptable for production:** Real-time systems need <10ms per image

---

### **Problem 2: Perceptual Distance ≠ Pixel Distance**

**Example: Image shifts**

```
Original:        Shifted 1 pixel:
┌───┬───┬───┐    ┌───┬───┬───┐
│ 0 │100│200│    │100│200│ 0 │
├───┼───┼───┤ vs ├───┼───┼───┤
│ 0 │100│200│    │100│200│ 0 │
└───┴───┴───┘    └───┴───┴───┘

L2 distance: LARGE (almost all pixels changed!)
Perceptual similarity: HIGH (visually identical)
```

**Conclusion:** Raw pixel distance doesn't capture semantic similarity.

---

### **Problem 3: Curse of Dimensionality**

In high dimensions, all points become **equally distant**.

**Example:** Random points in d-dimensional hypercube

| Dimensions | Avg distance | Insight |
|------------|--------------|---------|
| 2D | Varies | Neighbors are meaningful |
| 10D | More uniform | Neighbors start losing meaning |
| 100D | Nearly identical | All points are "far" |
| 3072D (CIFAR) | **All points equidistant** | kNN breaks down |

**Mathematical intuition:**
```
In d dimensions, volume of hypersphere ~ r^d

To cover constant fraction of space, need:
N_samples ~ exp(d)  (exponential growth!)
```

---

## 6. Mathematical Deep Dive

### **Why Does kNN Work? (Statistical View)**

**Theorem (Cover & Hart, 1967):**

As N → ∞, 1-NN error rate ≤ 2 × Bayes error rate

**Intuition:**
- With infinite data, nearest neighbor captures local density
- Optimal decision boundary is recovered in the limit

**Reality:**
- We don't have infinite data
- Curse of dimensionality makes "neighbors" meaningless in high dims

---

### **Distance Metrics as Inner Products**

**L2 distance can be rewritten:**
```
‖x - y‖₂² = (x - y)ᵀ(x - y)
          = xᵀx - 2xᵀy + yᵀy
          = ‖x‖² - 2⟨x, y⟩ + ‖y‖²
```

**Key insight:** Distance depends on **inner product** ⟨x, y⟩

This connects to:
- Kernel methods (SVMs)
- Neural network similarity learning
- Modern metric learning

---

### **When kNN Actually Works**

**Good use cases:**
1. **Low-dimensional data** (d < 20)
2. **Large datasets** relative to dimensionality
3. **Meaningful distance metrics** (not raw pixels)

**Modern usage:**
```
Deep Learning Features + kNN
           ↓
1. Use CNN to extract features (e.g., ResNet)
   Image (224×224×3) → Embedding (512,)
2. Apply kNN on embeddings
3. Embeddings capture semantic similarity!
```

This is used in:
- Image retrieval systems
- Few-shot learning
- Anomaly detection

---

## 📚 Summary

### **Key Takeaways**

1. ✅ **kNN is simple:** Train = memorize, Predict = find neighbors
2. ✅ **Distance metric matters:** L1 vs L2 is a hyperparameter choice
3. ✅ **Validation set is essential:** Never tune on test data
4. ❌ **kNN fails for raw images:** Pixel distance ≠ perceptual distance
5. ❌ **Curse of dimensionality:** Need exponentially more data in high dims
6. ✅ **Modern use:** kNN on learned features (CNN embeddings) works great

### **What's Next?**

kNN taught us:
- Classification pipeline (train/val/test)
- Importance of distance metrics
- Need for better representations

**Next:** Linear classifiers - our first **parametric** model!
- Instead of memorizing data, **learn parameters**
- Fast at test time (O(D) instead of O(N×D))
- Foundation for neural networks

---

## 📖 References

1. Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification.* IEEE Transactions on Information Theory.
2. Weinberger, K. Q., & Saul, L. K. (2009). *Distance metric learning for large margin nearest neighbor classification.* JMLR.
3. CS231n Course Notes: [Image Classification](https://cs231n.github.io/classification/)

---

**End of Theory Notes** 📚
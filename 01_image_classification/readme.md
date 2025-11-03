# 📚 Lecture 2: Image Classification & k-Nearest Neighbors

> **CS231n: Deep Learning for Computer Vision**  
> **Author:** Hamza  
> **Date:** November 2024  
> **Status:** ✅ Completed

---

## 🎯 Learning Objectives

By the end of this lecture, I understood:

- ✅ **Image Classification Pipeline:** How to frame vision problems as classification tasks
- ✅ **k-Nearest Neighbors (kNN):** A simple, non-parametric baseline classifier
- ✅ **Distance Metrics:** L1 (Manhattan) and L2 (Euclidean) distance for comparing images
- ✅ **Hyperparameter Tuning:** Using validation sets to choose optimal k and distance metrics
- ✅ **Train/Val/Test Split:** Why we need separate datasets for hyperparameter search

---

## 📂 Repository Structure

```
lecture-02-image-classification/
├── README.md                      # This file
├── 01-theory-notes.md            # Detailed theory and math explanations
├── 02-knn-implementation.ipynb   # Clean kNN implementation from scratch
├── 03-experiments.ipynb          # Distance metrics comparison & visualizations
└── assets/                       # Images and plots
```

---

## 🧠 Key Concepts

### **1. The Image Classification Problem**

**Goal:** Given an image, assign it a label from a fixed set of categories.

**Challenges:**
- **Viewpoint variation:** Same object from different angles
- **Illumination:** Lighting conditions affect pixel values
- **Deformation:** Objects can deform (e.g., cats in different poses)
- **Occlusion:** Parts of objects may be hidden
- **Background clutter:** Objects blend with surroundings
- **Intra-class variation:** Cats come in many breeds, sizes, colors

### **2. k-Nearest Neighbors (kNN)**

**Core Idea:** "You are the average of your k nearest neighbors"

**Algorithm:**
1. **Train:** Memorize all training data (O(1) time)
2. **Predict:** For each test image:
   - Compute distance to all training images
   - Find k nearest neighbors
   - Predict label by majority vote

**Time Complexity:**
- Train: O(1) - just store data
- Test: O(N) per image - compare to all N training examples

**Why kNN is bad for images:**
- ❌ Slow at test time (unacceptable for production)
- ❌ Distance metrics on raw pixels are not semantically meaningful
- ❌ Curse of dimensionality (need exponentially more data in high dimensions)

### **3. Distance Metrics**

| Metric | Formula | Use Case |
|--------|---------|----------|
| **L1 (Manhattan)** | `Σ|I₁ - I₂|` | Coordinate-wise differences |
| **L2 (Euclidean)** | `√(Σ(I₁ - I₂)²)` | Geometric distance |

**Key Insight:** Choice of distance metric is a **hyperparameter** - must be chosen using validation data.

### **4. Hyperparameter Selection**

**Hyperparameters:** Choices we make before training (not learned from data)
- `k`: Number of neighbors to consider
- Distance metric: L1 vs L2

**NEVER choose hyperparameters based on test set performance!**

**Correct Process:**
1. Split data: 80% train, 10% validation, 10% test
2. Try different hyperparameters on validation set
3. Choose best hyperparameters
4. **Run on test set once** at the very end

---

## 💻 Implementation Highlights

### **kNN Classifier (NumPy)**

```python
class NearestNeighbor:
    def train(self, X, y):
        """Simply memorize training data"""
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        """Predict labels using L1 distance"""
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        
        for i in range(num_test):
            # Compute L1 distance to all training examples
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        
        return Ypred
```

**Key Operations:**
- `np.abs()` - Absolute value for L1 distance
- `np.sum(..., axis=1)` - Sum across feature dimension
- `np.argmin()` - Find index of smallest distance

---

## 📊 Experiments & Results

### **Experiment 1: L1 vs L2 Distance**
- Compared Manhattan and Euclidean distance on CIFAR-10 subset
- **Finding:** L2 slightly outperforms L1 for natural images

### **Experiment 2: Effect of k**
- Tested k = 1, 3, 5, 10, 50
- **Finding:** k=5 provides good balance (reduces noise, maintains local structure)

### **Experiment 3: Validation Set Size**
- Tested 10%, 20%, 30% validation split
- **Finding:** 10-20% is sufficient for hyperparameter selection

---

## 🎓 What I Learned

### **Theoretical Insights**
1. **kNN is a lazy learner:** No training phase, all computation at test time
2. **Curse of dimensionality:** In high dimensions, all points are far apart
3. **Raw pixels ≠ semantic similarity:** Shifted images have large L2 distance despite being visually identical

### **Practical Takeaways**
1. **kNN is not used in production computer vision:** Too slow, doesn't work well with raw pixels
2. **However, kNN on learned features (embeddings) is powerful:** Modern systems use deep learning to extract features, then kNN for retrieval
3. **Always use validation set for hyperparameter tuning:** This is fundamental ML hygiene

### **Implementation Skills**
1. ✅ Implemented kNN from scratch using NumPy
2. ✅ Understood vectorized distance computation
3. ✅ Practiced train/val/test split methodology

---

## 🔗 Resources

- **Lecture Slides:** [CS231n Lecture 2](http://cs231n.stanford.edu/slides/2024/lecture_2.pdf)
- **Course Notes:** [Image Classification Pipeline](https://cs231n.github.io/classification/)
- **Related Papers:**
  - Cover & Hart (1967): *Nearest Neighbor Pattern Classification*
  - Weinberger & Saul (2009): *Distance Metric Learning for Large Margin Nearest Neighbor Classification*

---

## 🚀 Next Steps

- ✅ **Lecture 2 Complete:** Understood kNN and its limitations
- ⏭️ **Lecture 3:** Linear Classification & Loss Functions
- 🎯 **Goal:** Move from non-parametric (kNN) to parametric models (linear classifiers)

---

## 📌 Key Quotes

> *"k-Nearest Neighbors: Simple to understand, terrible in practice for images."*  
> — CS231n Teaching Team

> *"Never tune hyperparameters on test data. That's like studying with the exam questions in hand."*  
> — ML Best Practices

---

## ✅ Completion Checklist

- [x] Understood image classification problem formulation
- [x] Implemented kNN classifier from scratch
- [x] Experimented with L1 vs L2 distance metrics
- [x] Practiced proper train/val/test split methodology
- [x] Identified why kNN fails for raw pixel comparison
- [x] Ready to move to parametric models (linear classifiers)

---

**Status:** Lecture 2 complete. Moving to Lecture 3 (Gradient Descent & Optimization). 🚀
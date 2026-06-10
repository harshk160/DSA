# ⏱️ Time Complexity Cheat Sheet

**Purpose:** Quick reference for Big O complexity of all data structures and operations.

---

## 📊 COMPLETE COMPLEXITY TABLE

### Basic Data Structures

| Data Structure | Access | Search | Insert | Delete | Space | Notes |
|----------------|--------|--------|--------|--------|-------|-------|
| **Array** | O(1) | O(n) | O(n) | O(n) | O(n) | Fixed size |
| **Dynamic Array** | O(1) | O(n) | O(1)* | O(n) | O(n) | *Amortized |
| **Singly Linked List** | O(n) | O(n) | O(1)† | O(1)† | O(n) | †At head |
| **Doubly Linked List** | O(n) | O(n) | O(1)† | O(1)† | O(n) | †At ends |
| **Stack** | O(n) | O(n) | O(1) | O(1) | O(n) | Only top |
| **Queue** | O(n) | O(n) | O(1) | O(1) | O(n) | FIFO |
| **Circular Queue** | O(n) | O(n) | O(1) | O(1) | O(n) | No overflow |
| **Deque** | O(n) | O(n) | O(1) | O(1) | O(n) | Both ends |

### Tree Structures

| Data Structure | Access | Search | Insert | Delete | Space | Notes |
|----------------|--------|--------|--------|--------|-------|-------|
| **Binary Tree** | O(n) | O(n) | O(n) | O(n) | O(n) | Not ordered |
| **BST (Avg)** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Balanced |
| **BST (Worst)** | O(n) | O(n) | O(n) | O(n) | O(n) | Skewed |
| **AVL Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Self-balancing |
| **Red-Black Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Self-balancing |
| **Min/Max Heap** | O(1)‡ | O(n) | O(log n) | O(log n) | O(n) | ‡Get min/max |
| **B-Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Database use |

### Hash-based Structures

| Data Structure | Access | Search | Insert | Delete | Space | Notes |
|----------------|--------|--------|--------|--------|-------|-------|
| **Hash Table (Avg)** | N/A | O(1) | O(1) | O(1) | O(n) | Good hash |
| **Hash Table (Worst)** | N/A | O(n) | O(n) | O(n) | O(n) | Collisions |
| **Hash Set** | N/A | O(1) | O(1) | O(1) | O(n) | Unique values |
| **Hash Map** | N/A | O(1) | O(1) | O(1) | O(n) | Key-value |

### Graph Structures

| Data Structure | Space | Add Vertex | Add Edge | Remove Edge | Query |
|----------------|-------|------------|----------|-------------|-------|
| **Adjacency Matrix** | O(V²) | O(V²) | O(1) | O(1) | O(1) |
| **Adjacency List** | O(V+E) | O(1) | O(1) | O(E) | O(V) |
| **Edge List** | O(E) | O(1) | O(1) | O(E) | O(E) |

### Advanced Structures

| Data Structure | Build | Query | Update | Space | Use Case |
|----------------|-------|-------|--------|-------|----------|
| **Trie** | O(m·n) | O(m) | O(m) | O(ALPHABET·n·m) | Prefix search |
| **Segment Tree** | O(n) | O(log n) | O(log n) | O(n) | Range queries |
| **Fenwick Tree** | O(n) | O(log n) | O(log n) | O(n) | Prefix sums |
| **Union-Find** | O(n) | O(α(n)) | O(α(n)) | O(n) | Connectivity |
| **Suffix Array** | O(n log n) | O(m + log n) | N/A | O(n) | Pattern matching |

*m = length of string/word, n = number of elements, α(n) = inverse Ackermann (nearly constant)*

---

## 🔍 OPERATION-SPECIFIC COMPLEXITIES

### Dynamic Array (Vector)

| Operation | Best | Average | Worst | Notes |
|-----------|------|---------|-------|-------|
| push_back | O(1) | O(1) | O(n) | Worst when resize |
| pop_back | O(1) | O(1) | O(1) | Just decrement |
| insert(front) | O(n) | O(n) | O(n) | Shift all |
| insert(middle) | O(n) | O(n) | O(n) | Shift half |
| insert(end) | O(1) | O(1) | O(n) | Like push_back |
| erase(front) | O(n) | O(n) | O(n) | Shift all |
| erase(middle) | O(n) | O(n) | O(n) | Shift half |
| erase(end) | O(1) | O(1) | O(1) | Like pop_back |
| at/[] | O(1) | O(1) | O(1) | Direct access |
| resize | O(n) | O(n) | O(n) | May reallocate |
| reserve | O(n) | O(n) | O(n) | Reallocate |

### Linked List

| Operation | Singly | Doubly | Notes |
|-----------|--------|--------|-------|
| insertAtHead | O(1) | O(1) | Direct |
| insertAtTail | O(n)† | O(1) | †Without tail |
| insertAtPosition | O(n) | O(n) | Traverse |
| deleteAtHead | O(1) | O(1) | Direct |
| deleteAtTail | O(n) | O(1) | Need prev |
| deleteAtPosition | O(n) | O(n) | Traverse |
| search | O(n) | O(n) | Linear scan |
| reverse | O(n) | O(n) | All nodes |
| getMiddle | O(n) | O(n) | Slow-fast |
| detectCycle | O(n) | O(n) | Floyd's |

### Binary Search Tree

| Operation | Balanced | Skewed | Notes |
|-----------|----------|--------|-------|
| search | O(log n) | O(n) | Binary search |
| insert | O(log n) | O(n) | May unbalance |
| delete | O(log n) | O(n) | 3 cases |
| findMin | O(log n) | O(n) | Leftmost |
| findMax | O(log n) | O(n) | Rightmost |
| inorder | O(n) | O(n) | All nodes |
| preorder | O(n) | O(n) | All nodes |
| postorder | O(n) | O(n) | All nodes |
| height | O(n) | O(n) | All nodes |

### Heap

| Operation | Time | Notes |
|-----------|------|-------|
| insert | O(log n) | Heapify up |
| extractMin/Max | O(log n) | Heapify down |
| getMin/Max | O(1) | Root element |
| heapifyUp | O(log n) | Height of tree |
| heapifyDown | O(log n) | Height of tree |
| buildHeap | O(n) | Floyd's method |
| delete | O(log n) | Replace + heapify |
| search | O(n) | Not optimized |

### Hash Table

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| insert | O(1) | O(n) | With collisions |
| search | O(1) | O(n) | With collisions |
| delete | O(1) | O(n) | With collisions |
| rehash | O(n) | O(n) | Copy all |

**Load Factor Impact:**
- α < 0.5: Very fast, O(1) expected
- 0.5 ≤ α < 0.7: Fast, O(1) expected
- 0.7 ≤ α < 1.0: Slower, consider rehashing
- α ≥ 1.0: Slow, O(n) possible

### Graph Algorithms

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| BFS | O(V + E) | O(V) | Queue-based |
| DFS | O(V + E) | O(V) | Stack/recursion |
| Dijkstra | O((V+E) log V) | O(V) | Priority queue |
| Bellman-Ford | O(V·E) | O(V) | Negative edges |
| Floyd-Warshall | O(V³) | O(V²) | All pairs |
| Prim's MST | O(E log V) | O(V) | Priority queue |
| Kruskal's MST | O(E log E) | O(V) | Union-find |
| Topological Sort | O(V + E) | O(V) | DAG only |
| Tarjan's SCC | O(V + E) | O(V) | Strongly connected |

### Trie

| Operation | Time | Notes |
|-----------|------|-------|
| insert(word) | O(m) | m = word length |
| search(word) | O(m) | m = word length |
| startsWith(prefix) | O(m) | m = prefix length |
| delete(word) | O(m) | m = word length |
| countWords | O(n) | n = total nodes |
| getAllWords | O(n) | n = total nodes |

### Segment Tree

| Operation | Time | Notes |
|-----------|------|-------|
| build | O(n) | Bottom-up |
| query(range) | O(log n) | Binary search |
| update(point) | O(log n) | Single element |
| update(range) | O(log n) | With lazy prop |

### Union-Find

| Operation | Time | Notes |
|-----------|------|-------|
| find (naive) | O(n) | Without optimization |
| find (path compression) | O(log n) | With optimization |
| find (optimal) | O(α(n)) | Path + union by rank |
| union (naive) | O(n) | Without optimization |
| union (by rank) | O(log n) | With optimization |
| union (optimal) | O(α(n)) | With path compression |

*α(n) = inverse Ackermann function ≈ O(1) for practical purposes*

---

## 🎯 COMPARISON BY OPERATION TYPE

### Best for Access (O(1))
1. **Array / Vector** - Random access
2. **Hash Table** - Key-based access (average)
3. **Heap** - Min/Max only

### Best for Search
1. **Hash Table** - O(1) average
2. **Balanced BST** - O(log n)
3. **Sorted Array** - O(log n) with binary search
4. **Trie** - O(m) for strings

### Best for Insert
1. **Hash Table** - O(1) average
2. **Linked List (at head)** - O(1)
3. **Heap** - O(log n)
4. **Balanced BST** - O(log n)

### Best for Delete
1. **Hash Table** - O(1) average
2. **Linked List (at head)** - O(1)
3. **Heap** - O(log n)
4. **Balanced BST** - O(log n)

### Best for Range Queries
1. **Segment Tree** - O(log n)
2. **Fenwick Tree** - O(log n)
3. **Sorted Array** - O(log n + k) where k = results

### Best for Ordered Data
1. **Balanced BST** - O(log n) operations
2. **Sorted Array** - O(1) access, O(n) insert
3. **Skip List** - O(log n) probabilistic

---

## 📈 GROWTH RATES COMPARISON

```
O(1)      < O(log n) < O(√n)     < O(n)      < O(n log n) < O(n²)     < O(2ⁿ)     < O(n!)
Constant    Logarithmic Square Root  Linear     Linearithmic  Quadratic   Exponential Factorial

For n = 1,000,000:
O(1)      = 1 operation
O(log n)  ≈ 20 operations
O(√n)     = 1,000 operations
O(n)      = 1,000,000 operations
O(n log n)≈ 20,000,000 operations
O(n²)     = 1,000,000,000,000 operations
O(2ⁿ)     = Impossible (too large)
O(n!)     = Impossible (too large)
```

---

## 🔄 AMORTIZED ANALYSIS

### Dynamic Array push_back()
- Individual operation: O(1) or O(n)
- Sequence of n operations: O(n)
- **Amortized per operation: O(1)**

**Why?**
```
Resize happens at: 1, 2, 4, 8, 16, 32, ...
Cost to insert n elements: n + (1+2+4+8+...+n/2) = n + (n-1) = 2n-1
Average per element: (2n-1)/n ≈ 2 = O(1)
```

### Union-Find with Optimizations
- Individual operation: O(log n) without optimization
- With path compression + union by rank: O(α(n))
- **Nearly O(1) for practical purposes**

---

## ⚠️ SPACE COMPLEXITY NOTES

### Recursive Algorithms
- **Call Stack Space:** O(h) where h = recursion depth
- Tree recursion: O(height)
- Linear recursion: O(n) worst case

### Iterative vs Recursive
```
Recursive DFS:
Time: O(V + E)
Space: O(V) for recursion stack

Iterative DFS:
Time: O(V + E)
Space: O(V) for explicit stack

Same time, but iterative may have less overhead
```

### In-place vs Out-of-place
```
In-place: Modify original data, O(1) extra space
Out-of-place: Create copy, O(n) extra space

Example - Array Reversal:
In-place: O(1) space
Out-of-place: O(n) space
```

---

## 💡 PRACTICAL BENCHMARKS

### How long for 1 million operations?

| Complexity | Operations | ~Time (1M ops/sec) |
|------------|------------|-------------------|
| O(1) | 1 | < 1 microsecond |
| O(log n) | 20 | 20 microseconds |
| O(n) | 1,000,000 | 1 second |
| O(n log n) | 20,000,000 | 20 seconds |
| O(n²) | 1,000,000,000,000 | 31,700 years |

### Acceptable Complexity by Input Size

| Input Size (n) | Maximum Complexity | Example |
|----------------|-------------------|---------|
| n ≤ 10 | O(n!) | Brute force |
| n ≤ 20 | O(2ⁿ) | Backtracking |
| n ≤ 500 | O(n³) | Floyd-Warshall |
| n ≤ 5,000 | O(n²) | Bubble sort |
| n ≤ 1,000,000 | O(n log n) | Merge sort |
| n ≤ 100,000,000 | O(n) | Linear scan |
| Any n | O(log n) | Binary search |
| Any n | O(1) | Hash table |

---

## 🎯 QUICK DECISION TABLE

**Given operation requirement:**

| Need | Best Choice | Complexity |
|------|-------------|------------|
| Fast random access | Array | O(1) |
| Fast search by key | Hash Table | O(1) avg |
| Fast insert/delete at ends | Deque | O(1) |
| Maintain sorted order | BST | O(log n) |
| Fast min/max queries | Heap | O(1) get |
| Range queries | Segment Tree | O(log n) |
| Prefix search | Trie | O(m) |
| Connectivity queries | Union-Find | O(α(n)) |
| LIFO operations | Stack | O(1) |
| FIFO operations | Queue | O(1) |

---

**Total Pages:** ~8-10 pages when printed  
**Purpose:** Quick complexity lookup during problem-solving  
**Usage:** Keep handy during coding interviews and practice

---

*Master these complexities to make optimal data structure choices!*

# 📝 DSA Implementation Checklist (Print-Friendly)

**Purpose:** A condensed guide showing WHAT to implement for each data structure.

---

## 🟢 BASIC DATA STRUCTURES

### 1. Dynamic Array (Vector)

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] push_back(value) - Add at end
- [ ] pop_back() - Remove last
- [ ] at(index) - Access with bounds check
- [ ] operator[] - Direct access
- [ ] size() - Current size
- [ ] capacity() - Allocated space
- [ ] resize(newSize) - Change size
- [ ] reserve(newCapacity) - Pre-allocate
- [ ] clear() - Remove all
- [ ] empty() - Check if empty
- [ ] insert(index, value) - Insert at position

**Key Implementation Points:**
- Start with capacity = 1, double when full
- Amortized O(1) push_back
- resize() vs reserve(): resize changes size, reserve changes capacity

---

### 2. Singly Linked List

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insertAtHead(value) - Add at beginning
- [ ] insertAtTail(value) - Add at end
- [ ] insertAtPosition(pos, val) - Insert at index
- [ ] deleteAtHead() - Remove first
- [ ] deleteAtTail() - Remove last
- [ ] deleteAtPosition(pos) - Remove at index
- [ ] search(value) - Find element
- [ ] reverse() - Reverse list
- [ ] getMiddle() - Find middle (slow-fast pointers)
- [ ] detectCycle() - Check for cycle (Floyd's algorithm)
- [ ] display() - Print all elements
- [ ] size() - Get length

**Key Implementation Points:**
- Each node has: data + next pointer
- Keep head pointer and length counter
- Reverse: Use 3 pointers (prev, curr, next)
- Middle/Cycle: Use slow-fast pointer technique

---

### 3. Doubly Linked List

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insertAtHead(value)
- [ ] insertAtTail(value)
- [ ] insertAtPosition(pos, val)
- [ ] deleteAtHead()
- [ ] deleteAtTail()
- [ ] deleteAtPosition(pos)
- [ ] deleteNode(node) - Delete given node pointer
- [ ] reverse() - Reverse list
- [ ] displayForward() - Print head to tail
- [ ] displayBackward() - Print tail to head
- [ ] size()

**Key Implementation Points:**
- Each node has: prev + data + next
- Keep both head and tail pointers
- O(1) deletion at both ends (advantage over singly)
- Can traverse backward

---

### 4. Stack

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] push(value) - Add to top
- [ ] pop() - Remove from top
- [ ] peek()/top() - View top element
- [ ] isEmpty() - Check if empty
- [ ] isFull() - Check if full (array implementation)
- [ ] size() - Get count
- [ ] clear() - Remove all

**Key Implementation Points:**
- Array-based: Use top pointer (starts at -1)
- top = -1 means empty
- Increment top before push, decrement after pop
- LIFO: Last In, First Out

---

### 5. Queue

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] enqueue(value) - Add to rear
- [ ] dequeue() - Remove from front
- [ ] getFront() - View front element
- [ ] getRear() - View rear element
- [ ] isEmpty() - Check if empty
- [ ] isFull() - Check if full
- [ ] size() - Get count

**Key Implementation Points:**
- Use circular queue to avoid false overflow
- rear = (rear + 1) % capacity
- front = (front + 1) % capacity
- Use count variable to distinguish empty/full
- FIFO: First In, First Out

---

## 🟡 INTERMEDIATE DATA STRUCTURES

### 6. Binary Tree

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insert(value) - Level order insertion
- [ ] inorder() - Left → Root → Right
- [ ] preorder() - Root → Left → Right
- [ ] postorder() - Left → Right → Root
- [ ] levelOrder() - BFS traversal
- [ ] height() - Max depth
- [ ] diameter() - Longest path
- [ ] search(value) - Find node
- [ ] countNodes() - Total nodes
- [ ] countLeaves() - Count leaf nodes

**Key Implementation Points:**
- Each node: left + data + right
- Recursive traversals are simplest
- Use queue for level order (BFS)
- Height: 1 + max(left height, right height)

---

### 7. Binary Search Tree (BST)

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insert(value) - Maintain BST property
- [ ] search(value) - O(log n) average
- [ ] delete(value) - Handle 3 cases
- [ ] findMin() - Leftmost node
- [ ] findMax() - Rightmost node
- [ ] inorder() - Gives sorted order
- [ ] validate() - Check if valid BST
- [ ] LCA(n1, n2) - Lowest common ancestor
- [ ] successor(value) - Next larger
- [ ] predecessor(value) - Next smaller

**Key Implementation Points:**
- BST Property: left < root < right
- Delete cases: leaf, one child, two children
- Inorder traversal gives sorted sequence
- Average O(log n), worst O(n) if unbalanced

---

### 8. Min Heap

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insert(value) - Add and heapify up
- [ ] extractMin() - Remove root and heapify down
- [ ] getMin() - View minimum (root)
- [ ] heapifyUp(index) - Fix upward
- [ ] heapifyDown(index) - Fix downward
- [ ] buildHeap(array) - Create from array
- [ ] size() - Get count
- [ ] isEmpty() - Check if empty

**Key Implementation Points:**
- Array-based: parent = (i-1)/2, left = 2i+1, right = 2i+2
- Min heap: parent ≤ children
- Insert: Add at end, heapify up
- Extract: Replace root with last, heapify down
- buildHeap: O(n) using bottom-up

---

### 9. Max Heap

**Operations to Implement:**
- [ ] Same as Min Heap but with max property
- [ ] extractMax() instead of extractMin()
- [ ] getMax() instead of getMin()

**Key Implementation Points:**
- Max heap: parent ≥ children
- Everything else same as min heap
- Used for priority queue (highest priority first)

---

### 10. Hash Table

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insert(key, value) - Add key-value pair
- [ ] search(key) - Find value by key
- [ ] delete(key) - Remove key-value pair
- [ ] hashFunction(key) - Compute hash
- [ ] rehash() - Resize when load factor high
- [ ] size() - Get count
- [ ] display() - Show all entries

**Key Implementation Points:**
- Collision handling: Separate chaining or open addressing
- Load factor = size / capacity (rehash when > 0.7)
- Hash function: key % tableSize (simple)
- Chaining: Use linked list at each bucket
- Open addressing: Linear probing (check next slot)

---

## 🔴 ADVANCED DATA STRUCTURES

### 11. Graph (Adjacency List)

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] addVertex(v) - Add node
- [ ] addEdge(u, v) - Add edge (directed/undirected)
- [ ] removeEdge(u, v) - Remove edge
- [ ] BFS(start) - Breadth-first search
- [ ] DFS(start) - Depth-first search (recursive & iterative)
- [ ] hasPath(u, v) - Check connectivity
- [ ] isCyclic() - Detect cycle
- [ ] topologicalSort() - For DAG
- [ ] display() - Print graph

**Key Implementation Points:**
- Adjacency list: vector<vector<int>> or array of lists
- BFS: Use queue
- DFS: Use stack or recursion
- Keep visited array to avoid revisiting
- Undirected: Add edge in both directions

---

### 12. Trie (Prefix Tree)

**Operations to Implement:**
- [ ] Constructor/Destructor
- [ ] insert(word) - Add word
- [ ] search(word) - Exact match
- [ ] startsWith(prefix) - Prefix search
- [ ] delete(word) - Remove word
- [ ] countWordsWithPrefix(prefix)
- [ ] getAllWords() - Return all stored words
- [ ] isEmpty() - Check if empty

**Key Implementation Points:**
- Each node has: children[26] + isEndOfWord flag
- Children array for 'a'-'z'
- Insert: Create nodes for each character
- Search: Follow path, check isEndOfWord
- Used for: autocomplete, spell checker, IP routing

---

### 13. Union-Find (Disjoint Set Union)

**Operations to Implement:**
- [ ] Constructor - Initialize with makeset
- [ ] find(x) - Find representative (with path compression)
- [ ] union(x, y) - Merge sets (union by rank/size)
- [ ] connected(x, y) - Check if same set
- [ ] countSets() - Number of disjoint sets

**Key Implementation Points:**
- parent array: parent[i] = parent of i
- rank/size array for optimization
- Path compression: Make all nodes point to root
- Union by rank: Attach smaller tree to larger
- Nearly O(1) amortized time with both optimizations

---

### 14. Segment Tree

**Operations to Implement:**
- [ ] Constructor - Build from array
- [ ] build(arr, node, start, end) - Recursive build
- [ ] query(node, start, end, L, R) - Range query
- [ ] update(node, start, end, idx, val) - Point update
- [ ] updateRange(node, start, end, L, R, val) - Range update (lazy)

**Key Implementation Points:**
- Tree array: size = 4 * n
- Each node stores aggregate of range (sum, min, max)
- Build: O(n), Query: O(log n), Update: O(log n)
- Lazy propagation for range updates
- Used for: Range sum/min/max queries

---

## ✅ IMPLEMENTATION CHECKLIST

### Week 1: Basic Structures (Days 1-7)
- [ ] Day 1-2: Dynamic Array
- [ ] Day 3-4: Singly Linked List
- [ ] Day 5: Doubly Linked List
- [ ] Day 6: Stack
- [ ] Day 7: Queue

### Week 2: Intermediate Structures (Days 8-14)
- [ ] Day 8-9: Binary Tree
- [ ] Day 10-11: BST
- [ ] Day 12: Min/Max Heap
- [ ] Day 13-14: Hash Table

### Week 3: Advanced Structures (Days 15-21)
- [ ] Day 15-16: Graph
- [ ] Day 17-18: Trie
- [ ] Day 19-20: Union-Find
- [ ] Day 21: Segment Tree

---

## 🧪 TESTING CHECKLIST

### For Each Data Structure:

**Basic Operations:**
- [ ] Empty state operations
- [ ] Single element operations
- [ ] Multiple element operations
- [ ] Full capacity operations (if applicable)

**Edge Cases:**
- [ ] Insert/delete at boundaries (start, end)
- [ ] Operations on empty structure
- [ ] Operations on single element
- [ ] Duplicate values
- [ ] Invalid inputs (negative index, null values)

**Stress Testing:**
- [ ] Large dataset (1000+ elements)
- [ ] Random operations
- [ ] Repeated operations
- [ ] Memory leak check (valgrind)

---

## 📊 SUCCESS CRITERIA

By the end of 3 weeks, you should be able to:

✅ Implement any basic DS in 15-20 minutes  
✅ Implement any intermediate DS in 30-45 minutes  
✅ Explain time/space complexity of all operations  
✅ Identify which DS to use for a given problem  
✅ Debug your implementations independently  
✅ Write clean code with proper memory management  

---

## 💡 STUDY TIPS

1. **Code from scratch** - No copy-paste, build muscle memory
2. **Use pen and paper first** - Draw the structure and operations
3. **Write comments** - Explain what each line does
4. **Test thoroughly** - Use multiple test cases
5. **Compare with STL** - See how professionals do it
6. **Time yourself** - Aim for 30-45 mins per structure
7. **Redo weekly** - Repetition builds mastery
8. **Explain to others** - Teaching solidifies understanding

---

## 🎯 QUICK REFERENCE FORMULAS

**Array-based DS:**
- Parent of i: (i-1)/2
- Left child of i: 2*i + 1
- Right child of i: 2*i + 2

**Circular Queue:**
- next = (current + 1) % capacity
- prev = (current - 1 + capacity) % capacity

**Hash Function:**
- Simple: key % tableSize
- Better: (a*key + b) % tableSize (a, b are primes)

**Tree Height:**
- height = 1 + max(leftHeight, rightHeight)
- Perfect binary tree: height = log₂(n+1) - 1

**Load Factor:**
- loadFactor = numberOfElements / tableSize
- Rehash when > 0.7

---

**Total Pages:** ~10-12 pages when printed  
**Format:** Optimized for printing and quick reference  
**Use:** Keep beside you while coding

---

*Print this document and check off each operation as you implement it!*

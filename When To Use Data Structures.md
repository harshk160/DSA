# 🎯 When to Use Which Data Structure

**Purpose:** Decision guide for choosing the right data structure for any problem.

---

## 📋 QUICK DECISION FLOWCHART

```
START: What do you need to do?

├─ Need random access to elements? → Array/Vector
├─ Need fast search by key? → Hash Table
├─ Need sorted order maintained? → BST/Balanced Tree
├─ Need min/max quickly? → Heap
├─ Need LIFO operations? → Stack
├─ Need FIFO operations? → Queue
├─ Need both-end operations? → Deque
├─ Need prefix/pattern matching? → Trie
├─ Need range queries? → Segment Tree
└─ Need connectivity checks? → Union-Find
```

---

## 🟢 BASIC DATA STRUCTURES

### Array / Static Array

✅ **Use When:**
- Size is known at compile time
- Need O(1) random access
- Memory is continuous (cache-friendly)
- Simple iteration over all elements
- No insertions/deletions after creation
- Want minimal memory overhead

❌ **Avoid When:**
- Size is unknown or changes frequently
- Need frequent insertions/deletions
- Memory fragmentation is acceptable
- Size grows dynamically

**Real-world Examples:**
- Fixed-size configuration settings
- Lookup tables (days of week, months)
- Small, fixed collections
- Buffer of known size

---

### Dynamic Array (Vector)

✅ **Use When:**
- Size is unknown but grows
- Need random access O(1)
- Mostly append operations
- Want automatic memory management
- Cache performance matters
- Need to reserve space upfront

❌ **Avoid When:**
- Frequent insertions in middle O(n)
- Need O(1) insertions at front
- Memory is very limited (capacity overhead)
- Fixed size known at compile time

**Real-world Examples:**
- List of students in a class (grows over semester)
- Shopping cart items
- Dynamic collections in general
- Most general-purpose lists

**Pro Tips:**
- Use `reserve()` if final size is known
- Use `push_back()` instead of `insert()`
- Consider `deque` if front insertions needed

---

### Singly Linked List

✅ **Use When:**
- Frequent insertions/deletions at head O(1)
- Don't need random access
- Size changes frequently
- Memory fragmentation acceptable
- Implementing other DS (stack, queue, graph)
- Each element has next pointer overhead OK

❌ **Avoid When:**
- Need random access (O(n) vs O(1))
- Cache performance critical
- Memory overhead matters (pointer per node)
- Need to access middle elements frequently
- Need backward traversal

**Real-world Examples:**
- Implementation of stack (linked-based)
- Implementation of queue (linked-based)
- Undo list (recent first)
- Browser tab list (simple navigation)
- Implementing adjacency lists in graphs

**Pro Tips:**
- Keep tail pointer for O(1) tail insertion
- Use for simple LIFO/FIFO needs
- Good for learning pointers

---

### Doubly Linked List

✅ **Use When:**
- Need bidirectional traversal
- Frequent deletions given node pointer O(1)
- Need O(1) deletion at both ends
- Implementing LRU cache
- Browser history (back/forward)
- Music player (prev/next track)
- Undo/Redo functionality

❌ **Avoid When:**
- Memory is very limited (2 pointers per node)
- Only need forward traversal (use singly)
- Need random access
- Simple stack/queue suffices

**Real-world Examples:**
- LRU Cache implementation
- Browser history
- Text editor undo/redo
- Music playlist with prev/next
- Train bogies (can go both ways)

**Pro Tips:**
- Essential for LRU Cache (O(1) delete)
- Keep both head and tail pointers
- More memory but more flexible

---

### Stack

✅ **Use When:**
- Need LIFO (Last In, First Out) order
- Function call management (recursion)
- Expression evaluation (postfix, infix)
- Backtracking algorithms
- Undo operations
- Syntax parsing (brackets, HTML tags)
- Depth-First Search (DFS)
- Reversing elements

❌ **Avoid When:**
- Need FIFO order (use queue)
- Need to access middle elements
- Need random access
- Want to search for elements

**Real-world Examples:**
- Browser back button
- Text editor undo
- Recursive function calls
- Calculator expression evaluation
- Maze solving (backtracking)
- Matching brackets in code
- Reversing strings/arrays

**Pro Tips:**
- Use for any backtracking problem
- Natural choice for recursion
- Consider vector for array-based stack

---

### Queue

✅ **Use When:**
- Need FIFO (First In, First Out) order
- BFS traversal in graphs/trees
- Scheduling tasks (printer, CPU)
- Buffering (IO, data streams)
- Level-order processing
- Handling requests in order
- Producer-consumer problems

❌ **Avoid When:**
- Need LIFO order (use stack)
- Need random access
- Want to remove from middle

**Real-world Examples:**
- Printer queue
- CPU task scheduling
- Call center (first caller served first)
- BFS in graphs/trees
- Web server request handling
- Message queues

**Pro Tips:**
- Use circular queue to avoid false overflow
- Deque if need both-end operations
- Priority queue for priority-based serving

---

### Deque (Double-Ended Queue)

✅ **Use When:**
- Need insertions/deletions at both ends O(1)
- Sliding window problems
- Palindrome checking
- Implementing undo/redo with both operations
- Need flexibility of both stack and queue

❌ **Avoid When:**
- Only need one-end operations (use stack/queue)
- Need random access in middle
- Simple FIFO/LIFO suffices

**Real-world Examples:**
- Sliding window maximum/minimum
- Palindrome checker
- Stealing work algorithm
- Double-ended workflows

**Pro Tips:**
- Perfect for sliding window problems
- More flexible than stack or queue
- Slightly more overhead than simple queue

---

## 🟡 INTERMEDIATE DATA STRUCTURES

### Binary Tree

✅ **Use When:**
- Need hierarchical data representation
- Building expression trees
- Learning tree concepts
- Implementing other tree types
- Huffman coding

❌ **Avoid When:**
- Need fast search (use BST)
- Need sorted order
- Need balanced operations

**Real-world Examples:**
- Expression trees (math expressions)
- Organization hierarchy
- File system structure
- Huffman encoding trees

---

### Binary Search Tree (BST)

✅ **Use When:**
- Need sorted order maintained
- Frequent search operations O(log n) avg
- Need in-order traversal for sorting
- Range queries needed
- Implementing sets/maps without hashing

❌ **Avoid When:**
- Data is unsorted or random access needed
- BST might become unbalanced (use AVL/Red-Black)
- Hash table can be used (faster O(1))
- Tree can degenerate to O(n)

**Real-world Examples:**
- Implementing TreeSet/TreeMap
- Maintaining sorted data with insertions
- Dictionary with ordered iteration
- Range queries (find all between x and y)
- Database indexing (B-tree variant)

**Pro Tips:**
- Use balanced BST (AVL, Red-Black) in production
- Good for learning tree operations
- Self-balancing trees for worst-case O(log n)

---

### AVL Tree / Red-Black Tree

✅ **Use When:**
- Need guaranteed O(log n) operations
- BST but need to avoid skewing
- Implementing production-grade sets/maps
- Need strict balancing (AVL) or faster insertion (Red-Black)

❌ **Avoid When:**
- Simple BST is sufficient
- Hash table can be used
- Don't need sorted order

**Real-world Examples:**
- C++ std::map and std::set
- Java TreeMap and TreeSet
- Database indexing
- In-memory databases

---

### Min/Max Heap

✅ **Use When:**
- Need fast access to min/max O(1)
- Implementing priority queue
- Finding k smallest/largest elements
- Median finding (two heaps)
- Heap sort
- Scheduling tasks by priority

❌ **Avoid When:**
- Need fast search for arbitrary element
- Need sorted order (use BST)
- All elements equally important
- Need to access middle elements

**Real-world Examples:**
- Priority task scheduling
- Dijkstra's shortest path
- Finding median in stream
- Top k frequent elements
- Event-driven simulation
- Operating system task scheduling

**Pro Tips:**
- Min heap for smallest, max heap for largest
- Two heaps for median finding
- Heapify is O(n), faster than inserting n times

---

### Hash Table / Hash Map

✅ **Use When:**
- Need O(1) average search/insert/delete
- Key-value pairs needed
- Fast lookup by key is priority
- Don't need sorted order
- Have good hash function
- Collisions are rare

❌ **Avoid When:**
- Need sorted order (use BST)
- Need range queries
- Hash function is poor (many collisions)
- Memory is very limited
- Keys are not hashable

**Real-world Examples:**
- Caching (LRU cache with hash map)
- Counting frequencies
- Database indexing
- Spell checker
- Compiler symbol tables
- Anagram detection
- Two sum, subarray sum problems

**Pro Tips:**
- Use for fast lookups
- Combine with doubly linked list for LRU
- Rehash when load factor > 0.7
- Unordered_map in C++ STL

---

### Hash Set

✅ **Use When:**
- Need to check existence quickly O(1)
- Need unique elements
- Don't care about order
- Fast membership testing

❌ **Avoid When:**
- Need sorted order
- Need duplicates
- Keys not hashable

**Real-world Examples:**
- Remove duplicates
- Check if element exists
- Unique visitor tracking
- Set operations (union, intersection)

---

## 🔴 ADVANCED DATA STRUCTURES

### Graph (Adjacency List)

✅ **Use When:**
- Modeling relationships between entities
- Network problems
- Path finding
- Connected components
- Dependency resolution
- Social networks

❌ **Avoid When:**
- Simple linear/hierarchical data
- No relationships between elements
- Dense graph (use adjacency matrix)

**Real-world Examples:**
- Social network (friends)
- Map navigation (cities and roads)
- Web page links
- Course prerequisites
- Recommendation systems
- Network topology

**Pro Tips:**
- Adjacency list for sparse graphs
- Adjacency matrix for dense graphs
- Use BFS for shortest path (unweighted)
- Use DFS for cycle detection

---

### Trie (Prefix Tree)

✅ **Use When:**
- Prefix-based operations
- Autocomplete functionality
- Spell checker
- IP routing
- Pattern matching
- Dictionary with prefix search

❌ **Avoid When:**
- Simple key-value lookup (use hash)
- No prefix operations needed
- Memory is very limited (pointer overhead)
- Small alphabet not suitable

**Real-world Examples:**
- Search engine autocomplete
- Spell checker
- T9 predictive text
- IP routing tables
- Phone contact search
- Word games (Scrabble, Boggle)

**Pro Tips:**
- Perfect for autocomplete
- Space-efficient for common prefixes
- Fast prefix matching O(m)

---

### Union-Find (Disjoint Set Union)

✅ **Use When:**
- Need to check connectivity
- Kruskal's MST algorithm
- Network connectivity
- Dynamic connectivity problems
- Detecting cycles in undirected graphs
- Social network components

❌ **Avoid When:**
- Need detailed paths (use BFS/DFS)
- Dealing with directed graphs
- Need more than connectivity info

**Real-world Examples:**
- Kruskal's algorithm (MST)
- Network connectivity
- Social circles (connected groups)
- Percolation simulation
- Image segmentation
- Least common ancestor

**Pro Tips:**
- Use path compression
- Use union by rank/size
- Nearly O(1) with both optimizations

---

### Segment Tree

✅ **Use When:**
- Need range queries (sum, min, max)
- Need range updates
- Dynamic array with queries
- Online algorithm needed
- Array elements change frequently

❌ **Avoid When:**
- Static array (use prefix sums)
- Only point queries (use array)
- Memory is limited (4n space)
- Simple aggregation (use prefix sum)

**Real-world Examples:**
- Range sum queries with updates
- Finding minimum in range
- Range updates with queries
- Competitive programming problems
- Real-time data analysis

**Pro Tips:**
- Use for dynamic range queries
- Lazy propagation for range updates
- Simpler than other advanced structures

---

### Fenwick Tree (Binary Indexed Tree)

✅ **Use When:**
- Need prefix sum queries
- Point updates
- Less memory than segment tree
- Simpler implementation than segment tree

❌ **Avoid When:**
- Need range updates (use segment tree)
- Need other operations besides sum (use segment tree)

**Real-world Examples:**
- Cumulative frequency tables
- Range sum queries
- Inversion count
- Competitive programming

---

## 🔄 COMPARISON TABLES

### When You Need...

| Requirement | Best Choice | Alternative |
|-------------|-------------|-------------|
| Fast random access | Array | Vector |
| Fast search | Hash Table | BST |
| Sorted order | BST | Sorted Array |
| Min/Max quickly | Heap | Sorted Array |
| LIFO | Stack | Vector |
| FIFO | Queue | Deque |
| Both-end ops | Deque | Doubly LL |
| Prefix search | Trie | Hash Table |
| Range queries | Segment Tree | Prefix Sum |
| Connectivity | Union-Find | DFS/BFS |
| Relationships | Graph | Matrix |

---

### By Problem Type

| Problem Type | Data Structure | Why |
|--------------|----------------|-----|
| Undo/Redo | Stack + Stack | LIFO operations |
| LRU Cache | Hash Map + Doubly LL | O(1) access + delete |
| Autocomplete | Trie | Prefix matching |
| Task Scheduling | Priority Queue (Heap) | Get highest priority |
| Path Finding | Graph + Queue (BFS) | Level-by-level |
| Expression Eval | Stack | Postfix/Infix conversion |
| Sliding Window | Deque | Both-end operations |
| Two Sum | Hash Map | O(1) lookup |
| Median Finding | Two Heaps | Min + Max heap |
| Range Sum | Segment Tree | Dynamic updates |

---

## 💡 DECISION HEURISTICS

### If you need...

**O(1) operations:**
→ Hash Table (search, insert, delete)
→ Array (access)
→ Stack (top)
→ Queue (front/rear)
→ Heap (min/max)

**O(log n) operations:**
→ BST (search in sorted data)
→ Heap (insert/delete)
→ Segment Tree (range queries)

**Sorted data:**
→ BST (dynamic)
→ Sorted Array (static)

**Order preservation:**
→ Vector (insertion order)
→ Linked List (insertion order)
→ BST (sorted order)

**Memory efficiency:**
→ Array (no overhead)
→ Vector (small overhead)
→ Hash Table (moderate overhead)

**Cache-friendly:**
→ Array
→ Vector
→ Segment Tree

---

## 🎯 COMMON MISTAKE PATTERNS

### ❌ Using Wrong DS

**Mistake:** Using ArrayList for frequent front insertions
**Fix:** Use LinkedList or Deque

**Mistake:** Using LinkedList when need random access
**Fix:** Use ArrayList/Vector

**Mistake:** Using Array for dynamic size
**Fix:** Use Vector/ArrayList

**Mistake:** Using BST for unordered fast lookup
**Fix:** Use Hash Table

**Mistake:** Using Hash Table when need sorted iteration
**Fix:** Use BST (TreeMap)

---

## 📊 SPACE-TIME TRADEOFF GUIDE

### More Space for Better Time:
- Hash Table: O(n) space for O(1) operations
- Trie: O(ALPHABET × n × m) space for O(m) prefix search
- Segment Tree: O(4n) space for O(log n) queries
- Memoization: O(states) space for O(1) recomputation

### Less Space, Slower Time:
- Array for sorted data: O(n) space but O(n) insert
- Binary search in array: O(1) space but O(n) insert
- DFS for connectivity: O(1) space but O(V+E) time each query

---

## 🔍 INTERVIEW PATTERN RECOGNITION

### When interviewer says...

"Need to find elements quickly" → Hash Table  
"Need to maintain sorted order" → BST  
"Need to track min/max" → Heap  
"Need to process in order received" → Queue  
"Need to backtrack" → Stack  
"Need to find in range" → Segment Tree  
"Need to check if connected" → Union-Find  
"Need prefix matching" → Trie  
"Need to find path" → Graph  

---

**Total Pages:** ~10-12 pages when printed  
**Purpose:** Quick decision making during problem-solving  
**Usage:** Reference before implementing solution

---

*Choose the right tool for the job - it makes all the difference!*

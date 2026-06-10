# 🚀 Complete DSA Roadmap for Placements

### C++ Edition: Zero to Hero

> **A comprehensive, battle-tested roadmap combining strategic topic selection, efficient learning patterns, and placement-focused preparation for top tech companies.**

---

## 📋 Overview

This roadmap is designed to take you from **DSA beginner to placement-ready** in 4-5 months. It focuses on:

- ✅ **High-ROI Topics** tested by Google, Amazon, Microsoft, DE Shaw, Media.net
- ✅ **Pattern-Based Learning** instead of random problem-solving  
- ✅ **350-450 Quality Problems** over quantity
- ✅ **C++ STL Mastery** for competitive advantage
- ✅ **Progressive Difficulty** from basics to advanced

### 🎯 Target

| Metric | Goal |
|--------|------|
| **Timeline** | 4-5 months (consistent practice) |
| **Problems** | 350-450 quality problems |
| **Platform** | LeetCode (primary) + Codeforces (optional) |
| **Daily Commitment** | 2-3 hours minimum |
| **Contest Participation** | Weekly (LeetCode contests) |

---

## 📚 Table of Contents

1. [Phase 0: C++ Mastery & Prerequisites](#-phase-0-c-mastery--prerequisites-week-1-2)
2. [Phase 1: Logic Building & Complexity Analysis](#-phase-1-logic-building--complexity-analysis-week-3-5)
3. [Phase 2: Recursion & Backtracking](#-phase-2-recursion--backtracking-week-6-8)
4. [Phase 3: Searching & Sorting](#-phase-3-searching--sorting-week-8-9)
5. [Phase 4: Linear Data Structures](#-phase-4-linear-data-structures-week-10-12)
6. [Phase 5: Non-Linear Data Structures](#-phase-5-non-linear-data-structures-week-13-15)
7. [Phase 6: Graphs](#-phase-6-graphs-week-16-18)
8. [Phase 7: Dynamic Programming](#-phase-7-dynamic-programming-week-19-21)
9. [Phase 8: Advanced Topics](#-phase-8-advanced-topics--greedy-week-22-23)
10. [Study Strategy & Best Practices](#-study-strategy-the-golden-rules)
11. [Timeline & Schedule](#%EF%B8%8F-detailed-timeline-4-5-months)

---

## 🧱 PHASE 0: C++ Mastery & Prerequisites (Week 1-2)

**Goal:** Master the language and tools so you can focus on logic, not syntax.

### Core C++ Fundamentals (Must Master)

- Variables, data types, and operators
- Control structures: `if/else`, loops (`for`, `while`)
- **Pointers & References:** `*` vs `&`, pass-by-value vs pass-by-reference (critical for optimization)
- **Memory Management:** Stack vs Heap, `new`/`delete`, avoiding memory leaks
- **Structs & Classes:** Constructors, `this` pointer, operator overloading
- Arrays and strings (basic manipulation)

### The Standard Template Library (STL) - Your Best Friend

**Master these before proceeding:**

#### Containers
- **`vector`:** `push_back`, `resize`, `reserve`, internal working (dynamic resizing)
- **`pair` & `tuple`:** Storing complex data types
- **`map` vs `unordered_map`:** 
  - `map` → Red-Black Tree → O(log n)
  - `unordered_map` → Hash Table → O(1) average
- **`set` vs `unordered_set`:** Similar complexity differences

#### Iterators
- `begin()`, `end()`, `rbegin()`, `rend()`
- Iterator arithmetic and reverse iteration

#### Essential Algorithms
```cpp
// Sorting with custom comparator
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// Searching
binary_search(v.begin(), v.end(), target);
lower_bound(v.begin(), v.end(), target);  // First element >= target
upper_bound(v.begin(), v.end(), target);  // First element > target

// Other utilities
reverse(v.begin(), v.end());
accumulate(v.begin(), v.end(), 0);
max_element(v.begin(), v.end());
```

### ✅ Checkpoint
**Can you implement a custom comparator for sorting `vector<pair<int, int>>`?**  
If not, stay in Phase 0.

**Resources:**
- [C++ STL Documentation](https://en.cppreference.com/)
- [STL Cheat Sheet](https://github.com/gibsjose/cpp-cheat-sheet)

---

## 🟢 PHASE 1: Logic Building & Complexity Analysis (Week 3-5)

**Goal:** Learn to think in terms of Time and Space Complexity. Build problem-solving intuition.

### 1. Time & Space Complexity

- **Big O Notation:** O(1), O(log n), O(n), O(n log n), O(n²), O(2ⁿ)
- **Best, Average, Worst case** analysis
- **Constraint-based thinking:**
  - N = 10⁵ → O(N log N) required
  - N = 10³ → O(N²) acceptable
  - N = 20 → O(2ⁿ) backtracking possible
- **Space complexity:** 10⁸ operations ≈ 1 second
- **Master Theorem basics** for recursive complexity

### 2. Basic Math & Number Theory

```cpp
// GCD using Euclidean algorithm
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// Fast exponentiation
long long power(long long x, long long n) {
    if (n == 0) return 1;
    long long half = power(x, n / 2);
    if (n % 2 == 0) return half * half;
    return x * half * half;
}
```

**Topics:**
- Modulo arithmetic: `(a + b) % m`
- Prime numbers: Sieve of Eratosthenes
- GCD/LCM using Euclidean algorithm
- Fast exponentiation

### 3. Arrays & Vectors (Advanced Patterns)

> **Foundation of everything. Master these patterns.**

#### Key Techniques

**a) Prefix Sum**
```cpp
// 1D Prefix Sum
vector<int> prefix(n + 1, 0);
for (int i = 1; i <= n; i++) {
    prefix[i] = prefix[i-1] + arr[i-1];
}
// Range sum [l, r] = prefix[r+1] - prefix[l]
```

**b) Two Pointers**
```cpp
// Opposite ends pattern (2Sum)
int left = 0, right = n - 1;
while (left < right) {
    int sum = arr[left] + arr[right];
    if (sum == target) return {left, right};
    else if (sum < target) left++;
    else right--;
}

// Slow-fast pattern (Remove duplicates)
int slow = 0;
for (int fast = 1; fast < n; fast++) {
    if (arr[fast] != arr[slow]) {
        arr[++slow] = arr[fast];
    }
}
```

**c) Sliding Window**
```cpp
// Fixed size window
int maxSum = 0, windowSum = 0;
for (int i = 0; i < k; i++) windowSum += arr[i];
maxSum = windowSum;

for (int i = k; i < n; i++) {
    windowSum += arr[i] - arr[i-k];
    maxSum = max(maxSum, windowSum);
}

// Variable size window (Longest substring without repeating)
unordered_map<char, int> freq;
int left = 0, maxLen = 0;
for (int right = 0; right < s.length(); right++) {
    freq[s[right]]++;
    while (freq[s[right]] > 1) {
        freq[s[left]]--;
        left++;
    }
    maxLen = max(maxLen, right - left + 1);
}
```

#### Must-Know Algorithms

**Kadane's Algorithm** (Maximum Subarray Sum)
```cpp
int maxSum = arr[0], currSum = arr[0];
for (int i = 1; i < n; i++) {
    currSum = max(arr[i], currSum + arr[i]);
    maxSum = max(maxSum, currSum);
}
```

**Dutch National Flag** (Sort 0s, 1s, 2s)
```cpp
int low = 0, mid = 0, high = n - 1;
while (mid <= high) {
    if (arr[mid] == 0) swap(arr[low++], arr[mid++]);
    else if (arr[mid] == 1) mid++;
    else swap(arr[mid], arr[high--]);
}
```

**Moore's Voting Algorithm** (Majority Element)
```cpp
int candidate = 0, count = 0;
for (int num : arr) {
    if (count == 0) candidate = num;
    count += (num == candidate) ? 1 : -1;
}
```

#### Practice Problems

| Difficulty | Problem | Pattern |
|-----------|---------|---------|
| Easy | Reverse Array | Basic manipulation |
| Easy | Find Max/Min | Linear search |
| Medium | Container With Most Water | Two pointers |
| Medium | Trapping Rainwater | Multiple approaches |
| Medium | Longest Substring Without Repeating | Sliding window |

### 4. Strings

- C-style strings vs `std::string`
- **Palindrome check** (two pointers)
- **Anagrams** (frequency counting / sorting)
- **Frequency count** using maps
- **Substrings** and subsequences
- **Reverse words** in a string
- **Pattern Matching:**
  - KMP Algorithm (advanced but high value)
  - Rabin-Karp (Rolling Hash technique)

### 5. Bit Manipulation (The "Secret" Weapon)

```cpp
// Check if power of 2
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Count set bits
int countBits(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);  // Remove rightmost set bit
        count++;
    }
    return count;
}

// Set/Unset i-th bit
int setBit(int n, int i) { return n | (1 << i); }
int unsetBit(int n, int i) { return n & ~(1 << i); }
int toggleBit(int n, int i) { return n ^ (1 << i); }

// Check if i-th bit is set
bool isSet(int n, int i) { return (n & (1 << i)) != 0; }
```

**Key Problems:**
- Single Number (XOR trick)
- Subsets using Bitmasking (2ⁿ subsets)
- Power Set generation

---

## 🟡 PHASE 2: Recursion & Backtracking (Week 6-8)

**Goal:** Master the "call stack" mindset. Build foundation for Trees, Graphs, and DP.

> ⚠️ **Very important for trees, DP, and backtracking.**

### 1. Recursion

**The Trust Factor:** Believe your function returns the right answer for `n-1`

**Core Concepts:**
- Base case identification (when to stop)
- Recursive case (how to break down)
- **Recursion Tree Visualization** (draw it to calculate complexity)
- Stack memory understanding (why stack overflow happens)

#### Practice Problems

```cpp
// Factorial
int factorial(int n) {
    if (n <= 1) return 1;  // Base case
    return n * factorial(n - 1);  // Recursive case
}

// Fibonacci
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);  // T(n) = O(2^n) without memoization
}

// Reverse array
void reverse(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    swap(arr[l], arr[r]);
    reverse(arr, l + 1, r - 1);
}

// Generate all subsets
void generateSubsets(vector<int>& arr, int i, vector<int>& curr, vector<vector<int>>& result) {
    if (i == arr.size()) {
        result.push_back(curr);
        return;
    }
    // Exclude current element
    generateSubsets(arr, i + 1, curr, result);
    // Include current element
    curr.push_back(arr[i]);
    generateSubsets(arr, i + 1, curr, result);
    curr.pop_back();
}
```

**Additional Problems:**
- Tower of Hanoi
- Power(x, n)
- Print all permutations
- Josephus Problem

### 2. Backtracking (Controlled Recursion)

**Core Pattern:** `Do → Recurse → Undo`

**State space tree pruning** to avoid unnecessary exploration

#### Essential Problems

**Generate Parentheses**
```cpp
void generate(int open, int close, string curr, vector<string>& result) {
    if (open == 0 && close == 0) {
        result.push_back(curr);
        return;
    }
    if (open > 0) generate(open - 1, close, curr + '(', result);
    if (close > open) generate(open, close - 1, curr + ')', result);
}
```

**N-Queens**
```cpp
bool isSafe(vector<string>& board, int row, int col, int n) {
    // Check column
    for (int i = 0; i < row; i++)
        if (board[i][col] == 'Q') return false;
    
    // Check diagonals
    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j] == 'Q') return false;
    
    for (int i = row, j = col; i >= 0 && j < n; i--, j++)
        if (board[i][j] == 'Q') return false;
    
    return true;
}

void solve(int row, int n, vector<string>& board, vector<vector<string>>& result) {
    if (row == n) {
        result.push_back(board);
        return;
    }
    for (int col = 0; col < n; col++) {
        if (isSafe(board, row, col, n)) {
            board[row][col] = 'Q';
            solve(row + 1, n, board, result);
            board[row][col] = '.';  // Backtrack
        }
    }
}
```

**Practice List:**
- Subsets (distinct and duplicate elements)
- Permutations (distinct and duplicate elements)
- Combination Sum
- Sudoku Solver
- Rat in a Maze
- Word Search
- Palindrome Partitioning

---

## 🟠 PHASE 3: Searching & Sorting (Week 8-9)

**Goal:** Master the art of reducing search space.

### 1. Binary Search (HIGH ROI Topic)

> 🔥 **One of the most powerful tools in competitive programming**

#### Classic Binary Search

```cpp
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;  // Avoid overflow
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// First occurrence
int firstOccurrence(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1, result = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left
        } else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return result;
}
```

#### Binary Search on Answer (IMPORTANT)

**Pattern:** When answer lies in a range and we can check if a value is valid

**Book Allocation Problem**
```cpp
bool canAllocate(vector<int>& books, int students, int maxPages) {
    int count = 1, pages = 0;
    for (int book : books) {
        if (book > maxPages) return false;
        if (pages + book > maxPages) {
            count++;
            pages = book;
        } else {
            pages += book;
        }
    }
    return count <= students;
}

int allocateBooks(vector<int>& books, int students) {
    int left = *max_element(books.begin(), books.end());
    int right = accumulate(books.begin(), books.end(), 0);
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (canAllocate(books, students, mid)) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}
```

**Practice Problems:**
- Search in Rotated Sorted Array
- Find Peak Element
- Aggressive Cows
- Painter's Partition
- Koko Eating Bananas
- Minimum Days to Make Bouquets

#### STL Binary Search
```cpp
// Returns iterator to first element >= target
auto it = lower_bound(v.begin(), v.end(), target);

// Returns iterator to first element > target
auto it = upper_bound(v.begin(), v.end(), target);

// Count occurrences in sorted array
int count = upper_bound(v.begin(), v.end(), target) - lower_bound(v.begin(), v.end(), target);
```

### 2. Sorting Algorithms

**Understanding the mechanics:**

| Algorithm | Time Complexity | Space | Stable? | When to Use |
|-----------|----------------|-------|---------|-------------|
| Merge Sort | O(n log n) | O(n) | ✅ Yes | Linked Lists, External sorting |
| Quick Sort | O(n log n) avg, O(n²) worst | O(log n) | ❌ No | Arrays, in-place needed |
| Heap Sort | O(n log n) | O(1) | ❌ No | Space-constrained |

**Merge Sort (Divide & Conquer)**
```cpp
void merge(vector<int>& arr, int l, int mid, int r) {
    vector<int> temp;
    int i = l, j = mid + 1;
    
    while (i <= mid && j <= r) {
        if (arr[i] <= arr[j]) temp.push_back(arr[i++]);
        else temp.push_back(arr[j++]);
    }
    while (i <= mid) temp.push_back(arr[i++]);
    while (j <= r) temp.push_back(arr[j++]);
    
    for (int k = 0; k < temp.size(); k++) {
        arr[l + k] = temp[k];
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int mid = l + (r - l) / 2;
    mergeSort(arr, l, mid);
    mergeSort(arr, mid + 1, r);
    merge(arr, l, mid, r);
}
```

**Custom Comparators**
```cpp
// Sort pairs by second element descending
sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) {
    return a.second > b.second;
});

// Sort by multiple criteria
sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
    if (a[0] != b[0]) return a[0] < b[0];  // Sort by start
    return a[1] < b[1];  // Then by end
});
```

---

## 🔵 PHASE 4: Linear Data Structures (Week 10-12)

**Goal:** Master memory layout and pointer manipulation.

### 1. Linked Lists

> **Teaches pointer manipulation - crucial foundation**

#### Implementation

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
```

#### Key Techniques

**Tortoise & Hare (Slow & Fast Pointer)**
```cpp
// Find middle of linked list
ListNode* findMiddle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

// Detect cycle (Floyd's Algorithm)
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

**Reversing Linked List**
```cpp
// Iterative
ListNode* reverse(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// Recursive
ListNode* reverseRecursive(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverseRecursive(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

#### Essential Problems

| Problem | Pattern/Technique |
|---------|-------------------|
| Reverse Linked List | Iterative & Recursive |
| Middle of Linked List | Slow-Fast pointer |
| Merge Two Sorted Lists | Two pointers |
| Remove Nth Node From End | Two pointers with gap |
| Detect & Remove Cycle | Floyd's Algorithm |
| Intersection of Two Linked Lists | Two pointers |
| Palindrome Linked List | Reverse second half |
| Reverse Nodes in k-Group | Advanced reversal |
| Flatten Linked List | Merge sorted lists |
| Clone List with Random Pointer | HashMap + traversal |
| **LRU Cache** | Doubly LL + HashMap (Very Important) |

**LRU Cache Implementation**
```cpp
class LRUCache {
    struct Node {
        int key, value;
        Node *prev, *next;
        Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
    };
    
    int capacity;
    unordered_map<int, Node*> cache;
    Node *head, *tail;
    
    void addNode(Node* node) {
        node->next = head->next;
        node->prev = head;
        head->next->prev = node;
        head->next = node;
    }
    
    void removeNode(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    
public:
    LRUCache(int capacity) : capacity(capacity) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (cache.find(key) == cache.end()) return -1;
        Node* node = cache[key];
        removeNode(node);
        addNode(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            removeNode(cache[key]);
        }
        Node* node = new Node(key, value);
        addNode(node);
        cache[key] = node;
        if (cache.size() > capacity) {
            Node* lru = tail->prev;
            removeNode(lru);
            cache.erase(lru->key);
        }
    }
};
```

### 2. Stacks

#### Implementation
```cpp
// Using array
class Stack {
    vector<int> arr;
public:
    void push(int x) { arr.push_back(x); }
    void pop() { arr.pop_back(); }
    int top() { return arr.back(); }
    bool empty() { return arr.empty(); }
};

// Using STL
stack<int> st;
```

#### Applications

**Valid Parentheses**
```cpp
bool isValid(string s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '{' || c == '[') {
            st.push(c);
        } else {
            if (st.empty()) return false;
            char top = st.top();
            if ((c == ')' && top != '(') ||
                (c == '}' && top != '{') ||
                (c == ']' && top != '[')) return false;
            st.pop();
        }
    }
    return st.empty();
}
```

**Monotonic Stack (Very Important)**

Pattern: Find next/previous greater/smaller element

```cpp
// Next Greater Element
vector<int> nextGreaterElement(vector<int>& arr) {
    int n = arr.size();
    vector<int> result(n, -1);
    stack<int> st;
    
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && st.top() <= arr[i]) {
            st.pop();
        }
        if (!st.empty()) result[i] = st.top();
        st.push(arr[i]);
    }
    return result;
}

// Largest Rectangle in Histogram
int largestRectangle(vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    heights.push_back(0);  // Sentinel
    
    for (int i = 0; i < heights.size(); i++) {
        while (!st.empty() && heights[st.top()] > heights[i]) {
            int h = heights[st.top()];
            st.pop();
            int w = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        st.push(i);
    }
    return maxArea;
}
```

**Other Stack Problems:**
- Min Stack (track minimum in O(1))
- Infix to Postfix conversion
- Evaluate Postfix expression
- Trapping Rainwater (Stack solution)

### 3. Queues

> **Used heavily in BFS and tree level-order traversal**

#### Types & Implementation

```cpp
// Simple Queue (STL)
queue<int> q;
q.push(x);
q.pop();
q.front();

// Deque (Double-ended queue)
deque<int> dq;
dq.push_front(x);
dq.push_back(x);
dq.pop_front();
dq.pop_back();

// Priority Queue (Max heap by default)
priority_queue<int> pq;  // Max heap
priority_queue<int, vector<int>, greater<int>> minPq;  // Min heap
```

**Sliding Window Maximum (Deque)**
```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq;  // Store indices
    vector<int> result;
    
    for (int i = 0; i < nums.size(); i++) {
        // Remove out of window elements
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        // Remove smaller elements
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
        
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    return result;
}
```

**Circular Queue Implementation**
```cpp
class CircularQueue {
    vector<int> arr;
    int front, rear, size, capacity;
public:
    CircularQueue(int k) : capacity(k), front(0), rear(-1), size(0) {
        arr.resize(k);
    }
    
    bool enqueue(int value) {
        if (size == capacity) return false;
        rear = (rear + 1) % capacity;
        arr[rear] = value;
        size++;
        return true;
    }
    
    bool dequeue() {
        if (size == 0) return false;
        front = (front + 1) % capacity;
        size--;
        return true;
    }
};
```

**Practice Problems:**
- Implement Stack using Queues
- Implement Queue using Stacks
- Design Circular Queue

---

## 🟣 PHASE 5: Non-Linear Data Structures (Week 13-15)

**Goal:** Master hierarchical data. This is where DSA gets interesting and interview-heavy.

> 🔥 **Most common interview topic - extremely important**

### 1. Trees & Binary Trees

#### Tree Node Definition

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

#### Tree Traversals

**Recursive Traversals**
```cpp
// Inorder (Left → Root → Right)
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << " ";
    inorder(root->right);
}

// Preorder (Root → Left → Right)
void preorder(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";
    preorder(root->left);
    preorder(root->right);
}

// Postorder (Left → Right → Root)
void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val << " ";
}
```

**Iterative Traversals**
```cpp
// Inorder iterative
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while (curr || !st.empty()) {
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }
    return result;
}

// Level Order (BFS)
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

#### Tree Views

```cpp
// Right view (last node at each level)
vector<int> rightView(TreeNode* root) {
    if (!root) return {};
    vector<int> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            if (i == size - 1) result.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return result;
}
```

#### Key Problems

**Height & Diameter**
```cpp
int height(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(height(root->left), height(root->right));
}

int diameter(TreeNode* root, int& maxDiam) {
    if (!root) return 0;
    int left = diameter(root->left, maxDiam);
    int right = diameter(root->right, maxDiam);
    maxDiam = max(maxDiam, left + right);
    return 1 + max(left, right);
}
```

**Lowest Common Ancestor (LCA)**
```cpp
TreeNode* lca(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    TreeNode* left = lca(root->left, p, q);
    TreeNode* right = lca(root->right, p, q);
    if (left && right) return root;
    return left ? left : right;
}
```

**Maximum Path Sum**
```cpp
int maxPathSum(TreeNode* root, int& maxSum) {
    if (!root) return 0;
    int left = max(0, maxPathSum(root->left, maxSum));
    int right = max(0, maxPathSum(root->right, maxSum));
    maxSum = max(maxSum, left + right + root->val);
    return max(left, right) + root->val;
}
```

**Practice List:**
- Top/Bottom/Left/Right Views
- Vertical Order Traversal
- Boundary Traversal
- Serialize and Deserialize Binary Tree
- Construct Tree from Inorder & Preorder
- Burn a Tree (time to burn all nodes)

### 2. Binary Search Trees (BST)

> **Key Property:** Inorder traversal of BST is always sorted

#### BST Operations

```cpp
// Search in BST
TreeNode* search(TreeNode* root, int val) {
    if (!root || root->val == val) return root;
    if (val < root->val) return search(root->left, val);
    return search(root->right, val);
}

// Insert in BST
TreeNode* insert(TreeNode* root, int val) {
    if (!root) return new TreeNode(val);
    if (val < root->val) root->left = insert(root->left, val);
    else root->right = insert(root->right, val);
    return root;
}

// Delete in BST
TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;
    
    if (key < root->val) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->val) {
        root->right = deleteNode(root->right, key);
    } else {
        // Node with one child or no child
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        
        // Node with two children
        TreeNode* minNode = findMin(root->right);
        root->val = minNode->val;
        root->right = deleteNode(root->right, minNode->val);
    }
    return root;
}

TreeNode* findMin(TreeNode* root) {
    while (root->left) root = root->left;
    return root;
}
```

**Validate BST**
```cpp
bool isValidBST(TreeNode* root, long minVal, long maxVal) {
    if (!root) return true;
    if (root->val <= minVal || root->val >= maxVal) return false;
    return isValidBST(root->left, minVal, root->val) && 
           isValidBST(root->right, root->val, maxVal);
}
```

**LCA in BST** (Optimized using BST property)
```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (p->val < root->val && q->val < root->val)
        return lowestCommonAncestor(root->left, p, q);
    if (p->val > root->val && q->val > root->val)
        return lowestCommonAncestor(root->right, p, q);
    return root;
}
```

### 3. Heaps (Priority Queue)

> **Turns O(n²) or O(n log n) problems into O(n log k)**

#### Heap Concepts

- **Min Heap:** Parent ≤ Children
- **Max Heap:** Parent ≥ Children
- **Array Representation:** For node at index `i`:
  - Left child: `2*i + 1`
  - Right child: `2*i + 2`
  - Parent: `(i-1)/2`

#### STL Priority Queue

```cpp
// Max heap (default)
priority_queue<int> maxHeap;

// Min heap
priority_queue<int, vector<int>, greater<int>> minHeap;

// Custom comparator
auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
    return a.second > b.second;  // Min heap based on second element
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
```

#### Pattern Problems

**K-th Largest Element**
```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}
```

**Merge K Sorted Lists**
```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    
    for (auto list : lists) {
        if (list) pq.push(list);
    }
    
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (!pq.empty()) {
        curr->next = pq.top();
        pq.pop();
        curr = curr->next;
        if (curr->next) pq.push(curr->next);
    }
    return dummy->next;
}
```

**Median from Data Stream**
```cpp
class MedianFinder {
    priority_queue<int> maxHeap;  // Left half
    priority_queue<int, vector<int>, greater<int>> minHeap;  // Right half
public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();
        
        if (maxHeap.size() < minHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        if (maxHeap.size() > minHeap.size()) return maxHeap.top();
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};
```

**More Problems:**
- Top K Frequent Elements
- K Closest Points to Origin
- Reorganize String

### 4. Hashing

> 🔥 **Turns O(n²) into O(n) - game changer**

#### Hash Table Concepts

**Collision Handling:**
- **Chaining:** Linked list at each bucket
- **Open Addressing:** Linear probing, quadratic probing

**STL Maps:**
```cpp
// Hash Table - O(1) average
unordered_map<int, int> hashMap;
unordered_set<int> hashSet;

// Balanced BST - O(log n)
map<int, int> orderedMap;
set<int> orderedSet;
```

#### Common Patterns

**Two Sum**
```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}
```

**Frequency Counting**
```cpp
// Most frequent element
int majorityElement(vector<int>& nums) {
    unordered_map<int, int> freq;
    for (int num : nums) {
        if (++freq[num] > nums.size() / 2) return num;
    }
    return -1;
}
```

**Rolling Hash (Rabin-Karp)**
```cpp
class RollingHash {
    const int MOD = 1e9 + 7;
    const int BASE = 31;
    
public:
    long long computeHash(string s) {
        long long hash = 0, power = 1;
        for (char c : s) {
            hash = (hash + (c - 'a' + 1) * power) % MOD;
            power = (power * BASE) % MOD;
        }
        return hash;
    }
};
```

**Practice Problems:**
- Group Anagrams
- Subarray Sum Equals K
- Longest Consecutive Sequence
- First Unique Character
- Valid Sudoku

---

## 🔴 PHASE 6: Graphs (Week 16-18)

**Goal:** Master complex relationships and pathfinding. Critical for top product companies.

> ⚠️ **Study only after mastering trees. Very important for real-world problems.**

### 1. Graph Representation

```cpp
// Adjacency List (Preferred)
vector<vector<int>> adjList(n);
adjList[u].push_back(v);

// For weighted graphs
vector<vector<pair<int, int>>> graph(n);  // {neighbor, weight}
graph[u].push_back({v, weight});

// Adjacency Matrix (for dense graphs)
vector<vector<int>> adjMatrix(n, vector<int>(n, 0));
adjMatrix[u][v] = 1;
```

### 2. Graph Traversal

**BFS (Breadth-First Search)**
```cpp
void bfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;
    q.push(start);
    visited[start] = true;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";
        
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

**DFS (Depth-First Search)**
```cpp
void dfs(vector<vector<int>>& graph, int node, vector<bool>& visited) {
    visited[node] = true;
    cout << node << " ";
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs(graph, neighbor, visited);
        }
    }
}
```

#### Basic Graph Problems

**Number of Islands**
```cpp
void dfs(vector<vector<char>>& grid, int i, int j) {
    if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] == '0')
        return;
    grid[i][j] = '0';  // Mark as visited
    dfs(grid, i+1, j);
    dfs(grid, i-1, j);
    dfs(grid, i, j+1);
    dfs(grid, i, j-1);
}

int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[0].size(); j++) {
            if (grid[i][j] == '1') {
                count++;
                dfs(grid, i, j);
            }
        }
    }
    return count;
}
```

**Rotten Oranges (Multi-source BFS)**
```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    queue<pair<int, int>> q;
    int fresh = 0;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 2) q.push({i, j});
            else if (grid[i][j] == 1) fresh++;
        }
    }
    
    if (fresh == 0) return 0;
    
    int minutes = 0;
    int dirs[4][2] = {{0,1}, {0,-1}, {1,0}, {-1,0}};
    
    while (!q.empty()) {
        int size = q.size();
        bool rotted = false;
        for (int i = 0; i < size; i++) {
            auto [x, y] = q.front();
            q.pop();
            for (auto& dir : dirs) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    grid[nx][ny] = 2;
                    q.push({nx, ny});
                    fresh--;
                    rotted = true;
                }
            }
        }
        if (rotted) minutes++;
    }
    return fresh == 0 ? minutes : -1;
}
```

### 3. Cycle Detection

**Undirected Graph (using DFS)**
```cpp
bool hasCycleDFS(vector<vector<int>>& graph, int node, int parent, vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            if (hasCycleDFS(graph, neighbor, node, visited)) return true;
        } else if (neighbor != parent) {
            return true;  // Back edge found
        }
    }
    return false;
}
```

**Directed Graph (using DFS with recursion stack)**
```cpp
bool hasCycleDFS(vector<vector<int>>& graph, int node, 
                 vector<bool>& visited, vector<bool>& recStack) {
    visited[node] = true;
    recStack[node] = true;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            if (hasCycleDFS(graph, neighbor, visited, recStack)) return true;
        } else if (recStack[neighbor]) {
            return true;  // Back edge to node in current path
        }
    }
    
    recStack[node] = false;
    return false;
}
```

### 4. Topological Sort

**Kahn's Algorithm (BFS)**
```cpp
vector<int> topologicalSort(vector<vector<int>>& graph, int n) {
    vector<int> indegree(n, 0);
    for (int i = 0; i < n; i++) {
        for (int neighbor : graph[i]) {
            indegree[neighbor]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) q.push(i);
    }
    
    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : graph[node]) {
            if (--indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    return result.size() == n ? result : vector<int>();  // Empty if cycle
}
```

**DFS Method**
```cpp
void topologicalSortDFS(vector<vector<int>>& graph, int node,
                        vector<bool>& visited, stack<int>& st) {
    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            topologicalSortDFS(graph, neighbor, visited, st);
        }
    }
    st.push(node);
}
```

**Problems:**
- Course Schedule I & II
- Alien Dictionary

### 5. Shortest Path Algorithms

**Dijkstra's Algorithm** (Non-negative weights)
```cpp
vector<int> dijkstra(vector<vector<pair<int,int>>>& graph, int src, int n) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    
    dist[src] = 0;
    pq.push({0, src});  // {distance, node}
    
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

**Bellman-Ford** (Handles negative weights)
```cpp
vector<int> bellmanFord(vector<vector<int>>& edges, int n, int src) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    
    // Relax all edges n-1 times
    for (int i = 0; i < n - 1; i++) {
        for (auto& edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    
    // Check for negative cycle
    for (auto& edge : edges) {
        int u = edge[0], v = edge[1], w = edge[2];
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            return {};  // Negative cycle detected
        }
    }
    return dist;
}
```

**Floyd-Warshall** (All-pairs shortest path)
```cpp
vector<vector<int>> floydWarshall(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<vector<int>> dist = graph;
    
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    return dist;
}
```

**Problems:**
- Network Delay Time
- Cheapest Flights within K Stops
- Path with Maximum Probability

### 6. Minimum Spanning Tree (MST)

**Prim's Algorithm**
```cpp
int primMST(vector<vector<pair<int,int>>>& graph, int n) {
    vector<bool> inMST(n, false);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    
    int mstCost = 0;
    pq.push({0, 0});  // {weight, node}
    
    while (!pq.empty()) {
        auto [w, u] = pq.top();
        pq.pop();
        
        if (inMST[u]) continue;
        inMST[u] = true;
        mstCost += w;
        
        for (auto [v, weight] : graph[u]) {
            if (!inMST[v]) {
                pq.push({weight, v});
            }
        }
    }
    return mstCost;
}
```

**Kruskal's Algorithm** (uses Union-Find)
```cpp
class DSU {
    vector<int> parent, rank;
public:
    DSU(int n) : parent(n), rank(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);  // Path compression
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }
};

int kruskalMST(vector<vector<int>>& edges, int n) {
    sort(edges.begin(), edges.end(), [](auto& a, auto& b) {
        return a[2] < b[2];  // Sort by weight
    });
    
    DSU dsu(n);
    int mstCost = 0;
    
    for (auto& edge : edges) {
        int u = edge[0], v = edge[1], w = edge[2];
        if (dsu.unite(u, v)) {
            mstCost += w;
        }
    }
    return mstCost;
}
```

### 7. Disjoint Set Union (Union-Find)

```cpp
class UnionFind {
    vector<int> parent, size;
public:
    UnionFind(int n) : parent(n), size(n, 1) {
        iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        // Union by size
        if (size[px] < size[py]) swap(px, py);
        parent[py] = px;
        size[px] += size[py];
        return true;
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

**Problems:**
- Number of Provinces
- Accounts Merge
- Redundant Connection

---

## ⚫ PHASE 7: Dynamic Programming (Week 19-21)

**Goal:** Solve the "unsolvable". The topic that separates "Good" from "Great".

> 🔥 **THIS DECIDES YOUR PLACEMENT LEVEL. Master this thoroughly.**

### Core Concept

**Formula:**
- **Recursion + Memoization = Top Down DP**
- **Iteration + Table = Bottom Up DP**

**Standard Approach:**
1. Start with recursive solution
2. Add memoization (Top-Down DP)
3. Convert to tabulation (Bottom-Up DP)
4. Optimize space complexity

### Pattern 1: 1D DP (Foundation)

**Fibonacci**
```cpp
// Recursive (TLE)
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);  // O(2^n)
}

// Memoization (Top-Down)
int fibMemo(int n, vector<int>& dp) {
    if (n <= 1) return n;
    if (dp[n] != -1) return dp[n];
    return dp[n] = fibMemo(n-1, dp) + fibMemo(n-2, dp);  // O(n)
}

// Tabulation (Bottom-Up)
int fibTab(int n) {
    if (n <= 1) return n;
    vector<int> dp(n+1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// Space Optimized
int fibOptimized(int n) {
    if (n <= 1) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Climbing Stairs**
```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**House Robber**
```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 1) return nums[0];
    int prev2 = nums[0], prev1 = max(nums[0], nums[1]);
    
    for (int i = 2; i < n; i++) {
        int curr = max(prev1, nums[i] + prev2);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

### Pattern 2: 2D DP (Grids)

**Unique Paths**
```cpp
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}

// Space Optimized
int uniquePathsOptimized(int m, int n) {
    vector<int> dp(n, 1);
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j-1];
        }
    }
    return dp[n-1];
}
```

**Min Path Sum**
```cpp
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<int>> dp(m, vector<int>(n));
    dp[0][0] = grid[0][0];
    
    for (int i = 1; i < m; i++) dp[i][0] = dp[i-1][0] + grid[i][0];
    for (int j = 1; j < n; j++) dp[0][j] = dp[0][j-1] + grid[0][j];
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m-1][n-1];
}
```

### Pattern 3: Knapsack (Very Important)

**0/1 Knapsack**
```cpp
int knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n+1, vector<int>(W+1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i-1][w], 
                              values[i-1] + dp[i-1][w - weights[i-1]]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][W];
}
```

**Subset Sum**
```cpp
bool subsetSum(vector<int>& nums, int target) {
    int n = nums.size();
    vector<vector<bool>> dp(n+1, vector<bool>(target+1, false));
    
    for (int i = 0; i <= n; i++) dp[i][0] = true;
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= target; j++) {
            dp[i][j] = dp[i-1][j];  // Exclude
            if (nums[i-1] <= j) {
                dp[i][j] = dp[i][j] || dp[i-1][j - nums[i-1]];  // Include
            }
        }
    }
    return dp[n][target];
}
```

**Coin Change**
```cpp
// Minimum coins
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INT_MAX) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}

// Number of ways (Unbounded Knapsack)
int coinChangeWays(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}
```

### Pattern 4: String DP

**Longest Common Subsequence (LCS)**
```cpp
int longestCommonSubsequence(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

**Edit Distance**
```cpp
int editDistance(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m+1, vector<int>(n+1));
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],    // Delete
                                    dp[i][j-1],    // Insert
                                    dp[i-1][j-1]}); // Replace
            }
        }
    }
    return dp[m][n];
}
```

**Longest Palindromic Subsequence**
```cpp
int longestPalindromeSubseq(string s) {
    string rev = s;
    reverse(rev.begin(), rev.end());
    return longestCommonSubsequence(s, rev);
}
```

### Pattern 5: Longest Increasing Subsequence (LIS)

**O(n²) Approach**
```cpp
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    return *max_element(dp.begin(), dp.end());
}
```

**O(n log n) Approach (Binary Search)**
```cpp
int lengthOfLIS(vector<int>& nums) {
    vector<int> lis;
    for (int num : nums) {
        auto it = lower_bound(lis.begin(), lis.end(), num);
        if (it == lis.end()) {
            lis.push_back(num);
        } else {
            *it = num;
        }
    }
    return lis.size();
}
```

### Pattern 6: DP on Stocks

**Best Time to Buy and Sell Stock (Multiple Variations)**
```cpp
// Can buy/sell multiple times
int maxProfit(vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        profit += max(0, prices[i] - prices[i-1]);
    }
    return profit;
}

// At most 2 transactions
int maxProfitK2(vector<int>& prices) {
    int buy1 = INT_MIN, sell1 = 0;
    int buy2 = INT_MIN, sell2 = 0;
    
    for (int price : prices) {
        buy1 = max(buy1, -price);
        sell1 = max(sell1, buy1 + price);
        buy2 = max(buy2, sell1 - price);
        sell2 = max(sell2, buy2 + price);
    }
    return sell2;
}
```

### Pattern 7: Matrix DP

**Matrix Chain Multiplication**
```cpp
int matrixChainMultiplication(vector<int>& dims) {
    int n = dims.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i] * dims[k+1] * dims[j+1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }
    return dp[0][n-1];
}
```

---

## 🔥 PHASE 8: Advanced Topics & Greedy (Week 22-23)

**Goal:** Complete your toolkit. Do once basics are solid.

### 1. Greedy Algorithms

**Activity Selection**
```cpp
int maxActivities(vector<pair<int,int>>& activities) {
    sort(activities.begin(), activities.end(), [](auto& a, auto& b) {
        return a.second < b.second;  // Sort by end time
    });
    
    int count = 1, lastEnd = activities[0].second;
    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].first >= lastEnd) {
            count++;
            lastEnd = activities[i].second;
        }
    }
    return count;
}
```

**Job Scheduling**
```cpp
int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit) {
    int n = startTime.size();
    vector<tuple<int,int,int>> jobs;
    for (int i = 0; i < n; i++) {
        jobs.push_back({endTime[i], startTime[i], profit[i]});
    }
    sort(jobs.begin(), jobs.end());
    
    map<int, int> dp;  // {time, max profit}
    dp[0] = 0;
    
    for (auto [end, start, prof] : jobs) {
        auto it = dp.upper_bound(start);
        --it;
        int maxProfit = it->second + prof;
        if (maxProfit > dp.rbegin()->second) {
            dp[end] = maxProfit;
        }
    }
    return dp.rbegin()->second;
}
```

**Fractional Knapsack**
```cpp
double fractionalKnapsack(vector<pair<int,int>>& items, int capacity) {
    // {value, weight}
    sort(items.begin(), items.end(), [](auto& a, auto& b) {
        return (double)a.first/a.second > (double)b.first/b.second;
    });
    
    double maxValue = 0;
    for (auto [value, weight] : items) {
        if (capacity >= weight) {
            maxValue += value;
            capacity -= weight;
        } else {
            maxValue += (double)value * capacity / weight;
            break;
        }
    }
    return maxValue;
}
```

### 2. Advanced Trees

**Trie (Prefix Tree)**
```cpp
class Trie {
    struct TrieNode {
        TrieNode* children[26];
        bool isEnd;
        TrieNode() : isEnd(false) {
            fill(children, children + 26, nullptr);
        }
    };
    
    TrieNode* root;
    
public:
    Trie() { root = new TrieNode(); }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return true;
    }
};
```

**Segment Tree** (Range Queries)
```cpp
class SegmentTree {
    vector<int> tree;
    int n;
    
    void build(vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2*node, start, mid);
        build(arr, 2*node+1, mid+1, end);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
    
    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) return tree[node];
        int mid = (start + end) / 2;
        return query(2*node, start, mid, l, r) + 
               query(2*node+1, mid+1, end, l, r);
    }
    
public:
    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n-1);
    }
    
    int rangeSum(int l, int r) {
        return query(1, 0, n-1, l, r);
    }
};
```

**Fenwick Tree (Binary Indexed Tree)**
```cpp
class FenwickTree {
    vector<int> tree;
    int n;
    
public:
    FenwickTree(int size) : n(size) {
        tree.resize(n + 1, 0);
    }
    
    void update(int idx, int delta) {
        idx++;  // 1-indexed
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx);
        }
    }
    
    int query(int idx) {
        idx++;  // 1-indexed
        int sum = 0;
        while (idx > 0) {
            sum += tree[idx];
            idx -= idx & (-idx);
        }
        return sum;
    }
    
    int rangeQuery(int l, int r) {
        return query(r) - (l > 0 ? query(l-1) : 0);
    }
};
```

---

## 🧠 Study Strategy: The Golden Rules

### For EVERY Data Structure, Follow This Pattern

1. **What problem does it solve?** (Understand the motivation)
2. **Structure & working mechanism** (How it's built internally)
3. **Operations & time complexity** (CRUD operations analysis)
4. **Implementation in C++** (Code it from scratch)
5. **Solve 5-10 problems on it** (Pattern recognition)

### Daily Practice Guidelines

| Rule | Description |
|------|-------------|
| **Consistency > Intensity** | 2-3 problems daily beats 20 problems on Sunday |
| **Quality > Quantity** | Understanding 50 problems deeply > solving 200 blindly |
| **Breadth over Depth** | Don't do 50 Easy arrays. Do 10 Easy, 20 Medium, 5 Hard |
| **Always Dry Run** | Trace code on paper before typing |
| **Revisit Problems** | Solve the same problem again after 1 week |
| **Read Others' Solutions** | After solving, check optimized approaches |
| **Time Yourself** | Practice with a 45-minute timer |

### The 45-Minute Rule

- **0-15 min:** Try solving on your own
- **15-30 min:** Look at hints if stuck
- **30-45 min:** Study solution, understand approach
- **After 45 min:** Implement from scratch without looking

### Contest Participation

- **LeetCode Weekly Contests** every weekend
- **Goal:** Solve 3 out of 4 problems consistently
- **Codeforces** (optional): For speed and edge cases
- **Virtual contests:** Practice past contests

### Progress Tracking

Create a spreadsheet with:
- Problem name & link
- Difficulty (Easy/Medium/Hard)
- Topic/Pattern
- First attempt date
- Revisit dates
- Time taken
- Notes on approach

---

## ⏱️ Detailed Timeline (4-5 Months)

### Month 1: Foundation Building
**Weeks 1-2:** C++ & STL Mastery  
**Weeks 3-5:** Arrays, Strings, Complexity Analysis, Bit Manipulation

**Daily Goal:** 2 Easy / 1 Medium problem  
**Focus:** Build strong foundations, master syntax

### Month 2: Core Data Structures
**Weeks 6-8:** Recursion, Backtracking, Binary Search  
**Weeks 9-12:** Linked Lists, Stacks, Queues

**Daily Goal:** 2 Medium problems  
**Focus:** Pointer manipulation, search space reduction

### Month 3: Advanced Structures
**Weeks 13-15:** Trees, BST, Heaps, Hashing  
**Weeks 16-18:** Graphs (BFS, DFS, Shortest Paths)

**Daily Goal:** 1 Medium / 1 Hard problem  
**Focus:** Hierarchical thinking, graph algorithms

### Month 4-5: Mastery & Interview Prep
**Weeks 19-21:** Dynamic Programming (all patterns)  
**Weeks 22-23:** Greedy, Advanced DS, Revision

**Daily Goal:** Contest participation, 1-2 Hard problems  
**Focus:** Pattern recognition, mock interviews

---

## 💡 Final Success Tips

### Critical Habits

✅ **DSA is a muscle** - If you stop for a week, you will regress. Stay consistent.  
✅ **Don't memorize solutions** - Understand the pattern and approach.  
✅ **Learn to fail fast** - If stuck for 45+ mins, look at hints, then retry.  
✅ **Track your progress** - Maintain a spreadsheet of problems solved.  
✅ **Join study groups** - Discuss approaches with peers.  
✅ **Think out loud** - Practice explaining your approach (for interviews).  
✅ **Edge cases matter** - Always think: empty input, single element, all same, sorted/unsorted.

### Common Pitfalls to Avoid

❌ Jumping to hard problems too early  
❌ Not timing yourself during practice  
❌ Skipping the "understanding" phase  
❌ Not reviewing mistakes  
❌ Ignoring space complexity  
❌ Copy-pasting code without understanding  
❌ Not practicing on paper/whiteboard

### Interview Preparation (Last 2-4 Weeks)

1. **Company-specific prep:** Research common patterns at target companies
2. **Mock interviews:** Practice with peers or platforms like Pramp
3. **System Design basics:** Learn fundamentals (for experienced roles)
4. **Behavioral questions:** Prepare STAR method responses
5. **Communication practice:** Explain your thought process clearly

---

## 📊 Progress Checklist

Track your learning with this phase-wise checklist:

- [ ] **Phase 0:** ✅ C++ fundamentals, STL mastery, custom comparators
- [ ] **Phase 1:** ✅ Complexity analysis, array patterns, string algorithms
- [ ] **Phase 2:** ✅ Recursion visualization, backtracking problems (N-Queens, Sudoku)
- [ ] **Phase 3:** ✅ Binary search variations, search on answer, sorting algorithms
- [ ] **Phase 4:** ✅ Linked list manipulation, stack patterns, queue implementations
- [ ] **Phase 5:** ✅ Tree traversals, BST operations, heap problems, hashing patterns
- [ ] **Phase 6:** ✅ Graph traversals, shortest paths, topological sort, MST
- [ ] **Phase 7:** ✅ All DP patterns (1D, 2D, Knapsack, String, LIS, Stocks)
- [ ] **Phase 8:** ✅ Greedy algorithms, Trie, Segment Tree, advanced topics

---

## 🎯 Milestone Goals

| Milestone | Target | Metric |
|-----------|--------|--------|
| **2 months** | Phase 0-4 complete | 150-180 problems solved |
| **3 months** | Phase 0-6 complete | 250-300 problems solved |
| **4 months** | All phases complete | 350-400 problems solved |
| **5 months** | Interview ready | 400-450 problems + contests |

---

## 📚 Recommended Resources

### Platforms
- **LeetCode** (Primary) - Best for interview prep
- **Codeforces** - For competitive programming
- **GeeksforGeeks** - For theory and explanations
- **InterviewBit** - Structured learning path

### YouTube Channels
- Striver (TakeUForward)
- Abdul Bari (Algorithms)
- William Fiset (Graph Theory)
- Tushar Roy (DP patterns)

### Books (Optional)
- "Cracking the Coding Interview" by Gayle Laakmann McDowell
- "Introduction to Algorithms" (CLRS)
- "Competitive Programming" by Steven Halim

---

## 🚀 Final Words

Remember: **This roadmap is your guide, but your dedication and consistency will determine your success.**

- Start today, not tomorrow
- Stay consistent, not intense
- Trust the process
- Don't compare your progress with others
- Celebrate small wins
- Take breaks when needed
- Stay curious and keep learning

**You've got this! Now go ace those interviews! 💪**

---

### Contributing

Found an issue or want to suggest improvements? Feel free to:
- Open an issue
- Submit a pull request
- Share your success story

### License

This roadmap is open-source and free to use. Share it with anyone who might benefit!

---

**Created with ❤️ for CSE students aspiring for top tech placements**

*Last Updated: January 2026*

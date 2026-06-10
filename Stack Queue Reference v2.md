# 📘 DSA Reference Manual — Stack & Queue

**A Complete Reference Guide for C++ Implementations**

*Matches the detail level of the Vector & Linked List sections. Print-friendly.*

---

## 📋 Table of Contents

1. [Stack](#1-stack)
   - [Overview](#overview)
   - [Structure Definition](#structure-definition)
   - [Operations](#operations)
   - [Stack Using Linked List](#stack-using-linked-list)
   - [Applications](#applications-stack)
   - [Complete Implementation](#complete-implementation-stack)
   - [Complexity Summary](#complexity-summary-stack)
   - [Common Use Cases](#common-use-cases-stack)
   - [Key Takeaways](#key-takeaways-stack)

2. [Queue](#2-queue)
   - [Overview](#overview-1)
   - [Structure Definition](#structure-definition-1)
   - [The Problem with a Naive Queue](#the-problem-with-a-naive-queue)
   - [Operations](#operations-1)
   - [Queue Using Linked List](#queue-using-linked-list)
   - [Applications](#applications-queue)
   - [Complete Implementation](#complete-implementation-queue)
   - [Queue Variants Comparison](#queue-variants-comparison)
   - [Complexity Summary](#complexity-summary-queue)
   - [Common Use Cases](#common-use-cases-queue)
   - [Key Takeaways](#key-takeaways-queue)

---
---

# 1. Stack

---

## Overview

**What is it?**
A linear data structure that follows the **LIFO** (Last In, First Out) principle. You can only add or remove elements from one end — the **top**. Think of it as a one-way door.

**Why use it?**
- ✅ O(1) push and pop — extremely fast
- ✅ Models "most recent first" logic naturally
- ✅ Powers function call management, backtracking, expression evaluation
- ✅ Foundation for DFS, compilers, undo systems
- ✅ Simple to implement in both array and linked list form

**Real-world analogy:**
A stack of plates in a cafeteria. You can only place a new plate on top, and you can only take the top plate. You cannot reach the plate at the bottom without removing all the ones above it.

```
        ↑ Only access point (top)
   ┌─────────┐
   │ "plate" │  ← added last, removed first
   ├─────────┤
   │ "plate" │
   ├─────────┤
   │ "plate" │  ← added first, removed last
   └─────────┘
       Bottom (never directly accessed)
```

**LIFO in action:**
```
Push order:   10 → 20 → 30
Pop order:    30 → 20 → 10  (reversed!)

This reversal property is the entire power of a stack.
```

**Visual Representation:**
```
After pushing 10, 20, 30:

         Top
          ↓
     ┌────────┐
     │   30   │  ← push/pop happen only here
     ├────────┤
     │   20   │
     ├────────┤
     │   10   │
     └────────┘
       Bottom

top index = 2 (if 0-indexed array)
size = 3
```

---

## Structure Definition

```cpp
class Stack {
private:
    int* arr;       // Pointer to the underlying array
    int top;        // Index of the topmost element (-1 if empty)
    int capacity;   // Maximum number of elements the stack can hold

public:
    Stack(int size = 100);    // Constructor
    ~Stack();                 // Destructor

    void push(int val);       // Add to top
    void pop();               // Remove from top
    int  peek();              // View top element (no removal)
    bool isEmpty();           // Is stack empty?
    bool isFull();            // Is stack at capacity?
    int  size();              // How many elements?
    void clear();             // Remove all elements
};
```

**Member Variables Explained:**
```
arr      → Points to the heap-allocated array
           Stack stores elements at arr[0], arr[1], ..., arr[top]

top      → Index of the current top element.
           -1 means the stack is empty.
           When we push, top is incremented first.
           When we pop, top is decremented.

capacity → The maximum number of elements the array can hold.
           Array indices go from 0 to capacity-1.
           When top == capacity-1, the stack is full.
```

**How `top` tracks the stack state:**
```
capacity = 5

Empty:      top = -1
             ┌────┬────┬────┬────┬────┐
             │ __ │ __ │ __ │ __ │ __ │
             └────┴────┴────┴────┴────┘
               0    1    2    3    4

After push(10): top = 0
             ┌────┬────┬────┬────┬────┐
             │ 10 │ __ │ __ │ __ │ __ │
             └────┴────┴────┴────┴────┘
               ↑

After push(20): top = 1
             ┌────┬────┬────┬────┬────┐
             │ 10 │ 20 │ __ │ __ │ __ │
             └────┴────┴────┴────┴────┘
                    ↑

After push(30): top = 2
             ┌────┬────┬────┬────┬────┐
             │ 10 │ 20 │ 30 │ __ │ __ │
             └────┴────┴────┴────┴────┘
                         ↑
```

---

## Operations

### Operation 1: Constructor

**Purpose:** Allocate the internal array and initialize the stack to an empty state.

**Code:**
```cpp
Stack(int size = 100) {
    arr = new int[size];   // Allocate array on the heap
    capacity = size;       // Store max capacity
    top = -1;              // -1 signals "empty"
}
```

**Why top = -1 (not 0)?**
```
If we start top = 0, there is an ambiguity:
  - Is the stack empty, or does it have one element at index 0?

Starting at top = -1 removes this ambiguity:
  - top == -1 → definitely empty
  - top == 0  → exactly one element

size() = top + 1
  top = -1 → size = 0  (empty)
  top =  0 → size = 1  (one element)
  top =  n → size = n+1

This math works cleanly only with top starting at -1.
```

**Memory layout after construction:**
```
Stack s(5);  // capacity=5, top=-1

Stack object (on stack frame):
  arr      → ──────────────────────────────────┐
  top      = -1                                │
  capacity = 5                                 ↓
                                       [heap memory]
                                   ┌────┬────┬────┬────┬────┐
                                   │ __ │ __ │ __ │ __ │ __ │
                                   └────┴────┴────┴────┴────┘
                                     0    1    2    3    4
```

**Time Complexity:** O(1)
**Space Complexity:** O(n) where n = capacity

---

### Operation 2: Push (Add to Top)

**Purpose:** Add a new element on top of the stack.

**Code:**
```cpp
void push(int val) {
    // Step 1: Guard against overflow
    if (top >= capacity - 1) {
        throw std::overflow_error("Stack Overflow");
    }

    // Step 2: Increment top FIRST, then store value
    arr[++top] = val;
    // Equivalent to:
    //   top = top + 1;
    //   arr[top] = val;
}
```

**Detailed Step-by-Step:**

```
Initial state: top = -1, capacity = 5

push(10):
  Check: top (-1) >= capacity-1 (4)?  NO → proceed
  top = top + 1  →  top = 0        ← increment FIRST (pre-increment)
  arr[0] = 10                       ← then write
  ┌────┬────┬────┬────┬────┐
  │ 10 │ __ │ __ │ __ │ __ │
  └────┴────┴────┴────┴────┘
    ↑ top=0

push(20):
  Check: top (0) >= 4?  NO → proceed
  top = top + 1  →  top = 1
  arr[1] = 20
  ┌────┬────┬────┬────┬────┐
  │ 10 │ 20 │ __ │ __ │ __ │
  └────┴────┴────┴────┴────┘
         ↑ top=1

push(30):
  Check: top (1) >= 4?  NO → proceed
  top = top + 1  →  top = 2
  arr[2] = 30
  ┌────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ __ │ __ │
  └────┴────┴────┴────┴────┘
              ↑ top=2
```

**Stack Overflow — what happens when full:**
```
State: top = 4 (stack full, capacity = 5)
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
                      ↑ top=4

push(60):
  Check: top (4) >= capacity-1 (4)?  YES → OVERFLOW!
  Throws: std::overflow_error("Stack Overflow")
  Stack unchanged.
```

**Critical mistake — pre vs post increment:**
```cpp
// ✅ CORRECT — increment top first, then write
arr[++top] = val;
// top goes from 1 → 2, then arr[2] = val

// ❌ WRONG — write first using OLD top, then increment
arr[top++] = val;
// arr[1] = val (overwrites existing!), then top → 2

Rule: Always pre-increment (++top) when pushing.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 3: Pop (Remove from Top)

**Purpose:** Remove the top element from the stack.

**Code:**
```cpp
void pop() {
    // Guard against underflow
    if (top < 0) {
        throw std::underflow_error("Stack Underflow");
    }

    // Just move the top pointer down
    top--;
    // The value at arr[top+1] is now "invisible" to the stack
}
```

**Detailed Explanation:**

```
Before pop:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
              ↑
           top = 2

After pop:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │  ← 30 still in memory!
└────┴────┴────┴────┴────┘
       ↑     ↑
    top=1   "dead data" (ignored)
```

**Why not actually delete the element?**
```
Reason 1: Speed
  - top-- is a single CPU instruction → O(1)
  - Writing a zero or calling free() is extra work

Reason 2: The value is already inaccessible
  - peek() returns arr[top], which is now arr[1]
  - arr[2] is never shown to the user again

Reason 3: Next push will overwrite it
  - push(99): top++ → top=2, arr[2] = 99
  - The "dead data" 30 is cleanly overwritten
```

**Stack Underflow:**
```
Empty stack:
top = -1
┌────┬────┬────┬────┬────┐
│ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┘

pop():
  Check: top (-1) < 0?  YES → UNDERFLOW!
  Throws: std::underflow_error("Stack Underflow")
```

**Multiple pops — watching top shrink:**
```
Start: [10][20][30]  top=2

pop() → top=1   [10][20] (30 ignored)
pop() → top=0   [10]     (20 and 30 ignored)
pop() → top=-1  []       (all ignored, stack empty)
pop() → UNDERFLOW!
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 4: Peek / Top (View Without Removing)

**Purpose:** Return the value of the top element without changing the stack.

**Code:**
```cpp
int peek() {
    if (top < 0) {
        throw std::underflow_error("Stack is Empty");
    }
    return arr[top];    // Read, but do NOT decrement top
}
```

**Detailed Explanation:**
```
Stack state:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
              ↑ top=2

peek() → returns 30
Stack UNCHANGED (top still = 2)

Compare peek() vs pop():
  peek() → returns arr[top], top stays the same
  pop()  → just decrements top (no return value)
  
If you need both the value AND to remove it:
  int val = stack.peek();   // get value
  stack.pop();              // then remove
```

**Common Usage Pattern:**
```cpp
// Look before you pop
while (!s.isEmpty()) {
    int val = s.peek();
    if (val == sentinel) break;   // Decide based on top
    s.pop();
    process(val);
}

// Check if top is a matching bracket before popping
if (!s.isEmpty() && s.peek() == '(') {
    s.pop();  // Safely pop the matching bracket
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 5: isEmpty

**Purpose:** Check if the stack has no elements.

**Code:**
```cpp
bool isEmpty() {
    return top < 0;
    // Equivalent: return top == -1;
}
```

**Visual:**
```
Empty:      top = -1   → isEmpty() returns TRUE
One elem:   top =  0   → isEmpty() returns FALSE
Three elem: top =  2   → isEmpty() returns FALSE
```

**Common pattern — always check before peek/pop:**
```cpp
// Safe pattern:
if (!s.isEmpty()) {
    int val = s.peek();
    s.pop();
}

// Unsafe — will throw exception if empty:
int val = s.peek();  // ❌ Don't do this without checking
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 6: isFull

**Purpose:** Check if the stack has reached its maximum capacity.

**Code:**
```cpp
bool isFull() {
    return top >= capacity - 1;
    // Equivalent: return top == capacity - 1;
}
```

**Visual (capacity = 5):**
```
top = -1 → isFull() = FALSE   (empty)
top =  2 → isFull() = FALSE   (has room at indices 3, 4)
top =  4 → isFull() = TRUE    (all 5 slots used: 0..4)

Full stack (top = capacity - 1 = 4):
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
  0    1    2    3    4
                      ↑ top = 4 = capacity-1
```

**Why check before pushing:**
```cpp
// Always check if full before pushing (in array-based stack)
if (!s.isFull()) {
    s.push(val);
} else {
    cout << "Stack is full, cannot push " << val << "\n";
}

// Or just rely on the exception thrown inside push():
try {
    s.push(val);
} catch (const std::overflow_error& e) {
    cout << e.what() << "\n";
}
```

**Note:** If using the linked list version, there is no fixed capacity and `isFull()` is not needed.

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 7: Size

**Purpose:** Return the number of elements currently in the stack.

**Code:**
```cpp
int size() {
    return top + 1;
}
```

**Why top + 1?**
```
top tracks the INDEX of the top element.
Indices are 0-based, so size = top + 1.

top = -1 → size = -1 + 1 = 0   (empty)
top =  0 → size =  0 + 1 = 1   (one element)
top =  1 → size =  1 + 1 = 2   (two elements)
top =  4 → size =  4 + 1 = 5   (full, capacity=5)

Example:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
              ↑ top=2

size() = 2 + 1 = 3 ✓
```

**Common mistake:**
```cpp
Stack s(5);
s.push(10);
s.push(20);

// WRONG: Using capacity (max size) instead of size (actual elements)
int cap = 5;  // the capacity we passed to the constructor
for (int i = 0; i < cap; i++) {  // iterates 5 times!
    // Accesses garbage values at indices 2, 3, 4
}

// CORRECT:
for (int i = 0; i < s.size(); i++) { // iterates 2 times
    // Only valid elements
}
```

> **Note:** The Stack class above does not expose `getCapacity()` since users
> should only care about `size()` (how many elements) not the internal capacity.
> If you need capacity for debugging, add `int getCapacity() { return capacity; }`.

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 8: Clear

**Purpose:** Remove all elements from the stack, resetting it to an empty state.

**Code:**
```cpp
void clear() {
    top = -1;    // Reset top pointer to "empty" position
    // Memory is NOT freed; arr still allocated with full capacity
}
```

**Visual:**
```
Before clear: size=3, capacity=5
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
              ↑ top=2

After clear: size=0, capacity=5
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │  ← Old data still in memory
└────┴────┴────┴────┴────┘
top = -1   (all elements "invisible")
```

**Why just set top = -1 instead of zeroing the array?**
```
Zeroing array: O(n) — must overwrite every slot
Setting top:   O(1) — single assignment

The data at arr[0..old_top] is now "dead."
Next push will overwrite slots starting from arr[0].
The old values can never be accessed through the stack API.

Performance-critical code benefits greatly from this trick.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 9: Destructor

**Purpose:** Free the heap-allocated array when the Stack object goes out of scope.

**Code:**
```cpp
~Stack() {
    delete[] arr;   // Release heap memory
    // top and capacity are on the stack — auto-freed
}
```

**Memory lifecycle:**
```
Constructor:    arr = new int[size];  ← heap memory ALLOCATED
...use stack...
Destructor:     delete[] arr;         ← heap memory FREED

Without destructor:
  arr dangling pointer on heap — MEMORY LEAK!
  Every Stack object created leaks n * sizeof(int) bytes.
```

**The danger without a destructor:**
```cpp
void someFunction() {
    Stack s(100);         // 400 bytes allocated on heap
    s.push(1);
    s.push(2);
    // Function ends. Stack object destroyed.
    // But WITHOUT destructor, the 400 bytes are NEVER freed!
}  // ← 400 bytes leaked every time this function is called
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

## Stack Using Linked List

**When to use this version?**
- You don't know the maximum size in advance
- You want dynamic sizing (no fixed capacity)
- You cannot afford the wasted capacity of an array

**Node structure:**
```cpp
struct Node {
    int data;
    Node* next;
    Node(int val) : data(val), next(nullptr) {}
};
```

**Code:**
```cpp
class StackLL {
private:
    Node* topNode;   // Points to the top node
    int count;       // Track size

public:
    StackLL() : topNode(nullptr), count(0) {}

    void push(int val) {
        Node* newNode = new Node(val);
        newNode->next = topNode;   // New node points to old top
        topNode = newNode;         // New node IS the new top
        count++;
    }

    void pop() {
        if (!topNode) {
            throw std::underflow_error("Stack Underflow");
        }
        Node* temp = topNode;
        topNode = topNode->next;   // Move top down
        delete temp;               // Free popped node
        count--;
    }

    int peek() {
        if (!topNode) {
            throw std::underflow_error("Stack is Empty");
        }
        return topNode->data;
    }

    bool isEmpty() { return topNode == nullptr; }
    int  size()    { return count; }

    ~StackLL() {
        while (topNode) {
            Node* temp = topNode;
            topNode = topNode->next;
            delete temp;
        }
    }
};
```

**Visual — how push and pop work with LL:**
```
Initially empty:
topNode → NULL

push(10):
  newNode = [10|NULL]
  newNode->next = NULL (old topNode)
  topNode = newNode
  topNode → [10|NULL]

push(20):
  newNode = [20|NULL]
  newNode->next = topNode  →  newNode = [20|●]
                                                ↓
                                            [10|NULL]
  topNode = newNode
  topNode → [20|●]→[10|NULL]

push(30):
  topNode → [30|●]→[20|●]→[10|NULL]

pop():
  temp = topNode (points to [30])
  topNode = topNode->next  →  topNode → [20|●]→[10|NULL]
  delete temp (frees [30])
  
  topNode → [20|●]→[10|NULL]   ← peek() now returns 20
```

**Array vs Linked List Stack Comparison:**

| Feature          | Array-Based Stack     | Linked List Stack      |
|------------------|-----------------------|------------------------|
| **Max size**     | Fixed (capacity)      | Dynamic (unlimited)    |
| **Memory**       | May waste capacity    | Allocates only needed  |
| **Overflow**     | Possible — must check | Never (heap willing)   |
| **Per-element**  | Just data             | Data + next pointer    |
| **Cache perf.**  | Better (contiguous)   | Worse (scattered)      |
| **isFull()**     | Needed                | Not needed             |
| **Complexity**   | All O(1)              | All O(1)               |
| **Destructor**   | delete[] arr          | Must walk and delete   |

**Rule of thumb:**
- Know max size? → Use array (simpler, faster cache)
- Unknown / variable size? → Use linked list (no capacity limit)

---

## Applications (Stack)

### Application 1: Valid Parentheses

**Problem:** Check if a string of brackets is balanced.

**Examples:**
```
"()"        → Valid
"()[]{}"    → Valid
"({[]})"    → Valid
"(]"        → Invalid (wrong closing)
"([)]"      → Invalid (wrong order)
"{"         → Invalid (unclosed)
```

**Why a stack?**
```
Key insight: When you see a closing bracket,
it MUST match the MOST RECENTLY opened bracket.

"( { [ ] } )"
 1 2 3     ↑ close 3 first (most recent)
         ↑   then close 2
       ↑     then close 1

"Most recently opened" = top of stack = LIFO!
```

**Code:**
```cpp
#include <stack>
#include <string>
using namespace std;

bool isValidParentheses(string s) {
    stack<char> st;

    for (char c : s) {
        if (c == '(' || c == '{' || c == '[') {
            st.push(c);            // Opening: always push
        } else {
            if (st.empty()) return false;   // No matching open

            char topChar = st.top();

            if ((c == ')' && topChar != '(') ||
                (c == '}' && topChar != '{') ||
                (c == ']' && topChar != '[')) {
                return false;      // Wrong match
            }

            st.pop();              // Valid match — consume it
        }
    }

    return st.empty();  // True only if all brackets matched
}
```

**Step-by-step trace for "({[]})":**
```
c = '(' → push
Stack: ['(']

c = '{' → push
Stack: ['(', '{']

c = '[' → push
Stack: ['(', '{', '[']

c = ']' → closing
  top = '[' → matches ']' ✓  → pop
Stack: ['(', '{']

c = '}' → closing
  top = '{' → matches '}' ✓  → pop
Stack: ['(']

c = ')' → closing
  top = '(' → matches ')' ✓  → pop
Stack: []

Stack empty → return TRUE ✓
```

**Step-by-step trace for "([)]" (invalid):**
```
c = '(' → push
Stack: ['(']

c = '[' → push
Stack: ['(', '[']

c = ')' → closing
  top = '[' → does NOT match ')' ❌
  return FALSE immediately
```

**Time Complexity:** O(n) — each character processed once
**Space Complexity:** O(n) — stack holds at most n/2 items

---

### Application 2: Infix to Postfix Conversion

**Why do we need this?**
```
Humans write:   3 + 4 * 2          (infix — operator between operands)
Computers want: 3 4 2 * +          (postfix — operator after operands)

Computers evaluate postfix with NO parentheses rules or precedence:
just "take two numbers, apply operator" — much simpler!
```

> **Note on spacing:** Postfix notation for single-digit numbers can skip
> spaces (e.g., `342*+`), but multi-digit numbers REQUIRE spaces or a
> delimiter (e.g., `12 3 +` means 12 + 3, not 1, 2, 3). The trace below
> uses single-digit examples so spaces are omitted for brevity.

**Precedence rules:**
```
*  /  →  higher precedence  (2)
+  -  →  lower  precedence  (1)
(     →  treat as lowest when deciding to pop
```

**Code:**
```cpp
#include <stack>
#include <string>
using namespace std;

int precedence(char op) {
    if (op == '*' || op == '/') return 2;
    if (op == '+' || op == '-') return 1;
    return 0;
}

string infixToPostfix(string expr) {
    stack<char> st;
    string result = "";

    for (char c : expr) {
        // Case 1: Operand — directly append to output
        if (isalnum(c)) {
            result += c;
        }
        // Case 2: '(' — push onto stack
        else if (c == '(') {
            st.push(c);
        }
        // Case 3: ')' — pop until matching '('
        else if (c == ')') {
            while (!st.empty() && st.top() != '(') {
                result += st.top();
                st.pop();
            }
            st.pop();  // Discard the '('
        }
        // Case 4: Operator — pop higher/equal precedence first
        else {
            while (!st.empty() &&
                   st.top() != '(' &&
                   precedence(st.top()) >= precedence(c)) {
                result += st.top();
                st.pop();
            }
            st.push(c);
        }
    }

    // Pop remaining operators
    while (!st.empty()) {
        result += st.top();
        st.pop();
    }

    return result;
}
```

**Step-by-step trace for "3 + 4 * 2":**
```
(spaces removed: "3+4*2")

c = '3' → operand → output: "3"
Stack: []

c = '+' → operator
  Stack empty → just push
Stack: ['+']    output: "3"

c = '4' → operand → output: "34"
Stack: ['+']

c = '*' → operator
  Top is '+' with precedence 1
  '*' has precedence 2
  2 > 1 → do NOT pop '+', just push '*'
Stack: ['+', '*']    output: "34"

c = '2' → operand → output: "342"
Stack: ['+', '*']

End of string:
  Pop '*' → output: "342*"
  Pop '+' → output: "342*+"

Final postfix: "342*+"
Reads as: 3, 4, 2, *, +
  = 3 + (4 * 2) ✓
```

**Step-by-step trace for "(A+B)*C":**
```
c = '(' → push
Stack: ['(']    output: ""

c = 'A' → operand → output: "A"

c = '+' → operator
  Top is '(' → stop (never pop past '(')
Stack: ['(', '+']    output: "A"

c = 'B' → operand → output: "AB"

c = ')' → closing paren
  Pop until '(': pop '+' → output: "AB+"
  Discard '('
Stack: []    output: "AB+"

c = '*' → operator → push
Stack: ['*']    output: "AB+"

c = 'C' → operand → output: "AB+C"

End: pop '*' → output: "AB+C*"
Final: "AB+C*"  = (A+B)*C ✓
```

**Time Complexity:** O(n) — each character processed once
**Space Complexity:** O(n) — stack holds at most n operators

---

### Application 3: Evaluate Postfix Expression

**Problem:** Given a postfix expression, compute its value.

**Rule:** Scan left to right. If number → push. If operator → pop two numbers, apply, push result.

**Code:**
```cpp
#include <stack>
#include <string>
using namespace std;

int evaluatePostfix(string expr) {
    stack<int> st;

    for (char c : expr) {
        if (isdigit(c)) {
            st.push(c - '0');    // Convert char to int
        } else if (c != ' ') {
            // Pop two operands (order matters for - and /)
            int val2 = st.top(); st.pop();
            int val1 = st.top(); st.pop();

            switch (c) {
                case '+': st.push(val1 + val2); break;
                case '-': st.push(val1 - val2); break;
                case '*': st.push(val1 * val2); break;
                case '/': st.push(val1 / val2); break;
            }
        }
    }

    return st.top();   // Final answer
}
```

**Step-by-step trace for "342*+":**
```
c = '3' → push 3
Stack: [3]

c = '4' → push 4
Stack: [3, 4]

c = '2' → push 2
Stack: [3, 4, 2]

c = '*' → pop 2 and 4
  val2 = 2, val1 = 4
  push(4 * 2) = push(8)
Stack: [3, 8]

c = '+' → pop 8 and 3
  val2 = 8, val1 = 3
  push(3 + 8) = push(11)
Stack: [11]

return st.top() = 11

Verification: 3 + 4 * 2 = 3 + 8 = 11 ✓
```

**Why pop order matters:**
```
For "5 3 -" (which means 5 - 3 = 2):

Stack before '-': [5, 3]

val2 = st.top() = 3   ← popped first (last in)
val1 = st.top() = 5   ← popped second (first in)

Compute: val1 - val2 = 5 - 3 = 2 ✓

If reversed: val2 - val1 = 3 - 5 = -2 ✗
The ORDER of pops is critical for non-commutative operators!
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

### Application 4: Next Greater Element

**Problem:** For each element in an array, find the nearest greater element to its right. Return -1 if none exists.

**Example:**
```
Input:  [4, 5, 2, 10, 8]
Output: [5, 10, 10, -1, -1]

Explanation:
4  → next greater = 5
5  → next greater = 10
2  → next greater = 10
10 → nothing greater → -1
8  → nothing greater → -1
```

**Brute force:** For each element, scan right → O(n²)
**Stack (monotonic stack):** O(n)

**Code:**
```cpp
#include <stack>
#include <vector>
using namespace std;

vector<int> nextGreaterElement(vector<int>& arr) {
    int n = arr.size();
    vector<int> result(n, -1);
    stack<int> st;   // Stores INDICES (not values)

    for (int i = n - 1; i >= 0; i--) {   // Traverse right-to-left

        // Pop indices whose values are <= current
        while (!st.empty() && arr[st.top()] <= arr[i]) {
            st.pop();
        }

        // If stack not empty, top has next greater
        if (!st.empty()) {
            result[i] = arr[st.top()];
        }

        st.push(i);    // Push current index
    }

    return result;
}
```

**Step-by-step trace for [4, 5, 2, 10, 8]:**
```
i=4 (val=8):
  Stack empty → result[4] = -1
  Push index 4
  Stack: [4]  (values: [8])

i=3 (val=10):
  arr[st.top()] = arr[4] = 8 ≤ 10 → pop
  Stack empty → result[3] = -1
  Push index 3
  Stack: [3]  (values: [10])

i=2 (val=2):
  arr[st.top()] = arr[3] = 10 > 2 → STOP
  result[2] = arr[3] = 10
  Push index 2
  Stack: [3, 2]  (values: [10, 2])

i=1 (val=5):
  arr[st.top()] = arr[2] = 2 ≤ 5 → pop
  Stack: [3]
  arr[st.top()] = arr[3] = 10 > 5 → STOP
  result[1] = arr[3] = 10
  Push index 1
  Stack: [3, 1]  (values: [10, 5])

i=0 (val=4):
  arr[st.top()] = arr[1] = 5 > 4 → STOP
  result[0] = arr[1] = 5
  Push index 0
  Stack: [3, 1, 0]

Final: result = [5, 10, 10, -1, -1] ✓
```

**The "monotonic stack" concept:**
```
The stack always maintains indices in order such that
their corresponding VALUES are monotonically decreasing
(from bottom to top).

When we see a new element larger than the top:
  → The new element IS the "next greater" for the popped index
  → Pop it (it's been answered)

Elements remaining in stack at the end have no next greater → -1
```

**Time Complexity:** O(n) — each element pushed and popped at most once
**Space Complexity:** O(n) — worst case all elements on stack

---

### Application 5: Min Stack

**Problem:** Design a stack that supports push, pop, peek, AND getMin — all in O(1) time.

**Why is this hard?**
```
A regular stack loses track of the minimum when you pop elements.

Example: push 3, push 1, push 5, push 2
  Min = 1

pop() removes 2 → min is still 1   (easy)
pop() removes 5 → min is still 1   (easy)
pop() removes 1 → min is now 3     (how do we know this?!)

We need to REMEMBER what the minimum was before we pushed 1.
```

**Solution — Dual Stack:**
```
Maintain two stacks:
  mainStack   → stores all elements normally
  minStack    → stores the current minimum at each level

Rule: When pushing val, push min(val, minStack.top()) to minStack.
```

**Code:**
```cpp
class MinStack {
private:
    stack<int> mainStack;
    stack<int> minStack;

public:
    void push(int val) {
        mainStack.push(val);

        // Push new minimum (val vs current min)
        if (minStack.empty()) {
            minStack.push(val);
        } else {
            minStack.push(min(val, minStack.top()));
        }
    }

    void pop() {
        if (mainStack.empty()) throw underflow_error("Empty");
        mainStack.pop();
        minStack.pop();    // Both stacks stay in sync
    }

    int peek() {
        if (mainStack.empty()) throw underflow_error("Empty");
        return mainStack.top();
    }

    int getMin() {
        if (minStack.empty()) throw underflow_error("Empty");
        return minStack.top();    // O(1)!
    }

    bool isEmpty() { return mainStack.empty(); }
};
```

**Step-by-step trace:**
```
push(3):
  mainStack: [3]      minStack: [3]        getMin() = 3

push(1):
  min(1, 3) = 1
  mainStack: [3, 1]   minStack: [3, 1]     getMin() = 1

push(5):
  min(5, 1) = 1
  mainStack: [3, 1, 5] minStack: [3, 1, 1] getMin() = 1

push(2):
  min(2, 1) = 1
  mainStack: [3, 1, 5, 2] minStack: [3, 1, 1, 1] getMin() = 1

pop():
  mainStack: [3, 1, 5]    minStack: [3, 1, 1]    getMin() = 1

pop():
  mainStack: [3, 1]       minStack: [3, 1]        getMin() = 1

pop():
  mainStack: [3]          minStack: [3]            getMin() = 3

The minimum correctly "remembered" its previous value!
```

**Time Complexity:** All operations O(1)
**Space Complexity:** O(n) — extra minStack

---

## Complete Implementation (Stack)

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

class Stack {
private:
    int* arr;
    int top;
    int capacity;

public:
    // Constructor
    Stack(int size = 100) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }

    // Add to top
    void push(int val) {
        if (top >= capacity - 1) {
            throw overflow_error("Stack Overflow");
        }
        arr[++top] = val;
    }

    // Remove from top
    void pop() {
        if (top < 0) {
            throw underflow_error("Stack Underflow");
        }
        top--;
    }

    // View top element
    int peek() {
        if (top < 0) {
            throw underflow_error("Stack is Empty");
        }
        return arr[top];
    }

    // Check if empty
    bool isEmpty() { return top < 0; }

    // Check if full
    bool isFull() { return top >= capacity - 1; }

    // Current number of elements
    int size() { return top + 1; }

    // Remove all elements
    void clear() { top = -1; }

    // Destructor
    ~Stack() { delete[] arr; }
};

// Usage example
int main() {
    Stack s(5);

    // Push elements
    s.push(10);
    s.push(20);
    s.push(30);

    cout << "Size: " << s.size() << endl;    // 3
    cout << "Top:  " << s.peek() << endl;    // 30
    cout << "Full? " << (s.isFull() ? "Yes" : "No") << endl;  // No

    // Pop and print
    while (!s.isEmpty()) {
        cout << s.peek() << " ";   // 30 20 10
        s.pop();
    }
    cout << endl;

    cout << "Empty? " << (s.isEmpty() ? "Yes" : "No") << endl;  // Yes

    return 0;
}  // Destructor called automatically
```

---

## C++ STL `std::stack` Quick Reference

When solving problems in competitive programming or writing real code, you'll use the built-in `std::stack` instead of building your own. Here's how it maps to what you just learned:

```cpp
#include <stack>
using namespace std;

stack<int> s;          // Create an empty stack (dynamic, no capacity limit)

s.push(10);            // Same as our push()
s.push(20);
s.push(30);

s.top();               // Same as our peek() — returns 30
s.pop();               // Same as our pop() — removes 30 (no return value!)
s.empty();             // Same as our isEmpty() — returns true/false
s.size();              // Same as our size() — returns number of elements

// NOTE: std::stack has NO isFull() — it grows dynamically
// NOTE: std::stack has NO clear() — use while(!s.empty()) s.pop();
// NOTE: pop() does NOT return the value — use top() first, then pop()
```

**The pop-and-use pattern (very common):**
```cpp
// WRONG — pop() returns void, not the value
int val = s.pop();   // ❌ Does not compile

// CORRECT
int val = s.top();   // Step 1: read the value
s.pop();             // Step 2: remove it
```

**Underlying container:**
`std::stack` is a container adaptor. By default it uses `std::deque` internally, but you can change it:
```cpp
stack<int, vector<int>> s;   // Uses vector as backing storage
```

---

## Complexity Summary (Stack)

| Operation   | Time | Space | Notes                         |
|-------------|------|-------|-------------------------------|
| Constructor | O(1) | O(n)  | Allocates array of size n     |
| push()      | O(1) | O(1)  | Increment top, write value    |
| pop()       | O(1) | O(1)  | Decrement top (no deletion)   |
| peek()      | O(1) | O(1)  | Return arr[top]               |
| isEmpty()   | O(1) | O(1)  | Check top < 0                 |
| isFull()    | O(1) | O(1)  | Check top == capacity-1       |
| size()      | O(1) | O(1)  | Return top + 1                |
| clear()     | O(1) | O(1)  | Reset top to -1               |
| Destructor  | O(1) | O(1)  | Free array                    |

All stack operations are O(1). This is its greatest strength.

---

## Common Use Cases (Stack)

✅ **Use Stack When:**
- Need LIFO (Last In, First Out) order
- Implementing **function call management** (call stack)
- **Balanced parentheses** / bracket matching
- **Expression evaluation** (infix → postfix, postfix evaluation)
- **Backtracking** (maze, sudoku, N-Queens)
- **Depth-First Search (DFS)** — iterative version
- **Undo/Redo** operations in editors
- **Browser back button** (navigation history)
- Reversing data (strings, sequences)

❌ **Don't Use Stack When:**
- Need to access elements in the middle
- Need FIFO order (use Queue instead)
- Need random access by index
- Need to search for a specific element efficiently

---

## Key Takeaways (Stack)

1. **LIFO Principle:** The last element pushed is always the first popped.
2. **top = -1 means empty:** This convention makes size = top + 1 work cleanly.
3. **Pre-increment on push:** Use `arr[++top]`, not `arr[top++]`.
4. **Pop ≠ erase:** pop() just moves the top pointer; data stays in memory.
5. **Two implementations:** Array (fixed, cache-friendly) vs Linked List (dynamic).
6. **All O(1):** Push, pop, peek — all constant time. No loops.
7. **Recursion is a stack:** Every recursive call pushes a frame; every return pops one.
8. **Monotonic stacks** are a pattern for "next greater/smaller" problems in O(n).

---
---

# 2. Queue

---

## Overview

**What is it?**
A linear data structure following the **FIFO** (First In, First Out) principle. Elements enter from the **rear** (enqueue) and leave from the **front** (dequeue). The first element added is the first to be removed.

**Why use it?**
- ✅ Models any real-world waiting line
- ✅ Level-order (BFS) traversal of trees and graphs
- ✅ CPU and IO scheduling
- ✅ Buffering between producer and consumer
- ✅ Asynchronous request handling

**Real-world analogy:**
A line at a ticket counter. The person who arrives first gets served first. New people join at the back. No cutting in line allowed — that's FIFO.

```
Front (dequeue here)        Rear (enqueue here)
       ↓                           ↓
  ┌────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ 40 │ 50 │
  └────┴────┴────┴────┴────┘
  ← remove                     add →
```

**FIFO in action:**
```
Enqueue order: 10 → 20 → 30
Dequeue order: 10 → 20 → 30  (same order! not reversed like stack)

This "fairness" property is why queues model scheduling.
```

**Stack vs Queue at a glance:**
```
STACK (LIFO):               QUEUE (FIFO):
  Push/Pop at top only        Enqueue at rear, Dequeue at front

  ↓ ↑                         → [10][20][30] →
  ┌─┐                         enqueue →    → dequeue
  │3│  ← last in,             (rear)          (front)
  │2│     first out
  │1│  ← first in,
  └─┘     last out
```

---

## Structure Definition

```cpp
class Queue {
private:
    int* arr;       // Underlying array
    int front;      // Index of front element
    int rear;       // Index of rear element
    int capacity;   // Maximum number of elements
    int count;      // Current number of elements

public:
    Queue(int size = 100);   // Constructor
    ~Queue();                // Destructor

    void enqueue(int val);   // Add to rear
    void dequeue();          // Remove from front
    int  getFront();         // View front
    int  getRear();          // View rear
    bool isEmpty();          // Is queue empty?
    bool isFull();           // Is queue full?
    int  size();             // How many elements?
    void clear();            // Remove all elements
};
```

**Member Variables Explained:**
```
arr      → Heap-allocated array storing elements

front    → Index of the element that will be dequeued next.
           Starts at 0.

rear     → Index of the most recently enqueued element.
           Starts at -1 (nothing enqueued yet).

capacity → Maximum number of elements.

count    → The ONLY reliable indicator of empty vs full.
           (front and rear pointers alone are ambiguous
            after wrap-around — more on this below)
```

**Why we need BOTH front, rear, AND count:**
```
Without count, it's impossible to distinguish these two states:

FULL (5 elements, capacity=5):
front=2, rear=1
┌────┬────┬────┬────┬────┐
│ 60 │ 70 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
       ↑   ↑
     rear front

EMPTY (all elements removed):
front=2, rear=1   ← IDENTICAL to full!
┌────┬────┬────┬────┬────┐
│ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┘

count = 5 → FULL
count = 0 → EMPTY
count solves the ambiguity.
```

---

## The Problem with a Naive Queue

Before implementing a circular queue, understand WHY it's necessary.

**Naive (linear) queue approach:**
```
Enqueue at rear, advance rear rightward.
Dequeue at front, advance front rightward.

Start:
front=0, rear=-1
┌────┬────┬────┬────┬────┐
│ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┘

Enqueue 10, 20, 30:
front=0, rear=2
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
  ↑         ↑
front     rear

Dequeue twice:
front=2, rear=2
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ __ │ __ │
└────┴────┴────┴────┴────┘
             ↑
         front=rear=2

Enqueue 40, 50:
front=2, rear=4
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
  ↑ ↑           ↑
 free slots   rear=4

Enqueue 60:
rear = 4 + 1 = 5 → OUT OF BOUNDS! ← FALSE OVERFLOW!
(Indices 0 and 1 are FREE but we can't use them)
```

**Solution: Circular Queue with modulo arithmetic**
```
Treat the array as a circle:
rear = (rear + 1) % capacity

When rear reaches capacity-1 and wraps:
(4 + 1) % 5 = 0  ← Goes back to index 0!

Visualize as a ring:
         [0]
    [4]       [1]
    [3]       [2]

Elements can "wrap around" to reuse freed slots.
```

---

## Operations

### Operation 1: Constructor

**Purpose:** Allocate the array and initialize the queue to an empty state.

**Code:**
```cpp
Queue(int size = 100) {
    arr = new int[size];
    capacity = size;
    front = 0;       // Will point to first element when enqueued
    rear = -1;       // -1: nothing enqueued yet
    count = 0;       // Zero elements
}
```

**Initial state visual:**
```
Queue q(5);

capacity = 5
front    = 0
rear     = -1
count    = 0

┌────┬────┬────┬────┬────┐
│ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┘
  ↑
front=0, rear=-1, count=0
(Nothing to dequeue yet)
```

**Why rear starts at -1?**
```
When we enqueue the FIRST element:
  rear = (-1 + 1) % capacity = 0
  arr[0] = val

This correctly places the first element at index 0.
If rear started at 0, we'd skip index 0 and start at 1.
```

**Time Complexity:** O(1)
**Space Complexity:** O(n)

---

### Operation 2: Enqueue (Add to Rear)

**Purpose:** Add a new element at the rear of the queue.

**Code:**
```cpp
void enqueue(int val) {
    // Guard against overflow
    if (count == capacity) {
        throw overflow_error("Queue Overflow");
    }

    // Circular advance of rear
    rear = (rear + 1) % capacity;
    arr[rear] = val;
    count++;
}
```

**Detailed Step-by-Step:**

```
Initial: front=0, rear=-1, count=0, capacity=5

enqueue(10):
  rear = (-1 + 1) % 5 = 0
  arr[0] = 10,  count = 1
  ┌────┬────┬────┬────┬────┐
  │ 10 │ __ │ __ │ __ │ __ │
  └────┴────┴────┴────┴────┘
    ↑
  front=0, rear=0

enqueue(20):
  rear = (0 + 1) % 5 = 1
  arr[1] = 20,  count = 2
  ┌────┬────┬────┬────┬────┐
  │ 10 │ 20 │ __ │ __ │ __ │
  └────┴────┴────┴────┴────┘
    ↑    ↑
  front  rear=1

enqueue(30):
  rear = (1 + 1) % 5 = 2
  arr[2] = 30,  count = 3
  ┌────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ __ │ __ │
  └────┴────┴────┴────┴────┘
    ↑         ↑
  front=0   rear=2
```

**Wrap-around example (the circular magic):**
```
State: queue has [30, 40, 50] at indices 2, 3, 4
front=2, rear=4, count=3

┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
             ↑         ↑
           front=2   rear=4

enqueue(60):
  rear = (4 + 1) % 5 = 0  ← WRAPS AROUND!
  arr[0] = 60, count = 4
  ┌────┬────┬────┬────┬────┐
  │ 60 │ __ │ 30 │ 40 │ 50 │
  └────┴────┴────┴────┴────┘
    ↑        ↑         ↑
  rear=0  front=2   (wraps)

enqueue(70):
  rear = (0 + 1) % 5 = 1
  arr[1] = 70, count = 5
  ┌────┬────┬────┬────┬────┐
  │ 60 │ 70 │ 30 │ 40 │ 50 │
  └────┴────┴────┴────┴────┘
         ↑   ↑
       rear  front=2    count=5 (FULL)

Logical order: 30 → 40 → 50 → 60 → 70
                ↑ front                ↑ rear
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 3: Dequeue (Remove from Front)

**Purpose:** Remove the element at the front of the queue.

**Code:**
```cpp
void dequeue() {
    if (count == 0) {
        throw underflow_error("Queue Underflow");
    }

    // Circular advance of front
    front = (front + 1) % capacity;
    count--;
}
```

**Detailed Explanation:**

```
State: [30, 40, 50, 60, 70] → front=2, rear=1, count=5
┌────┬────┬────┬────┬────┐
│ 60 │ 70 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
       ↑   ↑
     rear  front=2

dequeue():
  front = (2 + 1) % 5 = 3
  count = 4
  ┌────┬────┬────┬────┬────┐
  │ 60 │ 70 │ 30 │ 40 │ 50 │  ← 30 is "dead data"
  └────┴────┴────┴────┴────┘
       ↑        ↑
     rear     front=3

getFront() now returns 40.

dequeue() again:
  front = (3 + 1) % 5 = 4
  count = 3
  ┌────┬────┬────┬────┬────┐
  │ 60 │ 70 │ 30 │ 40 │ 50 │
  └────┴────┴────┴────┴────┘
       ↑              ↑
     rear           front=4

getFront() now returns 50.
```

**Wrap-around dequeue:**
```
State: front=4, rear=1, count=3
┌────┬────┬────┬────┬────┐
│ 60 │ 70 │ __ │ __ │ 50 │
└────┴────┴────┴────┴────┘
       ↑                ↑
     rear=1          front=4

dequeue():
  front = (4 + 1) % 5 = 0  ← WRAPS AROUND!
  count = 2
  ┌────┬────┬────┬────┬────┐
  │ 60 │ 70 │ __ │ __ │ 50 │
  └────┴────┴────┴────┴────┘
    ↑    ↑
  front  rear=1
  front=0

getFront() now returns 60.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 4: getFront (View Front Element)

**Purpose:** Return the front element without removing it.

**Code:**
```cpp
int getFront() {
    if (count == 0) {
        throw underflow_error("Queue is Empty");
    }
    return arr[front];
}
```

**Visual:**
```
Queue state: front=2, rear=4, count=3
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
             ↑
           front=2

getFront() → returns 30
Queue UNCHANGED (front still = 2, count still = 3)
```

**getFront() vs dequeue():**
```
getFront() → reads arr[front], does NOT advance front
dequeue()  → advances front, does NOT return the value

To get-and-remove the front:
  int val = q.getFront();
  q.dequeue();
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 5: getRear (View Rear Element)

**Purpose:** Return the rear element (most recently added) without removing it.

**Code:**
```cpp
int getRear() {
    if (count == 0) {
        throw underflow_error("Queue is Empty");
    }
    return arr[rear];
}
```

**Visual:**
```
Queue state: front=2, rear=4, count=3
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
                      ↑
                    rear=4

getRear() → returns 50
Queue UNCHANGED
```

**Common use case:**
```cpp
// "Peek" at both ends without removing
cout << "Front: " << q.getFront() << endl;  // First to leave
cout << "Rear:  " << q.getRear()  << endl;  // Last to leave

// Useful in deque-based sliding window algorithms
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 6: isEmpty

**Purpose:** Check whether the queue has zero elements.

**Code:**
```cpp
bool isEmpty() {
    return count == 0;
}
```

**Why use count instead of checking if front == rear?**
```
front == rear actually means EXACTLY ONE element in our design —
it is never ambiguous for detecting "empty" on its own.

The real problem count solves is distinguishing FULL from EMPTY
after wrap-around, when the two states can produce IDENTICAL
(front, rear) values:

FULL  (capacity=5, 5 elements): front=2, rear=1
EMPTY (same queue, all removed): front=2, rear=1  ← identical!

count = 5 → FULL
count = 0 → EMPTY

Without count, FULL and EMPTY look the same after wrap-around.
count is the only reliable indicator.
```

**Common pattern:**
```cpp
// Process all elements in order
while (!q.isEmpty()) {
    int val = q.getFront();
    q.dequeue();
    process(val);
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 7: isFull

**Purpose:** Check whether the queue has reached maximum capacity.

**Code:**
```cpp
bool isFull() {
    return count == capacity;
}
```

**Visual:**
```
capacity = 5:

count = 0 → isFull() = FALSE  (empty)
count = 3 → isFull() = FALSE  (has room)
count = 5 → isFull() = TRUE   (cannot enqueue more)
```

**Always check before enqueuing (or catch the exception):**
```cpp
if (!q.isFull()) {
    q.enqueue(val);
} else {
    cout << "Queue is full!\n";
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 8: Size

**Purpose:** Return the number of elements currently in the queue.

**Code:**
```cpp
int size() {
    return count;
}
```

**Why count and not (rear - front + 1)?**
```
After wrap-around, (rear - front + 1) gives wrong results:

Example: front=3, rear=1 (wrapped), actual count=4

rear - front + 1 = 1 - 3 + 1 = -1  ← WRONG!

count = 4  ← always correct because it's updated
              every enqueue (+1) and dequeue (-1)
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 9: Clear

**Purpose:** Remove all elements, resetting the queue to its initial empty state.

**Code:**
```cpp
void clear() {
    front = 0;
    rear  = -1;
    count = 0;
    // Memory is NOT freed; array still allocated
}
```

**Visual:**
```
Before clear: front=2, rear=4, count=3
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘

After clear: front=0, rear=-1, count=0
┌────┬────┬────┬────┬────┐
│ __ │ __ │ 30 │ 40 │ 50 │  ← Old data in memory, but ignored
└────┴────┴────┴────┴────┘
  ↑
front=0, rear=-1
(Queue behaves as if freshly constructed)
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 10: Destructor

**Purpose:** Free the heap-allocated array.

**Code:**
```cpp
~Queue() {
    delete[] arr;
}
```

**Memory lifecycle:**
```
Constructor:   arr = new int[size];  ← heap ALLOCATED
...use queue...
Destructor:    delete[] arr;         ← heap FREED

Without destructor: n * sizeof(int) bytes leaked per Queue object.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

## Queue Using Linked List

**When to use this version?**
- Maximum queue size is unknown in advance
- You want truly unlimited capacity
- You don't want fixed capacity overhead

**Node structure:**
```cpp
struct Node {
    int data;
    Node* next;
    Node(int val) : data(val), next(nullptr) {}
};
```

**Code:**
```cpp
class QueueLL {
private:
    Node* frontNode;   // Points to front (dequeue from here)
    Node* rearNode;    // Points to rear (enqueue here)
    int count;

public:
    QueueLL() : frontNode(nullptr), rearNode(nullptr), count(0) {}

    void enqueue(int val) {
        Node* newNode = new Node(val);

        if (rearNode == nullptr) {
            // First element: both front and rear point to it
            frontNode = rearNode = newNode;
        } else {
            rearNode->next = newNode;  // Link at rear
            rearNode = newNode;        // Advance rear
        }
        count++;
    }

    void dequeue() {
        if (!frontNode) {
            throw underflow_error("Queue Underflow");
        }
        Node* temp = frontNode;
        frontNode = frontNode->next;   // Advance front

        if (frontNode == nullptr) {
            // Queue is now empty — reset rear too
            rearNode = nullptr;
        }

        delete temp;
        count--;
    }

    int getFront() {
        if (!frontNode) throw underflow_error("Queue is Empty");
        return frontNode->data;
    }

    int getRear() {
        if (!rearNode) throw underflow_error("Queue is Empty");
        return rearNode->data;
    }

    bool isEmpty() { return frontNode == nullptr; }
    int  size()    { return count; }

    ~QueueLL() {
        while (frontNode) {
            Node* temp = frontNode;
            frontNode = frontNode->next;
            delete temp;
        }
    }
};
```

**Visual — how enqueue and dequeue work:**
```
Initially empty:
frontNode → NULL,  rearNode → NULL

enqueue(10):
  newNode = [10|NULL]
  frontNode = rearNode = newNode
  frontNode → [10|NULL]
  rearNode  → [10|NULL]  (same node)

enqueue(20):
  newNode = [20|NULL]
  rearNode->next = newNode   → [10|●]→[20|NULL]
  rearNode = newNode
  frontNode → [10|●]→[20|NULL]
  rearNode  →         [20|NULL]

enqueue(30):
  frontNode → [10|●]→[20|●]→[30|NULL]
  rearNode  →                [30|NULL]

dequeue():
  temp = frontNode (points to [10])
  frontNode = frontNode->next  →  frontNode → [20|●]→[30|NULL]
  delete temp (frees [10])
  getFront() now returns 20.

dequeue():
  frontNode → [30|NULL]
  rearNode  → [30|NULL]
  getFront() returns 30.

dequeue():
  frontNode→NULL, rearNode→NULL  (special case: reset both)
  isEmpty() = true
```

**Array vs Linked List Queue Comparison:**

| Feature             | Array Queue (Circular)  | Linked List Queue       |
|---------------------|-------------------------|-------------------------|
| **Max size**        | Fixed capacity          | Unlimited (heap bound)  |
| **Memory**          | May waste slots         | Only allocates needed   |
| **Overflow**        | Possible                | Never                   |
| **Per-element**     | Just data               | Data + next pointer     |
| **Cache perf.**     | Better (contiguous)     | Worse (scattered)       |
| **No-capacity risk**| Yes — need to plan size | No                      |
| **isFull()**        | Needed                  | Not needed              |
| **Destructor**      | delete[] arr (O(1))     | Walk and delete (O(n))  |

---

## Applications (Queue)

### Application 1: BFS in a Graph

**Problem:** Traverse a graph level by level from a starting node.

**Why Queue?**
```
BFS explores ALL neighbors at the current depth before going deeper.
"Current neighbors" are added and visited in arrival order = FIFO.

Stack → DFS (dive deep first)
Queue → BFS (visit wide first)
```

**Code:**
```cpp
#include <queue>
#include <vector>
using namespace std;

void BFS(vector<vector<int>>& adj, int start, int n) {
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

**Step-by-step trace:**
```
Graph (undirected):
    0 — 1
    |   |
    2 — 3

Adjacency: 0:[1,2], 1:[0,3], 2:[0,3], 3:[1,2]

BFS from node 0:

Initial: Queue=[0], Visited={0}

Step 1: Dequeue 0, enqueue unvisited neighbors 1, 2
  Queue: [1, 2],  Visited: {0,1,2}
  Output: 0

Step 2: Dequeue 1, enqueue unvisited neighbor 3
  Queue: [2, 3],  Visited: {0,1,2,3}
  Output: 0 1

Step 3: Dequeue 2, neighbor 3 already visited
  Queue: [3]
  Output: 0 1 2

Step 4: Dequeue 3, no unvisited neighbors
  Queue: []
  Output: 0 1 2 3

Traversal: 0 → 1 → 2 → 3  (level by level)
  Level 0: {0}
  Level 1: {1, 2}
  Level 2: {3}
```

**Time Complexity:** O(V + E) where V=vertices, E=edges
**Space Complexity:** O(V) for visited array and queue

---

### Application 2: Level-Order Tree Traversal

**Problem:** Print all nodes of a binary tree level by level.

**Why Queue?**
```
Trees are hierarchical. Level-order means:
  First all nodes at depth 0 (root)
  Then all nodes at depth 1 (root's children)
  Then depth 2, etc.

Queue gives "process in the order you saw them" = FIFO = level order.
```

**Code:**
```cpp
#include <queue>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

void levelOrder(TreeNode* root) {
    if (!root) return;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int levelSize = q.size();  // Nodes at this level

        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            cout << node->val << " ";

            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
        cout << "\n";   // Newline after each level
    }
}
```

**Step-by-step trace:**
```
Tree:
        1
       / \
      2   3
     / \   \
    4   5   6

Queue progression:

Level 0:
  Queue: [1]
  Process 1, enqueue 2, 3
  Output: 1

Level 1:
  Queue: [2, 3]
  Process 2, enqueue 4, 5
  Process 3, enqueue 6
  Output: 2 3

Level 2:
  Queue: [4, 5, 6]
  Process 4 (no children)
  Process 5 (no children)
  Process 6 (no children)
  Output: 4 5 6

Final output:
  1
  2 3
  4 5 6
```

**The key trick — capturing level size:**
```cpp
int levelSize = q.size();   // Snapshot of how many nodes are at THIS level

// Process exactly levelSize nodes (this level's worth)
// Newly pushed children are at the NEXT level
// They are not counted in levelSize
```

**Time Complexity:** O(n) — each node enqueued and dequeued once
**Space Complexity:** O(n) — widest level can hold O(n) nodes

---

### Application 3: First Non-Repeating Character in a Stream

**Problem:** For a stream of characters, at each step print the first non-repeating character seen so far.

**Example:**
```
Stream:    a  b  c  b  a
Output:    a  a  a  a  c

Explanation:
After 'a': non-repeating = a → print a
After 'b': non-repeating = a → print a (still first)
After 'c': non-repeating = a → print a
After 'b': b seen twice now, non-rep = a → print a
After 'a': a seen twice now, non-rep = c → print c
```

**Code:**
```cpp
#include <queue>
#include <unordered_map>
#include <string>
using namespace std;

void firstNonRepeating(string stream) {
    queue<char> q;                // Queue of candidates (in order seen)
    unordered_map<char, int> freq;  // Frequency counter

    for (char c : stream) {
        freq[c]++;
        q.push(c);

        // Remove front if it's now repeating
        while (!q.empty() && freq[q.front()] > 1) {
            q.pop();
        }

        if (q.empty()) {
            cout << "#";    // No non-repeating char
        } else {
            cout << q.front();   // First non-repeating
        }
        cout << " ";
    }
    cout << "\n";
}
```

**Why Queue?**
```
We need the FIRST non-repeating character seen — i.e., the one
that has been in the stream the longest and is still unique.

Queue maintains insertion order (FIFO).
We discard repeaters from the front.
The front of the queue is always the oldest candidate.
```

**Trace:**
```
c='a': freq={a:1}, queue=[a], front=a → print 'a'
c='b': freq={a:1,b:1}, queue=[a,b], front=a → print 'a'
c='c': freq={a:1,b:1,c:1}, queue=[a,b,c], front=a → print 'a'
c='b': freq={a:1,b:2,c:1}, queue=[a,b,c], front=a → print 'a'
c='a': freq={a:2,b:2,c:1}, queue=[a,b,c]
        front='a' freq=2 → pop
        queue=[b,c], front='b' freq=2 → pop
        queue=[c], front='c' freq=1 → STOP
        print 'c'
```

**Time Complexity:** O(n) amortized — each char pushed/popped at most once
**Space Complexity:** O(1) — at most 26 unique characters

---

### Application 4: Sliding Window Maximum (Deque)

**Problem:** Given an array and window size k, find the maximum in each window.

**Example:**
```
Input:  [1, 3, -1, -3, 5, 3, 6, 7],  k = 3
Output: [3, 3, 5, 5, 6, 7]

Window [1, 3,-1] → max = 3
Window [3,-1,-3] → max = 3
Window [-1,-3,5] → max = 5
Window [-3, 5, 3] → max = 5
Window [5, 3, 6] → max = 6
Window [3, 6, 7] → max = 7
```

**Why Deque (Double-Ended Queue)?**
```
We need to:
1. Remove expired elements from the front (out of window)
2. Remove smaller elements from the rear (they'll never be max)
3. Find the current max at the front

Deque gives O(1) operations at BOTH ends!
```

**Code:**
```cpp
#include <deque>
#include <vector>
using namespace std;

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq;         // Stores INDICES (monotone decreasing by value)
    vector<int> result;

    for (int i = 0; i < (int)nums.size(); i++) {
        // Remove index that is out of current window
        if (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }

        // Remove indices whose values are smaller than nums[i]
        // (they can never be the max for current or future windows)
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        // Record result once first full window is formed
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }

    return result;
}
```

**Trace for [1, 3, -1, -3, 5, 3, 6, 7], k=3:**
```
i=0 (val=1):
  dq=[0] → window not full yet

i=1 (val=3):
  nums[dq.back()]=nums[0]=1 < 3 → pop 0
  dq=[1] → window not full

i=2 (val=-1):
  nums[dq.back()]=nums[1]=3 > -1 → keep
  dq=[1, 2]
  i >= k-1 → result: nums[dq.front()]=nums[1]=3
  result=[3]

i=3 (val=-3):
  nums[dq.back()]=nums[2]=-1 > -3 → keep
  dq=[1, 2, 3]
  dq.front()=1, i-k=0 → 1 > 0 → no expiry
  result: nums[1]=3
  result=[3, 3]

i=4 (val=5):
  dq.front()=1, i-k=1 → 1 <= 1 → EXPIRE → pop 1
  dq=[2, 3]
  nums[3]=-3 < 5 → pop 3
  nums[2]=-1 < 5 → pop 2
  dq=[4]
  result: nums[4]=5
  result=[3, 3, 5]

i=5 (val=3):
  dq.front()=4, i-k=2 → 4 > 2 → no expiry
  nums[4]=5 > 3 → keep
  dq=[4, 5]
  result: nums[4]=5
  result=[3, 3, 5, 5]

i=6 (val=6):
  dq.front()=4, i-k=3 → 4 > 3 → no expiry
  nums[5]=3 < 6 → pop 5
  nums[4]=5 < 6 → pop 4
  dq=[6]
  result: nums[6]=6
  result=[3, 3, 5, 5, 6]

i=7 (val=7):
  dq.front()=6, i-k=4 → 6 > 4 → no expiry
  nums[6]=6 < 7 → pop 6
  dq=[7]
  result: nums[7]=7
  result=[3, 3, 5, 5, 6, 7] ✓
```

**Time Complexity:** O(n) — each element pushed and popped at most once
**Space Complexity:** O(k) — deque holds at most k indices

---

## Complete Implementation (Queue)

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

class Queue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    // Constructor
    Queue(int size = 100) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    // Add element at rear
    void enqueue(int val) {
        if (count == capacity) {
            throw overflow_error("Queue Overflow");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = val;
        count++;
    }

    // Remove element from front
    void dequeue() {
        if (count == 0) {
            throw underflow_error("Queue Underflow");
        }
        front = (front + 1) % capacity;
        count--;
    }

    // View front element
    int getFront() {
        if (count == 0) {
            throw underflow_error("Queue is Empty");
        }
        return arr[front];
    }

    // View rear element
    int getRear() {
        if (count == 0) {
            throw underflow_error("Queue is Empty");
        }
        return arr[rear];
    }

    bool isEmpty()  { return count == 0; }
    bool isFull()   { return count == capacity; }
    int  size()     { return count; }
    void clear()    { front = 0; rear = -1; count = 0; }

    // Destructor
    ~Queue() { delete[] arr; }
};

// Usage example
int main() {
    Queue q(5);

    // Enqueue elements
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);

    cout << "Size:  " << q.size()     << endl;  // 3
    cout << "Front: " << q.getFront() << endl;  // 10
    cout << "Rear:  " << q.getRear()  << endl;  // 30
    cout << "Full?  " << (q.isFull() ? "Yes" : "No") << endl;  // No

    // Dequeue and print (FIFO order)
    while (!q.isEmpty()) {
        cout << q.getFront() << " ";  // 10 20 30
        q.dequeue();
    }
    cout << endl;

    cout << "Empty? " << (q.isEmpty() ? "Yes" : "No") << endl;  // Yes

    // Demonstrate wrap-around
    q.enqueue(40);
    q.enqueue(50);
    q.enqueue(60);    // rear wraps: (4+1)%5 = 0 ← WRAP-AROUND HERE
    q.dequeue();      // Remove 40
    q.enqueue(70);    // Goes to index 1 (no wrap)
    q.enqueue(80);    // Goes to index 2

    while (!q.isEmpty()) {
        cout << q.getFront() << " ";  // 50 60 70 80
        q.dequeue();
    }
    cout << endl;

    return 0;
}  // Destructor called automatically
```

---

## C++ STL `std::queue` Quick Reference

```cpp
#include <queue>
using namespace std;

queue<int> q;          // Create an empty queue (dynamic, no capacity limit)

q.push(10);            // Same as our enqueue() — adds to rear
q.push(20);
q.push(30);

q.front();             // Same as our getFront() — returns 10
q.back();              // Same as our getRear()  — returns 30
q.pop();               // Same as our dequeue()  — removes front (no return!)
q.empty();             // Same as our isEmpty()  — returns true/false
q.size();              // Same as our size()     — returns number of elements

// NOTE: std::queue has NO isFull() — it grows dynamically
// NOTE: std::queue has NO clear() — use while(!q.empty()) q.pop();
// NOTE: pop() does NOT return the value — use front() first, then pop()
```

**The dequeue-and-use pattern:**
```cpp
// CORRECT way to get and remove the front element:
int val = q.front();   // Step 1: read
q.pop();               // Step 2: remove
```

**STL name mapping cheatsheet:**

| Our implementation | `std::stack`  | `std::queue`  |
|--------------------|---------------|---------------|
| `push(val)`        | `push(val)`   | `push(val)`   |
| `pop()`            | `pop()`       | `pop()`       |
| `peek()`           | `top()`       | —             |
| `getFront()`       | —             | `front()`     |
| `getRear()`        | —             | `back()`      |
| `isEmpty()`        | `empty()`     | `empty()`     |
| `size()`           | `size()`      | `size()`      |
| `isFull()`         | *(none)*      | *(none)*      |

---

## Queue Variants Comparison

| Feature       | Simple Array Queue | Circular Queue | Linked List Queue | Deque          | Priority Queue |
|---------------|--------------------|----------------|-------------------|----------------|----------------|
| Enqueue       | Rear only          | Rear only      | Rear only         | Both ends      | By priority    |
| Dequeue       | Front only         | Front only     | Front only        | Both ends      | Highest first  |
| False overflow| Yes ❌              | No ✅           | N/A               | N/A            | N/A            |
| Memory        | Wastes             | Efficient      | Per-node alloc    | Efficient      | Heap-based     |
| Size          | Fixed              | Fixed          | Dynamic           | Dynamic        | Dynamic        |
| Access        | Front/rear only    | Front/rear     | Front/rear        | Both ends      | Max/min only   |
| Use case      | Simple FIFO        | Better FIFO    | Variable size     | Sliding window | Task priority  |

---

## Complexity Summary (Queue)

| Operation   | Time | Space | Notes                              |
|-------------|------|-------|------------------------------------|
| Constructor | O(1) | O(n)  | Allocates array                    |
| enqueue()   | O(1) | O(1)  | Circular rear advance              |
| dequeue()   | O(1) | O(1)  | Circular front advance             |
| getFront()  | O(1) | O(1)  | Return arr[front]                  |
| getRear()   | O(1) | O(1)  | Return arr[rear]                   |
| isEmpty()   | O(1) | O(1)  | Check count == 0                   |
| isFull()    | O(1) | O(1)  | Check count == capacity            |
| size()      | O(1) | O(1)  | Return count                       |
| clear()     | O(1) | O(1)  | Reset front, rear, count           |
| Destructor  | O(1) | O(1)  | Free array (array version); O(n) for LL version |

All core queue operations are O(1). This is why queues are ideal for scheduling.

---

## Common Use Cases (Queue)

✅ **Use Queue When:**
- Need FIFO (First In, First Out) order
- **BFS traversal** — graphs and trees
- **Level-order tree traversal**
- **CPU / task scheduling** (fairness — first come, first served)
- **IO buffering** (keyboard input, file reads, network packets)
- **Producer-consumer** problems (buffer between two rates)
- **Print spooler** — documents printed in submission order
- **Breadth-first search** in any implicit graph (words, grids)
- Call center systems (longest waiting customer served first)

✅ **Use Deque (Double-Ended Queue) When:**
- Need insert/remove at BOTH ends
- **Sliding window maximum/minimum** (monotonic deque)
- Palindrome checking (compare front and rear)
- Implementing both stack and queue simultaneously

❌ **Don't Use Queue When:**
- Need LIFO order (use Stack instead)
- Need random access by index (use Vector)
- Need to remove from or access the middle
- Need to find/search efficiently (use Hash Table or BST)

---

## Key Takeaways (Queue)

1. **FIFO Principle:** First In, First Out — the defining characteristic.
2. **Circular implementation is essential:** Prevents false overflow; always use modulo arithmetic for front and rear.
3. **Use count, not pointer comparison:** `count == 0` and `count == capacity` are the only reliable empty/full checks after wrap-around.
4. **Two pointers:** `front` tracks where to remove; `rear` tracks where to insert.
5. **rear = (rear + 1) % capacity:** The circular increment formula — memorize it.
6. **Linked list version = dynamic queue:** No fixed capacity, no overflow risk, but more memory per element.
7. **BFS is Queue's "killer app":** Any level-by-level or "explore neighbors first" algorithm naturally uses a queue.
8. **Deque extends Queue:** When you need O(1) access at both ends, upgrade to a deque.

---

## 📊 Stack vs Queue Side-by-Side

| Property          | Stack                        | Queue                          |
|-------------------|------------------------------|--------------------------------|
| **Order**         | LIFO (Last In, First Out)    | FIFO (First In, First Out)     |
| **Add at**        | Top                          | Rear                           |
| **Remove from**   | Top                          | Front                          |
| **Key operation** | push / pop / peek            | enqueue / dequeue / getFront   |
| **Top pointer**   | Single pointer (top)         | Two pointers (front + rear)    |
| **Wrap-around**   | Not needed                   | Essential (circular array)     |
| **Empty check**   | top < 0                      | count == 0                     |
| **Full check**    | top >= capacity-1            | count == capacity              |
| **Tree traversal**| DFS (depth-first)            | BFS (breadth-first)            |
| **Real-world**    | Plates, undo/redo, call stack| Ticket line, printer, CPU sched|

---

## 🧪 Testing Checklist

### Stack Tests
- [ ] Push to empty stack → peek returns pushed value
- [ ] Push multiple elements → peek returns last pushed
- [ ] Pop all elements → isEmpty() returns true
- [ ] Pop from empty stack → underflow error thrown
- [ ] Push to full stack → overflow error thrown
- [ ] clear() → size() = 0, isEmpty() = true
- [ ] size() after n pushes = n
- [ ] size() after push then pop = 0
- [ ] Linked list version — dynamic growth works correctly

### Queue Tests
- [ ] Enqueue to empty queue → getFront() = getRear() = that value
- [ ] Enqueue multiple, dequeue all in FIFO order
- [ ] Dequeue from empty → underflow error thrown
- [ ] Enqueue to full → overflow error thrown
- [ ] Wrap-around: fill, partially dequeue, enqueue new elements
- [ ] clear() → size() = 0, isEmpty() = true
- [ ] getFront() ≠ dequeue() — front stays after getFront()
- [ ] getRear() reflects latest enqueued element
- [ ] Linked list version — no capacity limit

---

## 🎯 Quick Reference Formulas

**Stack:**
```
top = -1              → empty
top = capacity - 1    → full
size = top + 1
push: arr[++top] = val
pop:  top--
peek: return arr[top]
```

**Queue (Circular):**
```
count = 0             → empty
count = capacity      → full
rear = (rear + 1) % capacity    → enqueue position
front = (front + 1) % capacity  → dequeue advance
size = count
```

**Both:**
```
Never access beyond valid range without checking isEmpty/isFull first.
All fundamental operations are O(1).
```

---

*Print this document and keep it beside you while implementing Stack and Queue from scratch.*

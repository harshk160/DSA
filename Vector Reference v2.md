# 📘 DSA Reference Manual — Dynamic Array (Vector)

**A Complete Reference Guide for C++ Implementations**

*Same depth and format as the Stack & Queue reference. Print-friendly.*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Memory Layout — The Core Idea](#memory-layout--the-core-idea)
3. [Structure Definition](#structure-definition)
4. [Operations](#operations)
   - Constructor
   - push_back
   - pop_back
   - at (safe access)
   - operator[] (fast access)
   - size
   - getCapacity
   - empty
   - clear
   - resize
   - reserve
   - insert
   - Destructor
5. [C++ STL `std::vector` Quick Reference](#c-stl-stdvector-quick-reference)
6. [Amortized Analysis — Why Doubling Works](#amortized-analysis--why-doubling-works)
7. [Common Patterns](#common-patterns)
8. [Complete Implementation](#complete-implementation)
9. [Usage Example](#usage-example)
10. [Complexity Summary](#complexity-summary)
11. [Vector vs Array vs Linked List](#vector-vs-array-vs-linked-list)
12. [Common Use Cases](#common-use-cases)
13. [Key Takeaways](#key-takeaways)

---

## Overview

**What is it?**
A **dynamic array** is a resizable array that lives in heap memory and automatically grows when it runs out of space. It gives you the best of both worlds: **random access like a regular array** and **automatic resizing like a container**.

**Why use it?**
- ✅ O(1) random access by index — jump to any element instantly
- ✅ O(1) amortized push_back — adding to the end is fast on average
- ✅ Contiguous memory — CPU cache-friendly, very fast to iterate
- ✅ No manual memory management for the user — grows automatically
- ✅ Foundation for almost all other data structures you'll build

**Real-world analogy:**
Think of a whiteboard with sticky notes. You start with space for 1 note. When it fills up, you get a whiteboard with double the space, move all notes over, and throw away the old board. The new board has plenty of empty space for more notes, so you won't need to switch boards for a while.

**The core concept before anything else:**
```
A dynamic array has TWO numbers that matter:

  SIZE     = how many elements are actually stored right now
  CAPACITY = how much total space is allocated

                 size = 3
              ┌──────────┐
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘
└──────────────────────────────────────┘
                capacity = 8

You can read/write indices 0..size-1  (valid zone)
Indices size..capacity-1 are garbage  (dead zone)
```

---

## Memory Layout — The Core Idea

This is the single most important concept for understanding a vector.

**A regular (static) array:**
```cpp
int arr[4] = {10, 20, 30, 40};

Memory (all on the STACK — fixed size, cannot grow):
Address: 100  104  108  112
         ┌────┬────┬────┬────┐
         │ 10 │ 20 │ 30 │ 40 │
         └────┴────┴────┴────┘
           ↑
         arr[0]

Problem: The array is declared with size 4.
         If you need a 5th element — you're stuck.
         You'd have to declare a new array and copy everything manually.
```

**A dynamic array (Vector):**
```cpp
Vector v;
v.push_back(10); v.push_back(20); v.push_back(30);

Vector object (on STACK — just 3 variables):
  arr      ───────────────────────────────────────────────┐
  capacity = 4                                            │
  size     = 3                                            ↓
                                              [HEAP memory]
                                    Address: 5000 5004 5008 5012
                                             ┌────┬────┬────┬────┐
                                             │ 10 │ 20 │ 30 │ __ │
                                             └────┴────┴────┴────┘

The Vector object itself is small (3 variables).
The actual data lives on the HEAP via the arr pointer.
When the heap array gets full, we allocate a NEW bigger heap array,
copy everything there, and free the old one.
The user sees none of this — they just push_back().
```

**Why heap and not stack?**
```
Stack memory:  Fast but limited (~1-8 MB). Size must be known at compile time.
Heap memory:   Larger (GBs available). Size can be decided at runtime.

A dynamic array MUST use heap memory because:
1. We don't know the size at compile time.
2. We need to reallocate when resizing — impossible on the stack.
```

**Contiguous memory = cache speed:**
```
Vector elements sit side by side in memory:
[10][20][30][40][50]  ← all adjacent, one memory "block"
 ↑                ↑
&arr[0]          &arr[4]

When the CPU reads arr[0], it automatically loads the next few
addresses into its cache. arr[1], arr[2], etc. are already cached.
This is why iterating a vector is extremely fast.

Linked List:
[10] →→→ [20] →→→ [30]
    scattered in memory — cache misses on every step.
```

---

## Structure Definition

```cpp
#include <stdexcept>   // for std::out_of_range, std::underflow_error
using namespace std;

class Vector {
private:
    int* arr;        // Pointer to heap-allocated array of ints
    int  capacity;   // Total number of slots allocated on the heap
    int  current;    // Number of elements actually stored (= size)

public:
    Vector();                          // Constructor
    ~Vector();                         // Destructor

    void push_back(int data);          // Add element at end
    void pop_back();                   // Remove last element
    int  at(int index);                // Safe access (bounds check)
    int  operator[](int index);        // Fast access (no bounds check)
    int  size();                       // Number of stored elements
    int  getCapacity();                // Total allocated space
    bool empty();                      // True if size == 0
    int  front();                      // First element (safe, throws if empty)
    int  back();                       // Last element  (safe, throws if empty)
    void clear();                      // Reset size to 0
    void resize(int newSize);          // Change size, fill new slots with 0
    void reserve(int newCapacity);     // Pre-allocate capacity
    void insert(int index, int val);   // Insert at position, shift right
};
```

**Member Variables Explained:**
```
arr      → Points to an array on the heap.
           Elements live at arr[0], arr[1], ..., arr[current-1].
           arr[current] onwards is uninitialized/garbage — never access it.

capacity → The number of int slots currently allocated on the heap.
           When current == capacity, the next push_back triggers a resize.
           Starts at 1; doubles on every resize: 1 → 2 → 4 → 8 → 16...

current  → The number of valid elements in the array.
           Also called "size". Indices 0 to current-1 are valid.
           Starts at 0 (empty vector).
```

**The relationship at any moment:**
```
Invariant: 0 ≤ current ≤ capacity

current = 0              → empty vector
current = capacity       → full, next push_back will resize
current < capacity       → has room, next push_back is O(1)
current > capacity       → IMPOSSIBLE (bug in our code if this happens)
```

---

## Operations

### Operation 1: Constructor

**Purpose:** Create an empty vector with initial capacity of 1.

**Code:**
```cpp
Vector() {
    arr = new int[1];   // Allocate 1 slot on the heap
    capacity = 1;       // Can hold 1 element before resizing
    current = 0;        // No elements stored yet
}
```

**Memory layout after construction:**
```
Vector v;

Stack frame:
  v.arr      ─────────────────────────────────┐
  v.capacity = 1                               │
  v.current  = 0                               ↓
                                        [Heap]
                                        ┌────┐
                                        │ __ │  ← 1 slot allocated
                                        └────┘
                                         [0]
```

**Why start with capacity = 1 (not 0 or 10)?**
```
capacity = 0: We'd need a special case in push_back
              ("if capacity is 0, allocate 1 first").
              Adds complexity for no real benefit.

capacity = 1: push_back just works — the doubling formula
              handles everything: 1→2→4→8→16...

capacity = 10: Wastes 9 slots for vectors that store only 1-2 items.
               Many vectors are small. Starting at 1 is optimal.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1) — only 1 slot allocated

---

### Operation 2: Push Back (Add to End)

**Purpose:** Append a new element to the end of the vector. Resize if full.

**Code:**
```cpp
void push_back(int data) {
    // Step 1: Is there room?
    if (current == capacity) {
        // No room — allocate a new array of double the size
        int* temp = new int[2 * capacity];

        // Copy all existing elements to the new array
        for (int i = 0; i < current; i++) {
            temp[i] = arr[i];
        }

        // Free the old array
        delete[] arr;

        // Swap in the new array and update capacity
        arr = temp;
        capacity = capacity * 2;
    }

    // Step 2: Write the new element at the next free slot
    arr[current] = data;
    current++;
}
```

**Scenario 1: Room Available — O(1)**
```
State: current=2, capacity=4
┌────┬────┬────┬────┐
│ 10 │ 20 │ __ │ __ │
└────┴────┴────┴────┘
  0    1    2    3

push_back(30):
  current (2) == capacity (4)?  NO → no resize
  arr[2] = 30
  current = 3

┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │
└────┴────┴────┴────┘
              ↑
          current=3
```

**Scenario 2: Full — Resize Triggered — O(n)**
```
State: current=4, capacity=4  ← FULL
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │
└────┴────┴────┴────┘

push_back(50):
  current (4) == capacity (4)?  YES → resize!

  Step 1: Allocate new array, size = 4 * 2 = 8
          ┌────┬────┬────┬────┬────┬────┬────┬────┐
  temp →  │ __ │ __ │ __ │ __ │ __ │ __ │ __ │ __ │
          └────┴────┴────┴────┴────┴────┴────┴────┘

  Step 2: Copy all 4 elements
          ┌────┬────┬────┬────┬────┬────┬────┬────┐
  temp →  │ 10 │ 20 │ 30 │ 40 │ __ │ __ │ __ │ __ │
          └────┴────┴────┴────┴────┴────┴────┴────┘

  Step 3: delete[] arr  ← old array freed from heap
  Step 4: arr = temp, capacity = 8
  Step 5: arr[4] = 50, current = 5

          ┌────┬────┬────┬────┬────┬────┬────┬────┐
  arr →   │ 10 │ 20 │ 30 │ 40 │ 50 │ __ │ __ │ __ │
          └────┴────┴────┴────┴────┴────┴────┴────┘
           0    1    2    3    4    5    6    7
                              current=5, capacity=8
```

**Why double and not just add 1?**
```
This is the key design decision. See Amortized Analysis section for full proof.

Short answer:
  Add-by-1: 1000 insertions → 999 resizes → ~500,000 total copies → O(n²) total
  Double:   1000 insertions → ~10 resizes  → ~2000 total copies   → O(n) total

With doubling, the TOTAL number of copies across all n insertions is at most 2n.
That makes the average (amortized) cost per insertion O(1).
Note: individual early elements can be copied more than twice — it's the
total across ALL insertions that stays within 2n.
```

**Time Complexity:**
```
Best case (no resize):  O(1)
Worst case (resize):    O(n) — must copy all n elements
Amortized:             O(1) — resizes happen so rarely they average out
```
**Space Complexity:** O(n) total for the array

---

### Operation 3: Pop Back (Remove Last Element)

**Purpose:** Remove the last element from the vector.

**Code:**
```cpp
void pop_back() {
    if (current == 0) {
        throw underflow_error("Vector is empty");
    }
    current--;
    // The data at arr[current] is now "dead" — not erased, just ignored
}
```

**Visual:**
```
Before: size=4, capacity=8
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘
                  ↑ current=4

pop_back():
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ __ │ __ │ __ │ __ │  ← 40 still in memory
└────┴────┴────┴────┴────┴────┴────┴────┘
             ↑ current=3  (40 is now "invisible")

at(3) → throws exception  (index 3 >= current 3)
v[3]  → returns 40  ← but this is undefined behavior! Don't do it.
```

**Why just decrement and not actually erase?**
```
Erasing (writing 0):   O(1) but slightly slower — one extra write
Decrementing:          O(1) and minimal work — one subtraction

The value at arr[current] is unreachable through the public API
(at() checks bounds, operator[] relies on user discipline).
The next push_back will overwrite it anyway.

Result: Same effect, zero extra cost.
```

**pop_back() chained — watching the size shrink:**
```
[10][20][30][40]  size=4
pop → size=3   [10][20][30]
pop → size=2   [10][20]
pop → size=1   [10]
pop → size=0   []
pop → underflow_error! (size already 0)
```

**Note: pop_back does NOT shrink capacity.** If you push 100 elements and pop 99, the vector still has capacity for 100+ elements. Capacity only ever grows in our implementation.

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 4: At (Safe Element Access)

**Purpose:** Return the element at a given index, with bounds checking.

**Code:**
```cpp
int at(int index) {
    if (index < 0 || index >= current) {
        throw out_of_range("Index " + to_string(index) +
                           " out of bounds for size " +
                           to_string(current));
    }
    return arr[index];
}
```

**Valid vs invalid indices:**
```
Vector state: size=4, capacity=8
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘
  0    1    2    3    4    5    6    7
  └──────────────┘    └──────────────┘
     valid range           garbage zone
   indices 0 to 3       never access these

at(0)  →  10 ✅
at(2)  →  30 ✅
at(3)  →  40 ✅ (last valid)
at(4)  →  throws out_of_range ❌
at(-1) →  throws out_of_range ❌
at(7)  →  throws out_of_range ❌
```

**Why bounds checking matters:**
```cpp
// Without bounds check (raw array):
int arr[4] = {10, 20, 30, 40};
int x = arr[10];   // UNDEFINED BEHAVIOR — could read anything!
                   // Could crash, could return garbage, could corrupt memory

// With at() — safe:
Vector v;
v.push_back(10);
try {
    int x = v.at(10);   // Throws cleanly — no undefined behavior
} catch (const out_of_range& e) {
    cout << "Caught: " << e.what() << endl;
}
```

**Time Complexity:** O(1) — direct index into array
**Space Complexity:** O(1)

---

### Operation 5: Operator[] (Fast Element Access)

**Purpose:** Return the element at a given index WITHOUT bounds checking.

**Code:**
```cpp
int operator[](int index) {
    return arr[index];
}
```

**How it's used:**
```cpp
Vector v;
v.push_back(10);
v.push_back(20);
v.push_back(30);

cout << v[0] << endl;   // 10 — same syntax as a regular array!
cout << v[1] << endl;   // 20
cout << v[2] << endl;   // 30

// This is why it's called operator overloading:
// We teach the [] operator to work on our Vector class
```

**at() vs operator[] — when to use which:**
```
at(index) → Use when:
  - Index comes from user input or external source (untrusted)
  - Debugging: you want a clean error instead of a crash
  - Safety is more important than speed
  - Any time you're not 100% sure the index is valid

v[index]  → Use when:
  - You've already validated the index
  - Inside tight loops where you know bounds are safe
  - Performance-critical code (no branch = faster)
  - Example: for (int i = 0; i < v.size(); i++) v[i]... → safe ✅

Rule of thumb: Default to at() while learning. Switch to [] only in
hot paths where you've proven safety.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 6: Size

**Purpose:** Return the number of elements currently stored in the vector.

**Code:**
```cpp
int size() {
    return current;
}
```

**Visual:**
```
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘
  └──────────┘
  size() = 3    NOT 8 (that's capacity)
```

**The most common loop pattern:**
```cpp
// Iterate over every valid element — always use size(), not capacity
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}

// Using size() ensures we never touch the garbage zone.
```

**Time Complexity:** O(1) — just returns a variable
**Space Complexity:** O(1)

---

### Operation 7: Get Capacity

**Purpose:** Return the total number of slots currently allocated on the heap.

**Code:**
```cpp
int getCapacity() {
    return capacity;
}
```

**Size vs Capacity — the full picture:**
```
After: push_back 3 times into a freshly constructed vector

  Construction:  capacity=1, size=0
  push_back(10): RESIZE 1→2,  capacity=2,  size=1
  push_back(20): RESIZE 2→4,  capacity=4,  size=2
  push_back(30): no resize,   capacity=4,  size=3

  ┌────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ __ │
  └────┴────┴────┴────┘

  size()        = 3   (3 elements stored)
  getCapacity() = 4   (4 slots allocated)
  "slack"       = 1   (can add 1 more before next resize)
```

**When is getCapacity() useful?**
```cpp
// 1. Understanding memory usage
cout << "Memory used: " << v.getCapacity() * sizeof(int) << " bytes\n";

// 2. Knowing when the next resize will happen
if (v.size() == v.getCapacity()) {
    cout << "Next push_back will trigger a resize!\n";
}

// 3. Deciding whether to reserve in advance
if (v.getCapacity() < needed) {
    v.reserve(needed);
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 8: Empty

**Purpose:** Check if the vector contains zero elements.

**Code:**
```cpp
bool empty() {
    return current == 0;
}
```

**Visual:**
```
Freshly constructed:   current=0   → empty() = TRUE
After push_back(10):   current=1   → empty() = FALSE
After pop_back():      current=0   → empty() = TRUE
After clear():         current=0   → empty() = TRUE
```

**Common patterns using empty():**
```cpp
// Drain a vector
while (!v.empty()) {
    process(v.at(v.size() - 1));   // Look at last element
    v.pop_back();                  // Remove it
}

// Guard against operating on an empty vector
if (!v.empty()) {
    int last = v.at(v.size() - 1);   // Safe: at least one element
}
```

**Why empty() instead of size() == 0?**
```
Both are correct and O(1). empty() is preferred because:
1. It reads as natural English: "if the vector is empty"
2. It's the convention in C++ STL (all containers have .empty())
3. On some containers (like std::list), empty() may be O(1) while
   size() is O(n) — so using empty() is a safe habit across all containers.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 9: Clear

**Purpose:** Remove all elements from the vector (size becomes 0), but keep the allocated capacity.

**Code:**
```cpp
void clear() {
    current = 0;
    // capacity is intentionally NOT reset
    // The heap array stays allocated
}
```

**Visual:**
```
Before clear: size=5, capacity=8
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘

After clear: size=0, capacity=8
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │ __ │ __ │ __ │  ← old data still there
└────┴────┴────┴────┴────┴────┴────┴────┘
                                              but invisible (current=0)
```

**Why keep capacity after clear?**
```
The classic pattern is: fill → process → clear → fill again.

Vector v;
v.reserve(100);

for (int round = 0; round < 10; round++) {
    // Fill
    for (int i = 0; i < 100; i++) v.push_back(i);

    // Process
    for (int i = 0; i < v.size(); i++) process(v[i]);

    // Clear (capacity stays at 100!)
    v.clear();

    // Next fill: NO reallocation needed — capacity already there
}

Without clear() keeping capacity:
  Each round would reallocate → 10 * O(n) = expensive.
With clear() keeping capacity:
  Only the first round allocates → O(n) once, then O(1) clears.
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Operation 10: Resize

**Purpose:** Set the size of the vector to a specific value. Grows (filling with 0) or shrinks.

**Code:**
```cpp
void resize(int newSize) {
    if (newSize < 0) {
        throw invalid_argument("Size cannot be negative");
    }

    if (newSize > capacity) {
        // Need more capacity — reserve it
        reserve(newSize);
    }

    // If growing: fill new slots with 0
    for (int i = current; i < newSize; i++) {
        arr[i] = 0;
    }

    // Set the new size (may grow or shrink)
    current = newSize;
}
```

**Case 1: Growing (newSize > current)**
```
Before: size=3, capacity=4
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ __ │
└────┴────┴────┴────┘

resize(6):
  newSize (6) > capacity (4) → call reserve(6)
    reserve sets capacity = exactly 6 (NOT doubled — reserve sets what you ask for)
  ┌────┬────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ __ │ __ │ __ │
  └────┴────┴────┴────┴────┴────┘

  Fill indices 3, 4, 5 with 0:
  ┌────┬────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │  0 │  0 │  0 │
  └────┴────┴────┴────┴────┴────┘
  current = 6, capacity = 6

size() = 6, v[3]=0, v[4]=0, v[5]=0

Note: push_back() doubles capacity internally.
      reserve() sets capacity to EXACTLY what you pass in.
      resize() calls reserve(newSize), so capacity = newSize exactly.
```

**Case 2: Shrinking (newSize < current)**
```
Before: size=5, capacity=8
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘

resize(3):
  newSize (3) <= capacity (8) → no reserve needed
  newSize (3) < current (5)  → no loop runs (nothing to fill)
  current = 3

┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │ __ │ __ │ __ │  ← 40, 50 still in memory
└────┴────┴────┴────┴────┴────┴────┴────┘
             ↑
           current=3    capacity unchanged at 8
```

**resize() vs reserve():**
```
resize(n):                  reserve(n):
  Changes size              Changes capacity only
  May change capacity       Does NOT change size
  Fills new slots with 0    Does NOT fill anything
  Affects what at() sees    Does NOT affect what at() sees

Example to feel the difference:
  Vector v;
  v.reserve(5);   // capacity=5, size=0
  v[0] = 10;      // ❌ WRONG! size is still 0, undefined behavior

  Vector w;
  w.resize(5);    // capacity=5 (or more), size=5, all zeroes
  w[0] = 10;      // ✅ CORRECT! index 0 is now valid (within size)
```

**Time Complexity:** O(n) if growing beyond capacity (reserve + fill), O(1) if shrinking
**Space Complexity:** O(n) if reallocation needed, O(1) otherwise

---

### Operation 11: Reserve

**Purpose:** Pre-allocate at least `newCapacity` slots so future push_backs don't trigger reallocations.

**Code:**
```cpp
void reserve(int newCapacity) {
    // No point reserving less than what we already have
    if (newCapacity <= capacity) return;

    // Allocate a bigger array
    int* temp = new int[newCapacity];

    // Copy existing elements
    for (int i = 0; i < current; i++) {
        temp[i] = arr[i];
    }

    // Free old array
    delete[] arr;

    // Update
    arr = temp;
    capacity = newCapacity;
    // current stays unchanged — size didn't change
}
```

**Visual:**
```
Before reserve(8): capacity=2, size=2
┌────┬────┐
│ 10 │ 20 │
└────┴────┘

reserve(8):
  8 > 2 → proceed
  Allocate new[8], copy [10, 20], delete old, update arr

After: capacity=8, size=2
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ __ │ __ │ __ │ __ │ __ │ __ │
└────┴────┴────┴────┴────┴────┴────┴────┘
  ↑    ↑    ↑─────────────────────────────┐
 [0]  [1]        6 pre-allocated slots ready for push_back
```

**Performance impact — reserve vs no reserve:**
```
WITHOUT reserve (inserting 1000 elements):
  Resizes: ~10 (capacity: 1→2→4→8→16→32→64→128→256→512→1024)
  Each resize copies all current elements.
  Total copies: 1+2+4+...+512 ≈ 1023 copy operations.
  Plus: 10 allocations, 10 deletions.

WITH reserve(1000) upfront:
  1 allocation (capacity jumps to 1000 immediately)
  0 resizes during the 1000 insertions
  0 copies (no re-copying needed)
  Faster by a significant margin.

Rule: If you know (or can estimate) how many elements you'll add,
      reserve() before the loop.
```

**Common patterns:**
```cpp
// Pattern 1: Known size upfront
int n; cin >> n;
Vector v;
v.reserve(n);                // One allocation
for (int i = 0; i < n; i++) v.push_back(i);  // No resizes

// Pattern 2: Reading file of unknown size
Vector v;
v.reserve(1000);             // Good first guess
// push_back as needed; will only resize if we exceed 1000

// Pattern 3: reserve after knowing size from another structure
Vector result;
result.reserve(source.size());  // Match source capacity
```

**Time Complexity:** O(n) — must copy all existing elements to new array
**Space Complexity:** O(n)

---

### Operation 12: Insert at Position

**Purpose:** Insert a new value at a specific index, shifting existing elements right to make room.

**Code:**
```cpp
void insert(int index, int val) {
    // index must be in [0, current] — inserting AT current == push_back
    if (index < 0 || index > current) {
        throw out_of_range("Insert index out of bounds");
    }

    // Grow if needed (same logic as push_back)
    if (current == capacity) {
        int* temp = new int[2 * capacity];
        for (int i = 0; i < current; i++) {
            temp[i] = arr[i];
        }
        delete[] arr;
        arr = temp;
        capacity *= 2;
    }

    // Shift elements RIGHT from index onward to make a gap
    // CRITICAL: shift from the END backwards to avoid overwriting
    for (int i = current; i > index; i--) {
        arr[i] = arr[i - 1];
    }

    // Place the new element in the gap
    arr[index] = val;
    current++;
}
```

**Detailed Step-by-Step:**

```
Before: [10][20][30][40][__][__]
         0   1   2   3
         current=4, capacity=6

insert(2, 25):   ← insert(index=2, val=25)  — index FIRST, value SECOND

  Step 1: Bounds check — 0 ≤ 2 ≤ 4  ✓
  Step 2: Capacity check — 4 < 6, no resize needed

  Step 3: Shift right, starting from the END (i=4 down to i=3):

    i=4: arr[4] = arr[3]  →  [10][20][30][40][40][__]
    i=3: arr[3] = arr[2]  →  [10][20][30][30][40][__]
    i=2: STOP (loop condition: i > index=2)

    Now index 2 is the "gap":
    [10][20][GAP][30][40][__]

  Step 4: arr[2] = 25
    [10][20][25][30][40][__]

  Step 5: current = 5
```

**Why shift from the END, not the beginning?**
```
WRONG — shifting left to right (overwrites data!):
[10][20][30][40][__]
i=2: arr[2] = arr[1]  → [10][20][20][40][__]   (30 is GONE!)
i=3: arr[3] = arr[2]  → [10][20][20][20][__]   (still wrong)

CORRECT — shifting right to left (preserves data):
[10][20][30][40][__]
i=4: arr[4] = arr[3]  → [10][20][30][40][40]   (40 saved)
i=3: arr[3] = arr[2]  → [10][20][30][30][40]   (30 saved)
i=2: gap is at index 2

The rule: when shifting right, always start from the rightmost element.
```

**Special cases — insert at boundaries:**
```
insert at index = 0 (worst case):
  Must shift ALL elements right by 1
  [10][20][30] → insert 5 at 0 → [5][10][20][30]
  Shifted: 3 elements → O(n)

insert at index = current (same as push_back):
  No shifting needed! The loop doesn't execute.
  [10][20][30] → insert 40 at 3 → [10][20][30][40]
  Shifted: 0 elements → O(1) (ignoring possible resize)

insert in middle:
  [10][20][30][40] → insert 25 at 2 → [10][20][25][30][40]
  Shifted: current - index elements → O(n/2) = O(n)
```

**Time Complexity:** O(n) — shifting elements dominates
**Space Complexity:** O(1), or O(n) if resize triggered

---

### Operation 13: Destructor

**Purpose:** Free the heap-allocated array when the Vector object is destroyed.

**Code:**
```cpp
~Vector() {
    delete[] arr;   // Free the heap array
    // capacity and current are on the stack — freed automatically
}
```

**The memory lifecycle:**
```
{
    Vector v;            // arr = new int[1] — heap ALLOCATED
    v.push_back(10);     // possible resize — new heap block
    v.push_back(20);
    v.push_back(30);
}  ← v goes out of scope here

~Vector() is called automatically:
    delete[] arr     → heap memory FREED

Without destructor: the heap array leaks forever (until program exits).
```

**How serious is a memory leak?**
```cpp
// Every call to this function leaks memory
void processData(int n) {
    Vector v;
    for (int i = 0; i < n; i++) v.push_back(i);
    // ... process v ...
}  // WITHOUT destructor: n * sizeof(int) bytes leaked each call!

// Called 1000 times with n=1000:
// 1000 * 1000 * 4 bytes = ~4MB leaked

// In a server running for days — eventually crashes with out-of-memory.
```

**`delete` vs `delete[]` — important distinction:**
```cpp
int* x = new int;      // Single object
delete x;              // ✅ Correct

int* arr = new int[5]; // Array
delete[] arr;          // ✅ Correct — MUST use delete[]
delete arr;            // ❌ WRONG — undefined behavior in C++!
                       // new[] and delete[] must always be paired.
                       // Using delete instead of delete[] is undefined behavior:
                       // for non-trivial types (classes with destructors)
                       // it skips destructors for all elements except the first.
                       // For int it may seem to work but is still illegal C++.
                       // Rule: new[]  → delete[]  (always, no exceptions)
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

## C++ STL `std::vector` Quick Reference

In competitive programming and real projects you'll use `std::vector` rather than building your own. Here's how it maps to what you just learned:

```cpp
#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> v;         // Empty vector (any type in <>, not just int)

    // Adding elements
    v.push_back(10);       // Same as our push_back()
    v.push_back(20);
    v.push_back(30);

    // Accessing
    v[1];                  // Same as our operator[] → 20 (no bounds check)
    v.at(1);               // Same as our at()       → 20 (bounds check)
    v.front();             // First element          → 10
    v.back();              // Last element           → 30

    // Size and capacity
    v.size();              // Same as our size()        → 3
    v.capacity();          // Same as our getCapacity() → 4 (implementation defined)
    v.empty();             // Same as our empty()       → false

    // Removing
    v.pop_back();          // Same as our pop_back()    → removes 30

    // Memory control
    v.reserve(100);        // Same as our reserve()     → pre-allocate 100 slots
    v.resize(5);           // Same as our resize()      → size becomes 5
    v.clear();             // Same as our clear()       → size becomes 0

    // Insert (STL version uses iterators)
    v.insert(v.begin() + 1, 99);  // Insert 99 at index 1

    // Iteration (range-based for — cleaner syntax)
    for (int x : v) {
        cout << x << " ";
    }
}
```

**STL name mapping cheatsheet:**

| Our implementation  | `std::vector` equivalent  |
|---------------------|---------------------------|
| `push_back(val)`    | `push_back(val)`          |
| `pop_back()`        | `pop_back()`              |
| `at(index)`         | `at(index)`               |
| `operator[]`        | `operator[]`              |
| `front()`           | `front()`                 |
| `back()`            | `back()`                  |
| `size()`            | `size()`                  |
| `getCapacity()`     | `capacity()`              |
| `empty()`           | `empty()`                 |
| `clear()`           | `clear()`                 |
| `resize(n)`         | `resize(n)`               |
| `reserve(n)`        | `reserve(n)`              |
| `insert(idx, val)`  | `insert(begin()+idx, val)`|
| `~Vector()`         | automatic                 |

**STL extras you get for free (not in our implementation):**
```cpp
v.erase(v.begin() + 2);          // Remove element at index 2
v.erase(v.begin()+1, v.begin()+3); // Remove range [1, 3)
v.shrink_to_fit();                // Release unused capacity
v.assign(5, 0);                  // Set vector to [0, 0, 0, 0, 0]
sort(v.begin(), v.end());         // Sort using <algorithm>
reverse(v.begin(), v.end());      // Reverse in place
```

---

## Amortized Analysis — Why Doubling Works

This section explains **why** the doubling strategy gives O(1) amortized push_back. Understanding this will come up in every DSA interview.

**The problem to explain:**
```
A single push_back that triggers resize costs O(n).
But we say push_back is O(1) amortized. How?
```

**Strategy: The "banker" model (Accounting Method)**
```
Imagine each push_back "pays" a cost of 3 tokens:
  1 token → for placing the element into the array (direct cost)
  2 tokens → "saved up" for when THIS element must be copied in a future resize

When a resize happens, each element being copied uses its 2 saved tokens.
Can we always afford to copy? Let's check.
```

**Trace with capacity doubling:**
```
Start: capacity=1, size=0

push_back(A): No resize. Cost=1. Saves 2. Total spent=1.
  [A]  capacity=1, size=1

push_back(B): Resize! capacity 1→2. Copy [A]. Cost=1+1(copy A)=2.
  A uses 1 saved token for its copy. Cost=1(push)+1(copy)=2.
  [A][B]  capacity=2, size=2

push_back(C): Resize! capacity 2→4. Copy [A,B]. Cost=1+2(copies)=3.
  [A][B][C]  capacity=4, size=3

push_back(D): No resize.
  [A][B][C][D]  capacity=4, size=4

push_back(E): Resize! capacity 4→8. Copy [A,B,C,D]. Cost=1+4=5.
  [A][B][C][D][E]  capacity=8, size=5
```

**The key math:**
```
After n insertions with doubling:
  Number of resizes: log₂(n)
  Copies at each resize:  1, 2, 4, 8, ..., n/2
  Total copies = 1 + 2 + 4 + ... + n/2 = n - 1  ← geometric series

Total work for n push_backs:
  n insertions + (n-1) copies ≈ 2n operations
  Per-push_back cost: 2n / n = 2 = O(1) amortized ✓

If we used add-by-1 instead:
  Copies at each resize: 1, 2, 3, ..., n-1
  Total copies = n(n-1)/2 ≈ n²/2
  Per-push_back: n/2 = O(n) amortized → terrible!
```

**Visual of total work:**
```
n = 8 insertions with doubling:

Resize at:    1    2    4    8
Copies:       1    2    4    (never, capacity = 8)
              ↑    ↑    ↑
              A   A,B  A,B,C,D

Total copies: 1 + 2 + 4 = 7 < 2*8 = 16

Total operations: 8 inserts + 7 copies = 15
Average per insert: 15/8 ≈ 1.875 ≈ O(1) ✓
```

**Why specifically double and not triple or 1.5x?**
```
Growth factor 1.5x: Slightly less memory waste, more resizes — still O(1) amortized.
Growth factor 2x:   Balance between speed and memory.
Growth factor 3x:   Fewer resizes, but more memory wasted.

The math works for any factor > 1. Most implementations use 2x.
C++ compilers often use 1.5x or 2x depending on the platform.
```

---

## Common Patterns

> **Note:** All patterns below use the C++ STL `std::vector<int>`, not our custom `Vector`
> class. The operations (`push_back`, `pop_back`, `size`, `empty`, `reserve`, `resize`,
> `at`, `operator[]`) work identically. The STL version also has `front()` and `back()`
> (which our custom class does not — see the Complete Implementation for those additions).

### Pattern 1: Build and Iterate
```cpp
// The most common usage — append, then read
#include <vector>
using namespace std;

vector<int> v;
int n; cin >> n;

for (int i = 0; i < n; i++) {
    int x; cin >> x;
    v.push_back(x);
}

// Iterate forward
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}

// Range-based for (cleaner, preferred in modern C++)
for (int x : v) {
    cout << x << " ";
}
```

### Pattern 2: Reserve for Performance
```cpp
// When you know (or can estimate) the number of elements
int n = 1000000;
vector<int> v;
v.reserve(n);        // One allocation upfront

for (int i = 0; i < n; i++) {
    v.push_back(i);  // Zero resizes!
}
```

### Pattern 3: Use as a Stack
```cpp
// Vectors are commonly used as stacks — push_back = push, pop_back = pop
vector<int> stk;

stk.push_back(10);   // push
stk.push_back(20);
stk.push_back(30);

while (!stk.empty()) {
    cout << stk.back() << " ";  // peek top  → 30 20 10
    stk.pop_back();              // pop
}
```

### Pattern 4: 2D Vector (Matrix)
```cpp
// A vector of vectors — like a 2D array, but dynamic
int rows = 3, cols = 4;
vector<vector<int>> matrix(rows, vector<int>(cols, 0));

// Initialize to identity matrix
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        if (i == j) matrix[i][j] = 1;
    }
}

// Access: matrix[row][col]
cout << matrix[1][1] << endl;  // 1
```

### Pattern 5: Filter into a new vector
```cpp
// Keep only elements satisfying a condition
vector<int> src = {1, 5, 2, 8, 3, 9, 4};
vector<int> result;

for (int x : src) {
    if (x > 4) result.push_back(x);  // Collect elements greater than 4
}
// result = [5, 8, 9]
```

---

## Complete Implementation

```cpp
#include <iostream>
#include <stdexcept>
#include <string>
using namespace std;

class Vector {
private:
    int* arr;
    int  capacity;
    int  current;

public:
    // Constructor
    Vector() {
        arr = new int[1];
        capacity = 1;
        current = 0;
    }

    // Add element at end
    void push_back(int data) {
        if (current == capacity) {
            int* temp = new int[2 * capacity];
            for (int i = 0; i < current; i++) temp[i] = arr[i];
            delete[] arr;
            arr = temp;
            capacity *= 2;
        }
        arr[current++] = data;
    }

    // Remove last element
    void pop_back() {
        if (current == 0) throw underflow_error("Vector is empty");
        current--;
    }

    // Safe access with bounds check
    int at(int index) {
        if (index < 0 || index >= current)
            throw out_of_range("Index " + to_string(index) +
                               " out of bounds for size " + to_string(current));
        return arr[index];
    }

    // Fast access without bounds check
    int operator[](int index) {
        return arr[index];
    }

    // Number of stored elements
    int size() { return current; }

    // Total allocated slots
    int getCapacity() { return capacity; }

    // True if no elements stored
    bool empty() { return current == 0; }

    // First element (safe)
    int front() {
        if (current == 0) throw underflow_error("Vector is empty");
        return arr[0];
    }

    // Last element (safe)
    int back() {
        if (current == 0) throw underflow_error("Vector is empty");
        return arr[current - 1];
    }

    // Reset size to 0, keep capacity
    void clear() { current = 0; }

    // Pre-allocate capacity (does NOT change size)
    void reserve(int newCapacity) {
        if (newCapacity <= capacity) return;
        int* temp = new int[newCapacity];
        for (int i = 0; i < current; i++) temp[i] = arr[i];
        delete[] arr;
        arr = temp;
        capacity = newCapacity;
    }

    // Change size; fill new slots with 0, or truncate
    void resize(int newSize) {
        if (newSize < 0) throw invalid_argument("Size cannot be negative");
        if (newSize > capacity) reserve(newSize);
        for (int i = current; i < newSize; i++) arr[i] = 0;
        current = newSize;
    }

    // Insert at index, shift elements right
    void insert(int index, int val) {
        if (index < 0 || index > current)
            throw out_of_range("Insert index out of bounds");
        if (current == capacity) {
            int* temp = new int[2 * capacity];
            for (int i = 0; i < current; i++) temp[i] = arr[i];
            delete[] arr;
            arr = temp;
            capacity *= 2;
        }
        for (int i = current; i > index; i--) arr[i] = arr[i - 1];
        arr[index] = val;
        current++;
    }

    // Destructor — free heap memory
    ~Vector() {
        delete[] arr;
    }
};
```

---

## Usage Example

```cpp
int main() {
    Vector v;

    // --- Basic push and access ---
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);

    cout << "Size:     " << v.size()        << endl;  // 3
    cout << "Capacity: " << v.getCapacity() << endl;  // 4
    cout << "Empty?    " << (v.empty() ? "Yes" : "No") << endl;  // No

    cout << "v[1]    = " << v[1]       << endl;  // 20 (no bounds check)
    cout << "v.at(2) = " << v.at(2)    << endl;  // 30 (with bounds check)

    // --- Insert in the middle ---
    v.insert(1, 15);  // Insert 15 at index 1
    // v = [10, 15, 20, 30]
    for (int i = 0; i < v.size(); i++) cout << v[i] << " ";
    cout << endl;  // 10 15 20 30

    // --- Pop back ---
    v.pop_back();  // Remove 30
    cout << "After pop: size = " << v.size() << endl;  // 3

    // --- Reserve before heavy insertion ---
    Vector big;
    big.reserve(100);
    cout << "After reserve: capacity = " << big.getCapacity() << ", size = " << big.size() << endl;
    // capacity=100, size=0

    for (int i = 0; i < 100; i++) big.push_back(i);
    cout << "After 100 pushes: size = " << big.size() << ", capacity = " << big.getCapacity() << endl;
    // size=100, capacity=100 (no resizes happened!)

    // --- Resize ---
    Vector r;
    r.push_back(1);
    r.push_back(2);
    r.resize(5);  // Grow to size 5, new slots = 0
    for (int i = 0; i < r.size(); i++) cout << r[i] << " ";
    cout << endl;  // 1 2 0 0 0

    r.resize(2);  // Shrink to size 2
    cout << "After shrink: size = " << r.size() << endl;  // 2

    // --- Clear and reuse ---
    v.clear();
    cout << "After clear: size = " << v.size()        << endl;  // 0
    cout << "After clear: cap  = " << v.getCapacity() << endl;  // still 4!
    v.push_back(99);  // No reallocation needed — capacity is still there
    cout << "Pushed after clear: " << v.at(0) << endl;  // 99

    return 0;
}  // All Vector destructors called automatically here
```

---

## Complexity Summary

| Operation      | Time (Best) | Time (Worst) | Time (Amortized) | Space | Notes                              |
|----------------|-------------|--------------|------------------|-------|------------------------------------|
| Constructor    | O(1)        | O(1)         | O(1)             | O(1)  | Allocates 1 slot                   |
| push_back      | O(1)        | O(n)         | **O(1)**         | O(n)  | O(n) only on resize                |
| pop_back       | O(1)        | O(1)         | O(1)             | O(1)  | Just decrements size               |
| at(index)      | O(1)        | O(1)         | O(1)             | O(1)  | Direct index + bounds check        |
| operator[]     | O(1)        | O(1)         | O(1)             | O(1)  | Direct index, no bounds check      |
| size()         | O(1)        | O(1)         | O(1)             | O(1)  | Returns `current`                  |
| getCapacity()  | O(1)        | O(1)         | O(1)             | O(1)  | Returns `capacity`                 |
| empty()        | O(1)        | O(1)         | O(1)             | O(1)  | Checks current == 0                |
| front()        | O(1)        | O(1)         | O(1)             | O(1)  | Returns arr[0], throws if empty    |
| back()         | O(1)        | O(1)         | O(1)             | O(1)  | Returns arr[current-1], throws if empty |
| clear()        | O(1)        | O(1)         | O(1)             | O(1)  | Resets current to 0                |
| resize(n)      | O(1)        | O(n)         | O(n)*            | O(n)  | O(n) if reallocation or fill needed|
| reserve(n)     | O(1)        | O(n)         | O(n)*            | O(n)  | O(n) copy to new array             |
| insert(idx, v) | O(1)        | O(n)         | O(n)             | O(1)  | O(n) shifting; O(1) insert-at-end  |
| Destructor     | O(1)        | O(1)         | O(1)             | O(1)  | Frees array                        |

> *`resize` and `reserve` are plain O(n) — amortized analysis does not apply to them
> since they are not called in a repeated sequence the way `push_back` is.
> The O(1) best case applies only when n ≤ current capacity (no reallocation needed).

---

## Vector vs Array vs Linked List

| Property              | Static Array       | Dynamic Array (Vector) | Singly Linked List  |
|-----------------------|--------------------|------------------------|---------------------|
| **Size**              | Fixed at compile   | Dynamic (grows/shrinks)| Dynamic             |
| **Memory**            | Stack or global    | Heap (via pointer)     | Heap (nodes)        |
| **Layout**            | Contiguous         | Contiguous             | Scattered           |
| **Random access**     | O(1) ✅             | O(1) ✅                 | O(n) ❌              |
| **Insert at end**     | O(1) (if room)     | O(1) amortized ✅       | O(n) or O(1)*       |
| **Insert at front**   | O(n) ❌             | O(n) ❌                 | O(1) ✅              |
| **Insert at middle**  | O(n)               | O(n)                   | O(n) (find) + O(1)  |
| **Search**            | O(n)               | O(n)                   | O(n)                |
| **Cache efficiency**  | Excellent ✅        | Excellent ✅            | Poor ❌              |
| **Memory overhead**   | None               | Unused capacity        | Pointer per node    |
| **Resize needed?**    | No — impossible    | Automatic              | Not applicable      |

*O(1) at tail only if tail pointer is maintained.

**When to choose each:**
```
Static array:  Size is known at compile time and will never change.
               e.g., chess board [8][8], lookup tables.

Vector:        Default choice for most situations — general purpose.
               Random access needed. Mostly appending to the end.

Linked List:   Frequent insertions/deletions at the HEAD.
               No random access needed.
               Size varies wildly and memory efficiency matters.
```

---

## Common Use Cases

✅ **Use Vector When:**
- You need random access by index → `v[i]` or `v.at(i)`
- You mostly add elements to the end → `push_back()` amortized O(1)
- You'll iterate through all elements (cache-friendly loops)
- The size is not known at compile time
- You need a dynamically-sized 1D or 2D array
- You want to use it as a **stack** (`push_back` + `pop_back` + `back()`)
- Sorting or searching with standard library algorithms

❌ **Don't Use Vector When:**
- You frequently insert or remove from the **beginning or middle**
  (use `std::list` or `std::deque` instead)
- You need **O(1) insertion at the front** (use `std::deque`)
- Memory is extremely tight and you cannot afford unused capacity
- You need a fixed-size array known at compile time (use `int arr[N]`)

---

## Testing Checklist

- [ ] push_back until resize triggered — verify size and capacity both correct
- [ ] pop_back until empty — verify underflow error thrown at zero
- [ ] at() with valid index → returns correct element
- [ ] at() with negative index → throws out_of_range
- [ ] at() with index == size → throws out_of_range
- [ ] operator[] at valid index → returns correct element
- [ ] empty() on new vector → true; after push_back → false
- [ ] front() returns first element; throws on empty vector
- [ ] back() returns last element; throws on empty vector
- [ ] clear() → size = 0, capacity unchanged, can push_back again
- [ ] reserve(n) → capacity = n, size unchanged
- [ ] reserve(n) where n < current capacity → no change
- [ ] resize(grow): new elements are 0
- [ ] resize(shrink): elements beyond new size gone; capacity unchanged
- [ ] insert at index 0 → all elements shifted correctly
- [ ] insert at index == size → same as push_back
- [ ] insert triggers resize correctly
- [ ] Destructor: no memory leak (check with valgrind)

---

## Key Takeaways

1. **Two numbers matter:** `size` (elements stored) and `capacity` (space allocated). Never confuse them.
2. **Doubling = amortized O(1):** The cost of resize is "spread" across all insertions — each element effectively costs 2 operations on average.
3. **Heap memory:** The array lives on the heap. The Vector object on the stack just holds a pointer, size, and capacity.
4. **pop_back and clear are O(1):** No deletion happens — the size counter is simply decremented. Data sits in memory until overwritten.
5. **reserve() before bulk inserts:** If you know the count, reserve it. Eliminates all resize overhead.
6. **resize() vs reserve():** resize changes `size` (and initializes new slots); reserve changes only `capacity`.
7. **at() vs operator[]:** at() is safe (throws); [] is fast (undefined behavior on bad index). Default to at() while learning.
8. **insert() is O(n):** Because elements must be shifted. Avoid inserting at index 0 in a loop — it's O(n²).
9. **Contiguous memory:** Vector's biggest advantage over linked list. Cache-friendliness makes iteration 5–10x faster in practice.
10. **std::vector is your default:** In real code, use `std::vector<T>`. Understanding the internals (which you just did) makes you a better programmer, not just a better exam-taker.

---

*Print this document and keep it beside you while implementing the Vector from scratch.*


# Data Structures and Algorithms in Java

This guide provides an in-depth overview of Data Structures and Algorithms (DSA) in Java. It includes code examples, comparison tables, and links to additional resources for further study.

## Table of Contents

1. [Introduction to DSA](#introduction-to-dsa)
2. [Arrays](#arrays)
3. [Linked Lists](#linked-lists)
4. [Stacks and Queues](#stacks-and-queues)
5. [Trees](#trees)
6. [Heaps](#heaps)
7. [Graphs](#graphs)
8. [Sorting Algorithms](#sorting-algorithms)
9. [Searching Algorithms](#searching-algorithms)
10. [Dynamic Programming](#dynamic-programming)
11. [Additional Resources](#additional-resources)

## Introduction to DSA

Data Structures and Algorithms (DSA) form the foundation of computer science and software development. Understanding DSA is crucial for writing efficient code and solving complex problems.

### Why Learn DSA?

- **Efficiency**: DSA helps in writing optimized code.
- **Problem Solving**: Essential for solving complex coding problems.
- **Competitive Programming**: A must for participating in coding competitions.
- **Interview Preparation**: Most technical interviews heavily focus on DSA concepts.

## Arrays

An array is a collection of elements, identified by index or key.

### Key Operations
- **Accessing**: `O(1)`
- **Insertion**: `O(n)` (in worst case)
- **Deletion**: `O(n)` (in worst case)

### Demo Code

```java
int[] arr = new int[5];
arr[0] = 10; // Accessing and setting values
System.out.println(arr[0]); // Outputs: 10
```

### Pros and Cons of Arrays

| Pros | Cons |
|------|------|
| Fast access to elements by index (`O(1)`). | Fixed size, not dynamic. |
| Simple and easy to use. | Insertion and deletion operations are costly (`O(n)`). |
| Memory is allocated in a contiguous block. | Inefficient use of memory if the array is sparsely populated. |

### Resources for Further Study

- [Java Arrays Documentation](https://docs.oracle.com/javase/tutorial/java/nutsandbolts/arrays.html)
- [GeeksforGeeks - Arrays](https://www.geeksforgeeks.org/array-data-structure/)

## Linked Lists

A linked list is a linear data structure where each element is a separate object, known as a node. Each node contains data and a reference to the next node.

### Types of Linked Lists
- **Singly Linked List**: Each node points to the next node.
- **Doubly Linked List**: Each node points to both the next and previous nodes.
- **Circular Linked List**: Last node points to the first node.

### Demo Code

```java
class Node {
    int data;
    Node next;

    Node(int data) {
        this.data = data;
        this.next = null;
    }
}

class LinkedList {
    Node head;

    void insert(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
        } else {
            Node temp = head;
            while (temp.next != null) {
                temp = temp.next;
            }
            temp.next = newNode;
        }
    }
}
```

### Pros and Cons of Linked Lists

| Pros | Cons |
|------|------|
| Dynamic size, can grow and shrink as needed. | No random access of elements. |
| Efficient insertions/deletions (especially at the beginning). | Requires more memory than arrays due to storage of pointers. |
| No need to know the size beforehand. | Sequential access is slow (`O(n)`). |

### Resources for Further Study

- [Java LinkedList Documentation](https://docs.oracle.com/javase/8/docs/api/java/util/LinkedList.html)
- [GeeksforGeeks - Linked Lists](https://www.geeksforgeeks.org/data-structures/linked-list/)

## Stacks and Queues

### Stack

A stack is a linear data structure that follows the Last In, First Out (LIFO) principle.

### Demo Code

```java
import java.util.Stack;

Stack<Integer> stack = new Stack<>();
stack.push(10);
stack.push(20);
System.out.println(stack.pop()); // Outputs: 20
```

### Pros and Cons of Stacks

| Pros | Cons |
|------|------|
| Simple and intuitive. | Limited to LIFO access, which can be restrictive. |
| Useful in recursive algorithms, backtracking problems. | Not suitable for accessing elements in any order. |

### Queue

A queue is a linear data structure that follows the First In, First Out (FIFO) principle.

### Demo Code

```java
import java.util.LinkedList;
import java.util.Queue;

Queue<Integer> queue = new LinkedList<>();
queue.add(10);
queue.add(20);
System.out.println(queue.remove()); // Outputs: 10
```

### Pros and Cons of Queues

| Pros | Cons |
|------|------|
| Simple and intuitive. | Limited to FIFO access, which can be restrictive. |
| Useful in scenarios like scheduling, buffering, and breadth-first search. | Not suitable for accessing elements in any order. |

### Resources for Further Study

- [Java Stack Documentation](https://docs.oracle.com/javase/8/docs/api/java/util/Stack.html)
- [Java Queue Documentation](https://docs.oracle.com/javase/8/docs/api/java/util/Queue.html)
- [GeeksforGeeks - Stacks and Queues](https://www.geeksforgeeks.org/stack-data-structure/)




## Trees

A tree is a hierarchical data structure that consists of nodes connected by edges. The top node is called the root, and each node contains data and references to its child nodes.

### Types of Trees
- **Binary Tree**: Each node has at most two children.
- **Binary Search Tree (BST)**: A binary tree with the property that the left child is smaller, and the right child is greater than the parent node.
- **AVL Tree**: A self-balancing binary search tree.
- **Red-Black Tree**: A balanced binary search tree with additional properties to ensure balance.

### Binary Search Tree Demo Code

```java
class Node {
    int data;
    Node left, right;

    public Node(int item) {
        data = item;
        left = right = null;
    }
}

class BinarySearchTree {
    Node root;

    BinarySearchTree() {
        root = null;
    }

    void insert(int data) {
        root = insertRec(root, data);
    }

    Node insertRec(Node root, int data) {
        if (root == null) {
            root = new Node(data);
            return root;
        }
        if (data < root.data)
            root.left = insertRec(root.left, data);
        else if (data > root.data)
            root.right = insertRec(root.right, data);
        return root;
    }

    void inorder() {
        inorderRec(root);
    }

    void inorderRec(Node root) {
        if (root != null) {
            inorderRec(root.left);
            System.out.print(root.data + " ");
            inorderRec(root.right);
        }
    }
}
```

### Pros and Cons of Trees

| Pros | Cons |
|------|------|
| Hierarchical data storage. | Complex to implement and maintain. |
| Fast search, insertion, and deletion in balanced trees. | Can become unbalanced, leading to poor performance. |
| Used in databases, file systems, and more. | Requires more memory than linear data structures. |

### Resources for Further Study

- [GeeksforGeeks - Trees](https://www.geeksforgeeks.org/binary-tree-data-structure/)
- [Binary Search Tree in Java](https://www.baeldung.com/java-binary-tree)

## Heaps

A heap is a special tree-based data structure that satisfies the heap property: in a max heap, each parent node is greater than or equal to its children, while in a min heap, each parent node is less than or equal to its children.

### Types of Heaps
- **Max Heap**: The largest element is at the root.
- **Min Heap**: The smallest element is at the root.

### Max Heap Demo Code

```java
class MaxHeap {
    private int[] heap;
    private int size;
    private int capacity;

    public MaxHeap(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        this.heap = new int[capacity];
    }

    public void insert(int element) {
        if (size == capacity) {
            resize();
        }
        heap[size] = element;
        size++;
        heapifyUp(size - 1);
    }

    public int extractMax() {
        if (size == 0) {
            throw new IllegalStateException("Heap is empty");
        }
        int max = heap[0];
        heap[0] = heap[size - 1];
        size--;
        heapifyDown(0);
        return max;
    }

    private void heapifyUp(int index) {
        int parentIndex = (index - 1) / 2;
        while (index > 0 && heap[index] > heap[parentIndex]) {
            swap(index, parentIndex);
            index = parentIndex;
            parentIndex = (index - 1) / 2;
        }
    }

    private void heapifyDown(int index) {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int largest = index;

        if (leftChild < size && heap[leftChild] > heap[largest]) {
            largest = leftChild;
        }
        if (rightChild < size && heap[rightChild] > heap[largest]) {
            largest = rightChild;
        }
        if (largest != index) {
            swap(index, largest);
            heapifyDown(largest);
        }
    }

    private void swap(int i, int j) {
        int temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }

    private void resize() {
        capacity *= 2;
        int[] newHeap = new int[capacity];
        System.arraycopy(heap, 0, newHeap, 0, size);
        heap = newHeap;
    }

    public void printHeap() {
        for (int i = 0; i < size; i++) {
            System.out.print(heap[i] + " ");
        }
        System.out.println();
    }
}
```

### Pros and Cons of Heaps

| Pros | Cons |
|------|------|
| Fast access to the maximum or minimum element. | Not efficient for searching arbitrary elements. |
| Useful in priority queues and scheduling tasks. | Complex to implement and maintain. |

### Resources for Further Study

- [GeeksforGeeks - Heaps](https://www.geeksforgeeks.org/heap-data-structure/)
- [Max Heap in Java](https://www.geeksforgeeks.org/max-heap-in-java/)

## Graphs

A graph is a collection of nodes, called vertices, and edges that connect pairs of vertices. Graphs can be used to model various types of relationships and structures, such as networks, social connections, and more.

### Types of Graphs
- **Directed Graph**: Edges have a direction.
- **Undirected Graph**: Edges have no direction.
- **Weighted Graph**: Edges have weights.
- **Unweighted Graph**: Edges have no weights.

### Graph Representation
- **Adjacency Matrix**: A 2D array where each cell `(i, j)` represents the presence of an edge between vertex `i` and vertex `j`.
- **Adjacency List**: An array of lists where each list represents the neighbors of a vertex.

### Graph Traversal
- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**

### BFS Demo Code

```java
import java.util.*;

class Graph {
    private int vertices;
    private LinkedList<Integer>[] adjList;

    Graph(int vertices) {
        this.vertices = vertices;
        adjList = new LinkedList[vertices];
        for (int i = 0; i < vertices; i++) {
            adjList[i] = new LinkedList<>();
        }
    }

    void addEdge(int source, int destination) {
        adjList[source].add(destination);
    }

    void BFS(int startVertex) {
        boolean[] visited = new boolean[vertices];
        LinkedList<Integer> queue = new LinkedList<>();

        visited[startVertex] = true;
        queue.add(startVertex);

        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");

            for (int adjVertex : adjList[vertex]) {
                if (!visited[adjVertex]) {
                    visited[adjVertex] = true;
                    queue.add(adjVertex);
                }
            }
        }
    }
}
```

### Pros and Cons of Graphs

| Pros | Cons |
|------|------|
| Can model complex relationships. | Complex to implement and manage. |
| Useful in network analysis, social networks, and more. | Traversal and pathfinding can be computationally expensive. |

### Resources for Further Study

- [GeeksforGeeks - Graph Data Structure](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)
- [Introduction to Graphs in Java](https://www.baeldung.com/cs/graph-data-structure)



## Sorting Algorithms

Sorting algorithms are fundamental to computer science, and they are used to arrange data in a specific order, typically ascending or descending. Here are some of the most common sorting algorithms:

### Bubble Sort

Bubble Sort is a simple comparison-based algorithm where each element is compared with the adjacent element, and they are swapped if they are in the wrong order.

#### Bubble Sort Demo Code

```java
class BubbleSort {
    void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    // Swap arr[j] and arr[j+1]
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}
```

### Quick Sort

Quick Sort is a divide-and-conquer algorithm that selects a 'pivot' element and partitions the array into sub-arrays, which are then sorted independently.

#### Quick Sort Demo Code

```java
class QuickSort {
    int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return (i + 1);
    }

    void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
}
```

### Merge Sort

Merge Sort is a divide-and-conquer algorithm that divides the array into halves, recursively sorts them, and then merges the sorted halves.

#### Merge Sort Demo Code

```java
class MergeSort {
    void merge(int[] arr, int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;
        int[] L = new int[n1];
        int[] R = new int[n2];

        for (int i = 0; i < n1; ++i)
            L[i] = arr[l + i];
        for (int j = 0; j < n2; ++j)
            R[j] = arr[m + 1 + j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }

    void mergeSort(int[] arr, int l, int r) {
        if (l < r) {
            int m = (l + r) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
}
```

### Comparison of Sorting Algorithms

| Algorithm   | Time Complexity (Best) | Time Complexity (Average) | Time Complexity (Worst) | Space Complexity | Stable |
|-------------|------------------------|---------------------------|-------------------------|------------------|--------|
| Bubble Sort | O(n)                    | O(n^2)                    | O(n^2)                  | O(1)             | Yes    |
| Quick Sort  | O(n log n)              | O(n log n)                | O(n^2)                  | O(log n)         | No     |
| Merge Sort  | O(n log n)              | O(n log n)                | O(n log n)              | O(n)             | Yes    |
| Insertion Sort | O(n)                 | O(n^2)                    | O(n^2)                  | O(1)             | Yes    |
| Selection Sort | O(n^2)               | O(n^2)                    | O(n^2)                  | O(1)             | No     |

### Resources for Further Study

- [GeeksforGeeks - Sorting Algorithms](https://www.geeksforgeeks.org/sorting-algorithms/)
- [Sorting Algorithms in Java](https://www.baeldung.com/java-sorting)
- [Visualization of Sorting Algorithms](https://visualgo.net/en/sorting)

## Searching Algorithms

Searching algorithms are designed to retrieve information stored within some data structure. The most basic form is searching in an array.

### Linear Search

Linear Search is the simplest search algorithm. It checks each element of the list sequentially until the desired element is found.

#### Linear Search Demo Code

```java
class LinearSearch {
    int linearSearch(int[] arr, int x) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            if (arr[i] == x)
                return i;
        }
        return -1;
    }
}
```

### Binary Search

Binary Search is a more efficient search algorithm that works on sorted arrays by repeatedly dividing the search interval in half.

#### Binary Search Demo Code

```java
class BinarySearch {
    int binarySearch(int[] arr, int x) {
        int l = 0, r = arr.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == x)
                return m;
            if (arr[m] < x)
                l = m + 1;
            else
                r = m - 1;
        }
        return -1;
    }
}
```

### Comparison of Searching Algorithms

| Algorithm     | Time Complexity (Best) | Time Complexity (Average) | Time Complexity (Worst) | Space Complexity | Sorted Data Required |
|---------------|------------------------|---------------------------|-------------------------|------------------|----------------------|
| Linear Search | O(1)                   | O(n)                      | O(n)                    | O(1)             | No                   |
| Binary Search | O(1)                   | O(log n)                  | O(log n)                | O(1)             | Yes                  |

### Resources for Further Study

- [GeeksforGeeks - Searching Algorithms](https://www.geeksforgeeks.org/searching-algorithms/)
- [Binary Search in Java](https://www.baeldung.com/java-binary-search)
- [Linear vs. Binary Search](https://www.geeksforgeeks.org/difference-between-linear-search-and-binary-search/)

## Hashing

Hashing is a technique used to uniquely identify a specific object from a group of similar objects. It involves the use of hash tables or hash maps.

### Hash Table

A hash table is a data structure that implements an associative array abstract data type, a structure that can map keys to values.

### Hashing Function

The hashing function is a function that converts a given key into a fixed-size integer that serves as the index of the key in the hash table.

#### Hash Table Demo Code

```java
import java.util.*;

class HashTable {
    private final int SIZE = 100;
    private LinkedList[] table;

    public HashTable() {
        table = new LinkedList[SIZE];
        for (int i = 0; i < SIZE; i++) {
            table[i] = new LinkedList<>();
        }
    }

    private int hashFunction(int key) {
        return key % SIZE;
    }

    public void insert(int key) {
        int index = hashFunction(key);
        table[index].add(key);
    }

    public boolean search(int key) {
        int index = hashFunction(key);
        return table[index].contains(key);
    }

    public void delete(int key) {
        int index = hashFunction(key);
        table[index].remove((Integer) key);
    }
}
```

### Pros and Cons of Hashing

| Pros | Cons |
|------|------|
| Fast access to elements. | Collisions can degrade performance. |
| Efficient for implementing associative arrays. | Requires a good hash function. |
| Supports dynamic sizing. | Not ordered. |

### Resources for Further Study

- [GeeksforGeeks - Hashing](https://www.geeksforgeeks.org/hashing-data-structure/)
- [Introduction to Hash Tables](https://www.baeldung.com/cs/hash-table)
- [Hashing and Collision Handling](https://www.geeksforgeeks.org/hashing-set-2-separate-chaining/)

## Dynamic Programming

Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems and solving them just once, storing their solutions.

### Common Dynamic Programming Problems

- **Fibonacci Sequence**
- **Knapsack Problem**
- **Longest Common Subsequence**
- **Matrix Chain Multiplication**


### Fibonacci Sequence Demo Code

Here's a basic implementation of the Fibonacci sequence using dynamic programming in Java:

```java
public class Fibonacci {

    public static void main(String[] args) {
        int n = 10; // Number of terms
        System.out.println("Fibonacci sequence up to " + n + " terms:");
        for (int i = 0; i < n; i++) {
            System.out.print(fib(i) + " ");
        }
    }

    // Function to return the nth Fibonacci number
    public static int fib(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 0;
        if (n > 0) {
            dp[1] = 1;
        }
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

### Knapsack Problem Demo Code

The Knapsack problem is a classic example of dynamic programming. Here's a basic implementation:

```java
public class Knapsack {

    public static void main(String[] args) {
        int[] values = {60, 100, 120}; // Values of items
        int[] weights = {10, 20, 30}; // Weights of items
        int capacity = 50; // Knapsack capacity

        System.out.println("Maximum value in Knapsack = " + knapSack(capacity, weights, values));
    }

    // Function to find the maximum value of items that can be put in the knapsack
    public static int knapSack(int capacity, int[] weights, int[] values) {
        int n = values.length;
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }
}
```

### Longest Common Subsequence Demo Code

The Longest Common Subsequence (LCS) problem can be solved efficiently using dynamic programming:

```java
public class LCS {

    public static void main(String[] args) {
        String s1 = "ABCBDAB";
        String s2 = "BDCAB";

        System.out.println("Length of LCS is " + lcs(s1, s2));
    }

    // Function to find the length of the longest common subsequence
    public static int lcs(String s1, String s2) {
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                } else if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
}
```

### Matrix Chain Multiplication Demo Code

Here's a Java implementation for solving the Matrix Chain Multiplication problem:

```java
public class MatrixChainMultiplication {

    public static void main(String[] args) {
        int[] p = {10, 20, 30, 40, 30}; // Dimensions of matrices

        System.out.println("Minimum number of multiplications is " + matrixChainOrder(p));
    }

    // Function to find the minimum number of scalar multiplications needed
    public static int matrixChainOrder(int[] p) {
        int n = p.length;
        int[][] dp = new int[n][n];

        for (int len = 2; len < n; len++) {
            for (int i = 1; i < n - len + 1; i++) {
                int j = i + len - 1;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i; k <= j - 1; k++) {
                    int q = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                    if (q < dp[i][j]) {
                        dp[i][j] = q;
                    }
                }
            }
        }
        return dp[1][n - 1];
    }
}
```

## Additional Resources

- [Dynamic Programming Overview - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
- [Introduction to Dynamic Programming - Brilliant.org](https://brilliant.org/wiki/dynamic-programming/)
- [Dynamic Programming - MIT OpenCourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-notes/MIT6_006F11_Lec07.pdf)



## Graph Algorithms

Graph algorithms are used to solve problems related to graph theory, including traversal, shortest paths, and network flow.

### Common Graph Algorithms

- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Dijkstra's Algorithm**
- **Bellman-Ford Algorithm**
- **Floyd-Warshall Algorithm**
- **Kruskal's Algorithm**
- **Prim's Algorithm**

### Depth-First Search (DFS) Demo Code

DFS explores as far as possible along each branch before backtracking. Here’s an implementation using an adjacency list:

```java
import java.util.*;

public class DFS {

    private Map<Integer, List<Integer>> graph = new HashMap<>();
    private Set<Integer> visited = new HashSet<>();

    public static void main(String[] args) {
        DFS dfs = new DFS();
        dfs.addEdge(1, 2);
        dfs.addEdge(1, 3);
        dfs.addEdge(2, 4);
        dfs.addEdge(2, 5);
        dfs.addEdge(3, 6);
        dfs.addEdge(3, 7);
        System.out.println("DFS starting from node 1:");
        dfs.dfs(1);
    }

    public void addEdge(int u, int v) {
        graph.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
        graph.computeIfAbsent(v, k -> new ArrayList<>()).add(u); // For undirected graph
    }

    public void dfs(int node) {
        if (visited.contains(node)) {
            return;
        }
        System.out.print(node + " ");
        visited.add(node);
        for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
            dfs(neighbor);
        }
    }
}
```

### Breadth-First Search (BFS) Demo Code

BFS explores nodes level by level. Here’s an implementation using an adjacency list:

```java
import java.util.*;

public class BFS {

    private Map<Integer, List<Integer>> graph = new HashMap<>();

    public static void main(String[] args) {
        BFS bfs = new BFS();
        bfs.addEdge(1, 2);
        bfs.addEdge(1, 3);
        bfs.addEdge(2, 4);
        bfs.addEdge(2, 5);
        bfs.addEdge(3, 6);
        bfs.addEdge(3, 7);
        System.out.println("BFS starting from node 1:");
        bfs.bfs(1);
    }

    public void addEdge(int u, int v) {
        graph.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
        graph.computeIfAbsent(v, k -> new ArrayList<>()).add(u); // For undirected graph
    }

    public void bfs(int start) {
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        visited.add(start);
        queue.add(start);

        while (!queue.isEmpty()) {
            int node = queue.poll();
            System.out.print(node + " ");
            for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.add(neighbor);
                }
            }
        }
    }
}
```

### Dijkstra's Algorithm Demo Code

Dijkstra’s algorithm finds the shortest path from a source to all other nodes in a weighted graph:

```java
import java.util.*;

public class Dijkstra {

    public static void main(String[] args) {
        int[][] graph = {
            {0, 7, 9, 0, 0, 14},
            {7, 0, 10, 15, 0, 0},
            {9, 10, 0, 11, 0, 2},
            {0, 15, 11, 0, 6, 0},
            {0, 0, 0, 6, 0, 9},
            {14, 0, 2, 0, 9, 0}
        };
        dijkstra(graph, 0);
    }

    public static void dijkstra(int[][] graph, int start) {
        int n = graph.length;
        int[] dist = new int[n];
        boolean[] sptSet = new boolean[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[]{start, 0});

        while (!pq.isEmpty()) {
            int[] u = pq.poll();
            int uIndex = u[0];
            if (sptSet[uIndex]) {
                continue;
            }
            sptSet[uIndex] = true;

            for (int v = 0; v < n; v++) {
                if (graph[uIndex][v] != 0 && !sptSet[v]) {
                    int newDist = dist[uIndex] + graph[uIndex][v];
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        pq.add(new int[]{v, dist[v]});
                    }
                }
            }
        }

        System.out.println("Vertex Distance from Source");
        for (int i = 0; i < n; i++) {
            System.out.println(i + " \t\t " + dist[i]);
        }
    }
}
```

### Bellman-Ford Algorithm Demo Code

The Bellman-Ford algorithm handles graphs with negative weights:

```java
public class BellmanFord {

    public static void main(String[] args) {
        int V = 5; // Number of vertices
        int E = 8; // Number of edges
        int[][] edges = {
            {0, 1, -1},
            {0, 2, 4},
            {1, 2, 3},
            {1, 3, 2},
            {1, 4, 2},
            {3, 2, 5},
            {3, 1, 1},
            {4, 3, -3}
        };
        bellmanFord(V, E, edges, 0);
    }

    public static void bellmanFord(int V, int E, int[][] edges, int src) {
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;

        for (int i = 1; i < V; i++) {
            for (int[] edge : edges) {
                int u = edge[0];
                int v = edge[1];
                int weight = edge[2];
                if (dist[u] != Integer.MAX_VALUE && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int weight = edge[2];
            if (dist[u] != Integer.MAX_VALUE && dist[u] + weight < dist[v]) {
                System.out.println("Graph contains negative weight cycle");
                return;
            }
        }

        System.out.println("Vertex Distance from Source");
        for (int i = 0; i < V; i++) {
            System.out.println(i + " \t\t " + dist[i]);
        }
    }
}
```

### Floyd-Warshall Algorithm Demo Code

Floyd-Warshall algorithm finds shortest paths between all pairs of vertices:

```java
public class FloydWarshall {

    public static void main(String[] args) {
        int V = 4;
        int[][] graph = {
            {0, 3, INF, INF},
            {2, 0, INF, INF},
            {INF, 7, 0, 1},
            {6, INF, INF, 0}
        };
        floydWarshall(graph, V);
    }

    static final int INF = 99999;

    public static void floydWarshall(int[][] graph, int V) {
        int[][] dist = new int[V][V];

        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }

        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        System.out.println("Shortest distance matrix:");
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == INF) {
                    System.out.print("INF ");
                } else {
                    System.out.print(dist[i][j] + " ");
                }
            }
            System.out.println();
        }
    }
}
```

### Kruskal's Algorithm Demo Code

Kruskal's algorithm finds the Minimum Spanning Tree (MST) of a graph:

```java
import java.util.*;

public class Kruskal {

    public static void main(String[] args) {
        int V = 4;
        int E = 5;
        Edge[] edges =

 new Edge[E];
        edges[0] = new Edge(0, 1, 10);
        edges[1] = new Edge(0, 2, 6);
        edges[2] = new Edge(0, 3, 5);
        edges[3] = new Edge(1, 3, 15);
        edges[4] = new Edge(2, 3, 4);

        kruskal(V, E, edges);
    }

    static class Edge {
        int src, dest, weight;

        Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
    }

    public static void kruskal(int V, int E, Edge[] edges) {
        Arrays.sort(edges, Comparator.comparingInt(e -> e.weight));

        int[] parent = new int[V];
        int[] rank = new int[V];
        for (int i = 0; i < V; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        int mstWeight = 0;
        System.out.println("Edges in MST:");

        for (Edge edge : edges) {
            int root1 = find(parent, edge.src);
            int root2 = find(parent, edge.dest);

            if (root1 != root2) {
                System.out.println(edge.src + " - " + edge.dest + ": " + edge.weight);
                mstWeight += edge.weight;
                union(parent, rank, root1, root2);
            }
        }

        System.out.println("Weight of MST: " + mstWeight);
    }

    public static int find(int[] parent, int i) {
        if (parent[i] != i) {
            parent[i] = find(parent, parent[i]);
        }
        return parent[i];
    }

    public static void union(int[] parent, int[] rank, int x, int y) {
        int rootX = find(parent, x);
        int rootY = find(parent, y);

        if (rootX != rootY) {
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
}
```

### Prim's Algorithm Demo Code

Prim's algorithm finds the MST of a connected weighted graph:

```java
import java.util.*;

public class Prim {

    public static void main(String[] args) {
        int V = 5;
        int[][] graph = {
            {0, 2, 0, 6, 0},
            {2, 0, 3, 8, 5},
            {0, 3, 0, 0, 7},
            {6, 8, 0, 0, 9},
            {0, 5, 7, 9, 0}
        };
        prim(graph, V);
    }

    public static void prim(int[][] graph, int V) {
        boolean[] inMST = new boolean[V];
        int[] parent = new int[V];
        int[] key = new int[V];

        Arrays.fill(key, Integer.MAX_VALUE);
        key[0] = 0;
        parent[0] = -1;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[]{0, 0});

        while (!pq.isEmpty()) {
            int[] u = pq.poll();
            int uIndex = u[0];

            if (inMST[uIndex]) {
                continue;
            }

            inMST[uIndex] = true;

            for (int v = 0; v < V; v++) {
                if (graph[uIndex][v] != 0 && !inMST[v] && graph[uIndex][v] < key[v]) {
                    key[v] = graph[uIndex][v];
                    parent[v] = uIndex;
                    pq.add(new int[]{v, key[v]});
                }
            }
        }

        System.out.println("Edges in MST:");
        for (int i = 1; i < V; i++) {
            System.out.println(parent[i] + " - " + i + ": " + graph[i][parent[i]]);
        }
    }
}
```

## Additional Resources

- [Graph Algorithms - GeeksforGeeks](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)
- [Introduction to Graph Algorithms - Khan Academy](https://www.khanacademy.org/computing/computer-science/algorithms)
- [Graph Algorithms - MIT OpenCourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-notes/MIT6_006F11_Lec09.pdf)



## Recommended Resources

### Books

- **"Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein**
  - Comprehensive coverage of algorithms and data structures.
  - [Find it on Amazon](https://www.amazon.com/Introduction-Algorithms-4th-Thomas-Cormen/dp/0262033844)

- **"Data Structures and Algorithm Analysis in Java" by Weiss**
  - Focuses on Java implementations and analysis.
  - [Find it on Amazon](https://www.amazon.com/Data-Structures-Algorithm-Analysis-Java/dp/013284737X)

- **"Algorithms" by Robert Sedgewick and Kevin Wayne**
  - Provides a broad range of algorithms with practical implementations.
  - [Find it on Amazon](https://www.amazon.com/Algorithms-4th-Robert-Sedgewick/dp/032157351X)

### Online Courses

- **[Coursera - Data Structures and Algorithm Specialization by University of California, San Diego & National Research University Higher School of Economics](https://www.coursera.org/specializations/data-structures-algorithms)**
  - A series of courses covering fundamental and advanced topics in DSA.

- **[edX - Data Structures and Algorithms by University of California, Irvine](https://www.edx.org/course/data-structures-and-algorithms)**
  - A comprehensive course focusing on DSA in Java.

- **[Udacity - Data Structures and Algorithms Nanodegree](https://www.udacity.com/course/data-structures-and-algorithms-nanodegree--nd256)**
  - In-depth coverage with real-world projects.

### YouTube Playlists and Channels

- **[MIT OpenCourseWare - Introduction to Algorithms](https://www.youtube.com/playlist?list=PL2S3p4q4Xz4cQjRDAghsO93i3S8mR6RQ9)**
  - Lecture series from MIT covering various algorithms and data structures.

- **[Abdul Bari - Data Structures and Algorithms](https://www.youtube.com/playlist?list=PLqM7QwxWklcg7c3KjJolZy5x2u4tdnIY5)**
  - A detailed playlist with explanations and visualizations of various DSA topics.

- **[William Fiset - Algorithms and Data Structures](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsB8ztU8Zlh6lO9b6YN5L7B4)**
  - Covers many advanced topics in DSA with clear explanations and implementations.

- **[The Coding Train - Data Structures and Algorithms](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bNnDkzFjCgu3mC1LAsZ03U)**
  - Fun and engaging tutorials on various algorithms and data structures.

- **[GeeksforGeeks - Data Structures and Algorithms](https://www.youtube.com/user/geeksforgeeksvideos)**
  - Tutorials and problem-solving sessions on different data structures and algorithms.

### Websites and Tutorials

- **[LeetCode](https://leetcode.com/)**
  - Practice problems and challenges to enhance algorithmic skills.

- **[HackerRank](https://www.hackerrank.com/domains/tutorials/10-days-of-javascript)**
  - Coding challenges and tutorials on algorithms and data structures.

- **[GeeksforGeeks](https://www.geeksforgeeks.org/)**
  - Comprehensive articles, tutorials, and problem sets for learning DSA.

- **[TopCoder](https://www.topcoder.com/)**
  - Competitive programming platform with problems related to algorithms and data structures.




## About the Author

**Salman Iyad** is a passionate software engineer and web developer with a deep interest in Data Structures and Algorithms. With a strong background in Java and a commitment to helping others learn.

Feel free to connect and explore more about Salman’s work and contributions!

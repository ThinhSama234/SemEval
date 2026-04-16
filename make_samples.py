"""
make_samples.py — Sinh 3 file parquet mẫu để test pipeline.

Chạy trên Colab / Kaggle:
    python make_samples.py

Output:
    samples/train_sample.parquet   (120 rows, có label + language)
    samples/val_sample.parquet     ( 40 rows, có label + language)
    samples/test_sample.parquet    ( 20 rows, chỉ có ID + code + language)
"""

import os
import random
import pandas as pd

random.seed(42)
os.makedirs('samples', exist_ok=True)

# ---------------------------------------------------------------------------
# Code snippets per language × style (human / AI)
# ---------------------------------------------------------------------------
SNIPPETS = {
    'python': {
        'human': [
            # quirky, informal style
            "def foo(x):\n    # idk why but this works\n    res=[]\n    for i in x:\n        if i%2==0:res.append(i*2)\n    return res",
            "import sys\ndef main():\n    n=int(input())\n    a=list(map(int,input().split()))\n    print(sum(a[:n]))\nmain()",
            "def read_file(path):\n    try:\n        f=open(path)\n        data=f.read()\n        f.close()\n        return data\n    except:\n        return ''",
            "class Node:\n    def __init__(self,v):\n        self.v=v;self.next=None\ndef push(head,v):\n    n=Node(v);n.next=head;return n",
            "xs=[1,3,5,7,9]\nprint([x**2 for x in xs if x>3])\n# quick test\nprint('done')",
            "def gcd(a,b):\n    while b:a,b=b,a%b\n    return a\nprint(gcd(48,18))",
        ],
        'ai': [
            # clean, structured, docstring style
            "def calculate_sum(numbers: list) -> int:\n    \"\"\"\n    Calculate the sum of a list of numbers.\n\n    Args:\n        numbers: A list of integers.\n\n    Returns:\n        The sum of all numbers in the list.\n    \"\"\"\n    return sum(numbers)",
            "def binary_search(arr: list, target: int) -> int:\n    \"\"\"\n    Perform binary search on a sorted array.\n\n    Returns the index of target or -1 if not found.\n    \"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "class Stack:\n    \"\"\"A simple stack implementation using a list.\"\"\"\n\n    def __init__(self):\n        self._items = []\n\n    def push(self, item):\n        \"\"\"Push an item onto the stack.\"\"\"\n        self._items.append(item)\n\n    def pop(self):\n        \"\"\"Pop an item from the stack.\"\"\"\n        if self.is_empty():\n            raise IndexError('Stack is empty')\n        return self._items.pop()\n\n    def is_empty(self) -> bool:\n        \"\"\"Return True if the stack is empty.\"\"\"\n        return len(self._items) == 0",
            "import os\nfrom pathlib import Path\n\n\ndef read_file_safely(file_path: str) -> str:\n    \"\"\"\n    Read a file and return its contents as a string.\n\n    Args:\n        file_path: Path to the file to read.\n\n    Returns:\n        File contents as a string, or empty string on error.\n    \"\"\"\n    path = Path(file_path)\n    if not path.exists():\n        return ''\n    with open(path, 'r', encoding='utf-8') as f:\n        return f.read()",
            "def fibonacci(n: int) -> list:\n    \"\"\"\n    Generate the first n Fibonacci numbers.\n\n    Args:\n        n: Number of Fibonacci numbers to generate.\n\n    Returns:\n        A list containing the first n Fibonacci numbers.\n    \"\"\"\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    sequence = [0, 1]\n    for _ in range(2, n):\n        sequence.append(sequence[-1] + sequence[-2])\n    return sequence",
            "def merge_sort(arr: list) -> list:\n    \"\"\"\n    Sort a list using the merge sort algorithm.\n\n    Args:\n        arr: The list to sort.\n\n    Returns:\n        A new sorted list.\n    \"\"\"\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return _merge(left, right)\n\n\ndef _merge(left: list, right: list) -> list:\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
        ],
    },
    'c++': {
        'human': [
            "#include<bits/stdc++.h>\nusing namespace std;\nint main(){\n    int n;cin>>n;\n    vector<int>a(n);\n    for(auto&x:a)cin>>x;\n    sort(a.begin(),a.end());\n    for(auto x:a)cout<<x<<' ';\n}",
            "#include<stdio.h>\nint gcd(int a,int b){return b?gcd(b,a%b):a;}\nint main(){\n    int a,b;scanf(\"%d%d\",&a,&b);\n    printf(\"%d\\n\",gcd(a,b));\n}",
            "// quick dp solution\n#include<bits/stdc++.h>\nusing namespace std;\nint dp[1001];\nint main(){\n    int n;cin>>n;\n    dp[0]=1;\n    for(int i=1;i<=n;i++)dp[i]=dp[i-1]+(i>1?dp[i-2]:0);\n    cout<<dp[n];\n}",
            "#include<iostream>\nusing namespace std;\nstruct Node{int val;Node*next;Node(int v):val(v),next(nullptr){}};\nvoid print(Node*h){while(h){cout<<h->val<<' ';h=h->next;}cout<<endl;}",
        ],
        'ai': [
            "#include <iostream>\n#include <vector>\n#include <algorithm>\n\n/**\n * @brief Sorts a vector of integers in ascending order.\n * @param numbers The vector to sort.\n * @return A sorted copy of the input vector.\n */\nstd::vector<int> sortNumbers(std::vector<int> numbers) {\n    std::sort(numbers.begin(), numbers.end());\n    return numbers;\n}\n\nint main() {\n    std::vector<int> nums = {5, 2, 8, 1, 9};\n    auto sorted = sortNumbers(nums);\n    for (const auto& n : sorted) {\n        std::cout << n << ' ';\n    }\n    std::cout << std::endl;\n    return 0;\n}",
            "#include <iostream>\n#include <stdexcept>\n\n/**\n * @brief A simple stack implementation.\n */\ntemplate <typename T>\nclass Stack {\npublic:\n    void push(T value) {\n        data_.push_back(value);\n    }\n\n    T pop() {\n        if (isEmpty()) {\n            throw std::underflow_error(\"Stack is empty\");\n        }\n        T top = data_.back();\n        data_.pop_back();\n        return top;\n    }\n\n    bool isEmpty() const {\n        return data_.empty();\n    }\n\nprivate:\n    std::vector<T> data_;\n};",
            "#include <iostream>\n#include <vector>\n\n/**\n * @brief Computes the nth Fibonacci number.\n * @param n The position in the Fibonacci sequence.\n * @return The nth Fibonacci number.\n */\nlong long fibonacci(int n) {\n    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    long long a = 0, b = 1;\n    for (int i = 2; i <= n; ++i) {\n        long long c = a + b;\n        a = b;\n        b = c;\n    }\n    return b;\n}\n\nint main() {\n    for (int i = 0; i <= 10; ++i) {\n        std::cout << \"F(\" << i << \") = \" << fibonacci(i) << std::endl;\n    }\n    return 0;\n}",
        ],
    },
    'java': {
        'human': [
            "import java.util.*;\npublic class Main{\npublic static void main(String[]args){\nScanner sc=new Scanner(System.in);\nint n=sc.nextInt();\nint[]a=new int[n];\nfor(int i=0;i<n;i++)a[i]=sc.nextInt();\nArrays.sort(a);\nfor(int x:a)System.out.print(x+\" \");\n}}",
            "public class Fib{\nstatic int f(int n){return n<2?n:f(n-1)+f(n-2);}\npublic static void main(String[]a){\nfor(int i=0;i<10;i++)System.out.println(f(i));}}",
        ],
        'ai': [
            "import java.util.Arrays;\nimport java.util.List;\n\n/**\n * Utility class for sorting operations.\n */\npublic class SortUtils {\n\n    /**\n     * Sorts an array of integers in ascending order.\n     *\n     * @param numbers The array to sort.\n     * @return A sorted copy of the input array.\n     */\n    public static int[] sortAscending(int[] numbers) {\n        int[] sorted = Arrays.copyOf(numbers, numbers.length);\n        Arrays.sort(sorted);\n        return sorted;\n    }\n}",
            "/**\n * A generic stack implementation.\n *\n * @param <T> The type of elements stored in the stack.\n */\npublic class Stack<T> {\n    private final java.util.Deque<T> deque = new java.util.ArrayDeque<>();\n\n    /**\n     * Pushes an element onto the stack.\n     *\n     * @param element The element to push.\n     */\n    public void push(T element) {\n        deque.push(element);\n    }\n\n    /**\n     * Pops an element from the stack.\n     *\n     * @return The top element.\n     * @throws java.util.EmptyStackException if the stack is empty.\n     */\n    public T pop() {\n        if (deque.isEmpty()) {\n            throw new java.util.EmptyStackException();\n        }\n        return deque.pop();\n    }\n\n    /** Returns true if the stack is empty. */\n    public boolean isEmpty() {\n        return deque.isEmpty();\n    }\n}",
        ],
    },
    'javascript': {
        'human': [
            "// messy but works\nfunction fib(n){\nif(n<=1)return n;\nlet a=0,b=1,c;\nfor(let i=2;i<=n;i++){c=a+b;a=b;b=c;}\nreturn b;\n}\nconsole.log(fib(10));",
            "const arr=[5,3,8,1,2];\narr.sort((a,b)=>a-b);\nconsole.log(arr);",
        ],
        'ai': [
            "/**\n * Calculates the factorial of a non-negative integer.\n *\n * @param {number} n - The non-negative integer.\n * @returns {number} The factorial of n.\n */\nfunction factorial(n) {\n    if (n < 0) {\n        throw new Error('Input must be a non-negative integer');\n    }\n    if (n === 0 || n === 1) {\n        return 1;\n    }\n    return n * factorial(n - 1);\n}",
            "/**\n * Removes duplicate elements from an array.\n *\n * @param {Array} array - The input array.\n * @returns {Array} A new array with duplicates removed.\n */\nfunction removeDuplicates(array) {\n    return [...new Set(array)];\n}",
        ],
    },
}


def build_rows(lang, style, ids_start):
    snips = SNIPPETS[lang][style]
    label = 1 if style == 'ai' else 0
    rows = []
    for i, code in enumerate(snips):
        rows.append({
            'ID': ids_start + i,
            'code': code,
            'label': label,
            'language': lang,
        })
    return rows


def make_split(n_target, include_label=True):
    rows = []
    id_counter = 0
    while len(rows) < n_target:
        for lang in SNIPPETS:
            for style in ('human', 'ai'):
                for code in SNIPPETS[lang][style]:
                    rows.append({
                        'ID': id_counter,
                        'code': code,
                        'label': (1 if style == 'ai' else 0),
                        'language': lang,
                    })
                    id_counter += 1
                    if len(rows) >= n_target:
                        break
                if len(rows) >= n_target:
                    break
            if len(rows) >= n_target:
                break

    df = pd.DataFrame(rows[:n_target])
    if not include_label:
        df = df.drop(columns=['label'])
    return df.reset_index(drop=True)


# Build splits
train_df = make_split(120, include_label=True)
val_df   = make_split( 40, include_label=True)
test_df  = make_split( 20, include_label=False)

# Randomise order
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=1).reset_index(drop=True)

train_df.to_parquet('samples/train_sample.parquet', index=False)
val_df.to_parquet(  'samples/val_sample.parquet',   index=False)
test_df.to_parquet( 'samples/test_sample.parquet',  index=False)

print('=== Sample files created ===')
print(f'train_sample: {train_df.shape}  label={train_df["label"].value_counts().to_dict()}')
print(f'val_sample  : {val_df.shape}  label={val_df["label"].value_counts().to_dict()}')
print(f'test_sample : {test_df.shape}  (no label)')
print(f'Languages   : {sorted(train_df["language"].unique())}')
print('\nSaved to samples/')

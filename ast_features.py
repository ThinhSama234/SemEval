"""
AST Feature Extraction for SemEval Task 13A
============================================
Extracts 10 tree-structure features from code.

Strategy:
  - Python: uses built-in `ast` module (exact parse)
  - Other languages: bracket/token-depth heuristic (no extra deps)
  - Graceful fallback to zeros on any parse failure

Features (10):
  ast_max_depth         — max nesting depth of the tree
  ast_mean_depth        — mean depth across all nodes/lines
  ast_depth_variance    — variance of depths (low = AI-regular, high = human-quirky)
  ast_node_count        — total node count (statement density proxy)
  ast_leaf_ratio        — leaf nodes / total nodes (tree balance)
  ast_avg_branching     — mean children per non-leaf node
  ast_type_entropy      — Shannon entropy of node type distribution
  ast_func_count        — number of function definitions
  ast_func_arg_mean     — mean argument count per function
  ast_func_arg_std      — std of argument counts (consistency signal)
"""

import ast
import re
import math
from collections import Counter, deque

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Regex helpers for heuristic parser
# ---------------------------------------------------------------------------
_RE_FUNC_HEURISTIC = re.compile(
    r'\b(?:def|function|func|fn)\s+\w+\s*\(([^)]*)\)'      # Python/JS/Go/Rust
    r'|(?:void|int|long|float|double|bool|string|char\s*\*?|auto)\s+\w+\s*\(([^)]*)\)',  # C/C++/Java
    re.MULTILINE,
)
_RE_COMMENT_LINE = re.compile(r'^\s*(?://|#|\*|/\*)')
_OPEN_BRACKETS  = frozenset('({[')
_CLOSE_BRACKETS = frozenset(')}]')
_TOKEN_KW = frozenset({
    'if', 'else', 'elif', 'for', 'while', 'do', 'switch', 'case', 'return',
    'def', 'class', 'function', 'func', 'fn', 'import', 'from', 'in',
    'new', 'delete', 'this', 'self', 'try', 'catch', 'except', 'finally',
    'and', 'or', 'not', 'true', 'false', 'null', 'nil', 'None',
    'int', 'void', 'bool', 'string', 'float', 'double', 'char', 'long', 'auto',
})


# ---------------------------------------------------------------------------
# Zero-feature dict (returned on unrecoverable failure)
# ---------------------------------------------------------------------------
def _zero_features() -> dict:
    return {
        'ast_max_depth': 0,
        'ast_mean_depth': 0.0,
        'ast_depth_variance': 0.0,
        'ast_node_count': 0,
        'ast_leaf_ratio': 0.0,
        'ast_avg_branching': 0.0,
        'ast_type_entropy': 0.0,
        'ast_func_count': 0,
        'ast_func_arg_mean': 0.0,
        'ast_func_arg_std': 0.0,
    }


# ---------------------------------------------------------------------------
# Python: exact parse via built-in ast
# ---------------------------------------------------------------------------
def _features_from_python_ast(code: str) -> dict | None:
    """Returns feature dict using Python ast, or None on SyntaxError."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    depths = []
    node_types = []
    branching = []
    func_args = []

    # Iterative BFS to avoid Python recursion limit on deep code
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        node_types.append(type(node).__name__)
        depths.append(depth)
        children = list(ast.iter_child_nodes(node))
        if children:
            branching.append(len(children))
            for child in children:
                queue.append((child, depth + 1))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_args.append(len(node.args.args))

    if not depths:
        return _zero_features()

    n = len(depths)
    max_depth = max(depths)
    mean_depth = sum(depths) / n
    depth_var = sum((d - mean_depth) ** 2 for d in depths) / n

    node_count = n
    leaf_count = node_count - len(branching)
    leaf_ratio = leaf_count / node_count

    avg_branching = sum(branching) / len(branching) if branching else 0.0

    type_counts = Counter(node_types)
    total_t = sum(type_counts.values())
    type_entropy = -sum(
        (c / total_t) * math.log2(c / total_t) for c in type_counts.values()
    ) if total_t > 0 else 0.0

    func_count = len(func_args)
    func_arg_mean = sum(func_args) / func_count if func_count else 0.0
    func_arg_std = (
        math.sqrt(sum((a - func_arg_mean) ** 2 for a in func_args) / func_count)
        if func_count > 1 else 0.0
    )

    return {
        'ast_max_depth': max_depth,
        'ast_mean_depth': round(mean_depth, 4),
        'ast_depth_variance': round(depth_var, 4),
        'ast_node_count': node_count,
        'ast_leaf_ratio': round(leaf_ratio, 4),
        'ast_avg_branching': round(avg_branching, 4),
        'ast_type_entropy': round(type_entropy, 4),
        'ast_func_count': func_count,
        'ast_func_arg_mean': round(func_arg_mean, 4),
        'ast_func_arg_std': round(func_arg_std, 4),
    }


# ---------------------------------------------------------------------------
# Generic: bracket/token-depth heuristic (works for any language)
# ---------------------------------------------------------------------------
def _features_from_heuristic(code: str) -> dict:
    """
    Language-agnostic AST approximation using bracket nesting + token analysis.
    Designed to produce features that correlate with true AST metrics.
    """
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip() and not _RE_COMMENT_LINE.match(l)]

    if not non_empty_lines:
        return _zero_features()

    # --- Depth tracking via bracket stack ---
    depth = 0
    line_depths = []
    max_depth = 0
    for line in non_empty_lines:
        for ch in line:
            if ch in _OPEN_BRACKETS:
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif ch in _CLOSE_BRACKETS:
                depth = max(0, depth - 1)
        line_depths.append(depth)

    n_lines = len(line_depths)
    mean_depth = sum(line_depths) / n_lines
    depth_var = sum((d - mean_depth) ** 2 for d in line_depths) / n_lines

    # --- Node count: non-empty, non-comment lines as statement proxy ---
    node_count = n_lines

    # --- Leaf ratio: lines at locally deepest depth ---
    if max_depth == 0:
        leaf_ratio = 1.0
    else:
        leaf_count = sum(1 for d in line_depths if d == max_depth)
        leaf_ratio = leaf_count / n_lines

    # --- Branching: branch keywords per line ---
    _re_branch = re.compile(r'\b(?:if|else|elif|for|while|switch|case|catch|except|try)\b')
    branch_counts = [len(_re_branch.findall(l)) for l in non_empty_lines]
    avg_branching = sum(branch_counts) / n_lines

    # --- Token type entropy ---
    tokens = code.split()
    categories = []
    for t in tokens:
        clean = t.strip('.,;:(){}[]<>\'"')
        if clean.lower() in _TOKEN_KW:
            categories.append('kw')
        elif re.fullmatch(r'[0-9][0-9._]*[fFdDlLuU]*', clean):
            categories.append('num')
        elif re.fullmatch(r'[a-zA-Z_]\w*', clean):
            categories.append('id')
        elif clean:
            categories.append('op')

    cat_counts = Counter(categories)
    total_c = sum(cat_counts.values())
    type_entropy = (
        -sum((c / total_c) * math.log2(c / total_c) for c in cat_counts.values())
        if total_c > 0 else 0.0
    )

    # --- Function definitions and argument counts ---
    func_args = []
    for m in _RE_FUNC_HEURISTIC.finditer(code):
        # group(1) = Python/JS/Go style, group(2) = C/C++ style
        args_str = m.group(1) or m.group(2) or ''
        args_str = args_str.strip()
        if args_str:
            # count by commas, filter empty splits
            n_args = len([a for a in args_str.split(',') if a.strip()])
        else:
            n_args = 0
        func_args.append(n_args)

    func_count = len(func_args)
    func_arg_mean = sum(func_args) / func_count if func_count else 0.0
    func_arg_std = (
        math.sqrt(sum((a - func_arg_mean) ** 2 for a in func_args) / func_count)
        if func_count > 1 else 0.0
    )

    return {
        'ast_max_depth': max_depth,
        'ast_mean_depth': round(mean_depth, 4),
        'ast_depth_variance': round(depth_var, 4),
        'ast_node_count': node_count,
        'ast_leaf_ratio': round(leaf_ratio, 4),
        'ast_avg_branching': round(avg_branching, 4),
        'ast_type_entropy': round(type_entropy, 4),
        'ast_func_count': func_count,
        'ast_func_arg_mean': round(func_arg_mean, 4),
        'ast_func_arg_std': round(func_arg_std, 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_PYTHON_LANGS = frozenset({'python', 'py', 'python3'})


def extract_ast_features(code: str, language: str | None = None) -> dict:
    """
    Extract 10 AST-derived features from a code snippet.

    Args:
        code:     Source code string.
        language: Optional language hint (e.g. 'python', 'cpp', 'java').
                  When 'python' (or None and code parses as Python), uses
                  the exact ast module; otherwise uses the heuristic parser.

    Returns:
        Dict with keys: ast_max_depth, ast_mean_depth, ast_depth_variance,
        ast_node_count, ast_leaf_ratio, ast_avg_branching, ast_type_entropy,
        ast_func_count, ast_func_arg_mean, ast_func_arg_std.
    """
    if not code or not code.strip():
        return _zero_features()

    lang = (language or '').lower().strip()

    # Try exact Python parse if language is Python or unknown
    if lang in _PYTHON_LANGS or lang == '':
        result = _features_from_python_ast(code)
        if result is not None:
            return result
        # SyntaxError → fall through to heuristic

    return _features_from_heuristic(code)


def extract_ast_features_batch(
    codes,
    languages=None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Batch extraction of AST features.

    Args:
        codes:         List or pd.Series of code strings.
        languages:     Optional list/Series of language strings (same length as codes).
        show_progress: Show tqdm progress bar.

    Returns:
        pd.DataFrame with 10 AST feature columns, same index as codes.
    """
    if not isinstance(codes, pd.Series):
        codes = pd.Series(codes)

    if languages is not None and not isinstance(languages, pd.Series):
        languages = pd.Series(languages, index=codes.index)

    items = list(codes.items())
    iterator = items
    if show_progress and _TQDM_AVAILABLE:
        iterator = tqdm(items, desc='AST features', total=len(items))

    records = []
    for idx, code in iterator:
        lang = languages[idx] if languages is not None else None
        try:
            feat = extract_ast_features(code, language=lang)
        except Exception:
            feat = _zero_features()
        records.append(feat)

    return pd.DataFrame(records, index=codes.index)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    _samples = [
        ('python', """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""),
        ('cpp', """
int bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j+1]) {
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
    return 0;
}
"""),
        ('java', """
public int add(int a, int b) { return a + b; }
"""),
    ]

    print(f"{'Language':<10}  {'max_d':>6}  {'mean_d':>7}  {'nodes':>6}  "
          f"{'leaf_r':>7}  {'branch':>7}  {'entropy':>8}  {'funcs':>5}  "
          f"{'arg_mu':>7}  {'arg_sd':>7}")
    print('-' * 80)
    for lang, code in _samples:
        f = extract_ast_features(code.strip(), language=lang)
        print(f"{lang:<10}  {f['ast_max_depth']:>6}  {f['ast_mean_depth']:>7.3f}  "
              f"{f['ast_node_count']:>6}  {f['ast_leaf_ratio']:>7.3f}  "
              f"{f['ast_avg_branching']:>7.3f}  {f['ast_type_entropy']:>8.4f}  "
              f"{f['ast_func_count']:>5}  {f['ast_func_arg_mean']:>7.3f}  "
              f"{f['ast_func_arg_std']:>7.3f}")

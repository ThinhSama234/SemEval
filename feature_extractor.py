import re
import math
import gzip
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# =============================================================================
# CONSTANTS & REGEXES
# =============================================================================
MULTI_LANG_KEYWORDS = {
    'if', 'else', 'elif', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'void', 'bool', 'boolean', 'string', 'var', 'let', 'const',
    'def', 'function', 'class', 'struct', 'interface', 'package', 'import', 'using', 'namespace',
    'public', 'private', 'protected', 'static', 'final', 'try', 'catch', 'finally',
    'throw', 'throws', 'new', 'delete', 'true', 'false', 'null', 'nil', 'None', 'self', 'this',
    'func', 'defer', 'go', 'map', 'chan', 'type', 'range', 'print', 'println', 'printf',
    'from', 'as', 'in', 'is', 'not', 'and', 'or', 'with', 'yield', 'lambda', 'pass',
    'assert', 'raise', 'except', 'global', 'nonlocal', 'async', 'await',
}

RE_WORDS = re.compile(r'\w+')
RE_IDENTIFIERS = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
RE_CAMEL = re.compile(r'[a-z][A-Z]')
RE_DIGITS = re.compile(r'\d')
RE_EQ_SPACED = re.compile(r' = ')
RE_EQ_NOSPACED = re.compile(r'(?<=[^\s!=<>])=(?=[^\s=])')
RE_OP_SPACED = re.compile(r' [+\-*/] ')
RE_OP_NOSPACED = re.compile(r'(?<=[^\s])[+\-*/](?=[^\s])')
RE_HUMAN_MARKERS = re.compile(r'\b(TODO|FIXME|XXX|HACK|DEBUG|WORKAROUND|TEMP|UGLY|REFACTOR)\b', re.IGNORECASE)


# =============================================================================
# A. BASIC FEATURES
# =============================================================================
def extract_code_features(code_series):
    """Extract structural features from code strings. vectorized operation expecting a pd.Series"""
    features = pd.DataFrame(index=code_series.index)

    features['char_count'] = code_series.str.len()
    features['line_count'] = code_series.str.count('\n') + 1
    features['avg_line_len'] = features['char_count'] / features['line_count']

    features['space_count'] = code_series.str.count(' ')
    features['tab_count'] = code_series.str.count('\t')
    features['space_ratio'] = features['space_count'] / features['char_count']
    features['empty_line_count'] = code_series.str.count(r'\n\s*\n')
    features['empty_line_ratio'] = features['empty_line_count'] / features['line_count']
    features['leading_space_lines'] = code_series.apply(lambda x: sum(1 for line in x.split('\n') if line and line[0] == ' '))
    features['leading_tab_lines'] = code_series.apply(lambda x: sum(1 for line in x.split('\n') if line and line[0] == '\t'))

    features['open_paren'] = code_series.str.count(r'\(')
    features['close_paren'] = code_series.str.count(r'\)')
    features['open_brace'] = code_series.str.count(r'\{')
    features['close_brace'] = code_series.str.count(r'\}')
    features['open_bracket'] = code_series.str.count(r'\[')
    features['close_bracket'] = code_series.str.count(r'\]')
    features['semicolons'] = code_series.str.count(';')
    features['colons'] = code_series.str.count(':')
    features['commas'] = code_series.str.count(',')

    features['single_line_comments'] = code_series.str.count(r'(?:^|\n)\s*(?://|#)')
    features['comment_ratio'] = features['single_line_comments'] / features['line_count']

    features['double_quotes'] = code_series.str.count('"')
    features['single_quotes'] = code_series.str.count("'")

    features['max_line_len'] = code_series.apply(lambda x: max((len(l) for l in x.split('\n')), default=0))
    features['min_line_len'] = code_series.apply(lambda x: min((len(l) for l in x.split('\n') if l.strip()), default=0))
    features['max_indent'] = code_series.apply(lambda x: max((len(l) - len(l.lstrip()) for l in x.split('\n') if l.strip()), default=0))

    features['num_keywords_def'] = code_series.str.count(r'\bdef\b')
    features['num_keywords_class'] = code_series.str.count(r'\bclass\b')
    features['num_keywords_import'] = code_series.str.count(r'\b(?:import|from|#include|using|require)\b')
    features['num_keywords_return'] = code_series.str.count(r'\breturn\b')
    features['num_keywords_if'] = code_series.str.count(r'\bif\b')
    features['num_keywords_for'] = code_series.str.count(r'\b(?:for|while)\b')

    features['unique_chars'] = code_series.apply(lambda x: len(set(x)))
    features['unique_char_ratio'] = features['unique_chars'] / features['char_count'].replace(0, 1) # prevent div by zero
    
    features['trailing_ws_lines'] = code_series.apply(lambda x: sum(1 for l in x.split('\n') if l != l.rstrip()))

    return features


# =============================================================================
# B. ADVANCED FEATURES
# =============================================================================
def shannon_entropy(text):
    if not text: return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())

def compression_ratio(text):
    if not text: return 0.0
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded)
    return len(compressed) / max(len(encoded), 1)

def max_nesting_depth(code):
    max_depth = 0; depth = 0
    for ch in code:
        if ch in ('{', '('): depth += 1; max_depth = max(max_depth, depth)
        elif ch in ('}', ')'): depth = max(0, depth - 1)
    return max_depth

def duplicate_line_ratio(code):
    lines = [l.strip() for l in code.split('\n') if l.strip()]
    if not lines: return 0.0
    freq = Counter(lines)
    duplicates = sum(c - 1 for c in freq.values() if c > 1)
    return duplicates / len(lines)

def line_length_std(code):
    lengths = [len(l) for l in code.split('\n')]
    if len(lengths) < 2: return 0.0
    mean_len = sum(lengths) / len(lengths)
    var = sum((l - mean_len)**2 for l in lengths) / (len(lengths) - 1)
    return math.sqrt(var)

def indent_consistency(code):
    indents = [len(l) - len(l.lstrip()) for l in code.split('\n') if l.strip()]
    if len(indents) < 2: return 0.0
    mean_ind = sum(indents) / len(indents)
    var = sum((i - mean_ind)**2 for i in indents) / (len(indents) - 1)
    return math.sqrt(var)

def avg_token_length(code):
    tokens = code.split()
    if not tokens: return 0.0
    return sum(len(t) for t in tokens) / len(tokens)

def vocabulary_richness(code):
    tokens = code.split()
    if not tokens: return 0.0
    return len(set(tokens)) / len(tokens)

def naming_features(code):
    identifiers = RE_IDENTIFIERS.findall(code)
    if not identifiers: return 0.0, 0.0, 0.0
    avg_name_len = sum(len(i) for i in identifiers) / len(identifiers)
    snake_count = sum(1 for i in identifiers if '_' in i and i.islower())
    camel_count = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and '_' not in i)
    total = max(len(identifiers), 1)
    return avg_name_len, snake_count / total, camel_count / total

def max_blank_streak(code):
    max_streak = 0; streak = 0
    for line in code.split('\n'):
        if not line.strip(): streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    return max_streak


# =============================================================================
# C. ENHANCED FEATURES
# =============================================================================

def identifier_features(code):
    identifiers = [w for w in RE_IDENTIFIERS.findall(code) if w.lower() not in MULTI_LANG_KEYWORDS and not w.isdigit()]
    if not identifiers: return 0.0, 0.0, 0.0, 0.0
    
    lens = [len(w) for w in identifiers]
    avg_len = np.mean(lens)
    
    all_chars = "".join(identifiers)
    id_entropy = 0.0
    if all_chars:
        char_counts = Counter(all_chars)
        total = sum(char_counts.values())
        id_entropy = -sum((c/total) * math.log2(c/total) for c in char_counts.values())
        
    short_ratio = sum(1 for w in identifiers if len(w) <= 2) / len(identifiers)
    num_ratio = sum(1 for w in identifiers if RE_DIGITS.search(w)) / len(identifiers)
    return avg_len, id_entropy, short_ratio, num_ratio

def style_consistency(code):
    identifiers = [w for w in RE_IDENTIFIERS.findall(code) if w.lower() not in MULTI_LANG_KEYWORDS and len(w) > 1]
    snake_count = sum(1 for w in identifiers if '_' in w)
    camel_count = sum(1 for w in identifiers if RE_CAMEL.search(w))
    total = snake_count + camel_count
    if total == 0: return 0.5
    return abs(snake_count - camel_count) / total

def operator_spacing_features(code):
    eq_spaced = len(RE_EQ_SPACED.findall(code))
    eq_nospaced = len(RE_EQ_NOSPACED.findall(code))
    total_eq = eq_spaced + eq_nospaced
    eq_dirty_ratio = eq_nospaced / total_eq if total_eq > 0 else 0.0
    
    op_spaced = len(RE_OP_SPACED.findall(code))
    op_nospaced = len(RE_OP_NOSPACED.findall(code))
    total_op = op_spaced + op_nospaced
    op_dirty_ratio = op_nospaced / total_op if total_op > 0 else 0.0
    
    all_spaced = eq_spaced + op_spaced
    all_nospaced = eq_nospaced + op_nospaced
    total_all = all_spaced + all_nospaced
    spacing_consistency = max(all_spaced, all_nospaced) / total_all if total_all > 0 else 0.5
    return eq_dirty_ratio, op_dirty_ratio, spacing_consistency

def human_marker_score(code):
    markers = RE_HUMAN_MARKERS.findall(code)
    return 1.0 if len(markers) > 0 else 0.0, len(markers)

def line_entropy(line):
    if not line.strip(): return 0.0
    freq = Counter(line)
    length = len(line)
    return -sum((c/length) * math.log2(c/length) for c in freq.values())

def burstiness_features(code):
    lines = [l for l in code.split('\n') if l.strip()]
    if len(lines) < 2: return 0.0, 0.0, 0.0
    entropies = [line_entropy(l) for l in lines]
    mean_ent = np.mean(entropies)
    std_ent = np.std(entropies)
    burstiness = (std_ent - mean_ent) / (std_ent + mean_ent) if (std_ent + mean_ent) > 0 else 0.0
    return burstiness, std_ent, mean_ent

def punctuation_entropy(code):
    puncts = [c for c in code if c in '(){}[]<>;:,.!?@#$%^&*+-=/\\|~`"\'']
    if not puncts: return 0.0, 0.0
    freq = Counter(puncts)
    total = sum(freq.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    density = len(puncts) / max(len(code), 1)
    return entropy, density

def whitespace_pattern_features(code):
    indent_patterns = [l[:len(l) - len(l.lstrip())] for l in code.split('\n') if l.strip()]
    if not indent_patterns: return 0, 0.0
    unique_patterns = len(set(indent_patterns))
    pattern_diversity = unique_patterns / len(indent_patterns)
    return unique_patterns, pattern_diversity

def halstead_features(code):
    operators = re.findall(r'[+\-*/=<>!&|^~%]+|[\(\)\{\}\[\];:,.]', code)
    operands = RE_IDENTIFIERS.findall(code)
    n1, n2 = len(set(operators)), len(set(operands))
    N1, N2 = len(operators), len(operands)
    n, N = n1 + n2, N1 + N2
    if n > 0 and N > 0:
        volume = N * math.log2(n)
        difficulty = (n1 / 2) * (N2 / max(n2, 1))
    else:
        volume = 0.0; difficulty = 0.0
    return n1, n2, volume, difficulty

def ngram_repetition(code, n=2):
    tokens = code.split()
    if len(tokens) < n: return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams: return 0.0
    freq = Counter(ngrams)
    repeated = sum(c - 1 for c in freq.values() if c > 1)
    return repeated / len(ngrams)

def tab_space_signal(code):
    lines = code.split('\n')
    tab_lines = sum(1 for l in lines if l and l[0] == '\t')
    space_lines = sum(1 for l in lines if l and l[0] == ' ')
    total = tab_lines + space_lines
    return tab_lines / total if total > 0 else 0.5

def code_structure_ratios(code):
    lines = [l for l in code.split('\n') if l.strip()]
    total_lines = max(len(lines), 1)
    docstring_count = len(re.findall(r'"""', code)) + len(re.findall(r"'''", code))
    has_docstring = 1.0 if docstring_count >= 2 else 0.0
    multiline_comments = len(re.findall(r'/\*', code))
    func_density = len(re.findall(r'\b(?:def|function|func|fn|sub|proc)\b', code)) / total_lines
    class_density = len(re.findall(r'\bclass\b', code)) / total_lines
    return has_docstring, multiline_comments, func_density, class_density

# =============================================================================
# D. STYLE-ONLY FEATURES (for hybrid pipeline)
# =============================================================================

RE_FUNC_DEF = re.compile(r'(?:^|\n)\s*(?:def |function |func |fn |public\s+.*\(|private\s+.*\(|protected\s+.*\(|static\s+.*\(|void\s+\w+\s*\(|int\s+\w+\s*\(|string\s+\w+\s*\()')
RE_INLINE_COMMENT = re.compile(r'[^#/\n]+(?:#|//)\s*\S')  # code then comment
RE_BRANCH_KW = re.compile(r'\b(?:if|else|elif|for|while|switch|case|catch|except|try)\b')

def _find_function_ranges(code):
    """Find (start_line, end_line) for each function in code."""
    lines = code.split('\n')
    func_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if RE_FUNC_DEF.match('\n' + line) or stripped.startswith('def ') or stripped.startswith('function '):
            func_starts.append(i)
    if not func_starts:
        return []
    ranges = []
    for j, start in enumerate(func_starts):
        end = func_starts[j + 1] - 1 if j + 1 < len(func_starts) else len(lines) - 1
        ranges.append((start, end))
    return ranges


def comment_completeness(code):
    """Ratio of functions preceded by a comment or docstring."""
    lines = code.split('\n')
    ranges = _find_function_ranges(code)
    if not ranges:
        return 0.0
    documented = 0
    for start, _ in ranges:
        if start > 0:
            prev = lines[start - 1].strip()
            if prev.startswith('#') or prev.startswith('//') or prev.startswith('/*') or \
               prev.startswith('"""') or prev.startswith("'''") or prev.endswith('"""') or prev.endswith("'''"):
                documented += 1
    return documented / len(ranges)


def blank_per_function(code):
    """Average blank lines per function body."""
    lines = code.split('\n')
    ranges = _find_function_ranges(code)
    if not ranges:
        return 0.0
    counts = []
    for start, end in ranges:
        body = lines[start:end + 1]
        blanks = sum(1 for l in body if not l.strip())
        counts.append(blanks)
    return sum(counts) / len(counts)


def comment_per_function(code):
    """Average comment lines per function body."""
    lines = code.split('\n')
    ranges = _find_function_ranges(code)
    if not ranges:
        return 0.0
    counts = []
    for start, end in ranges:
        body = lines[start:end + 1]
        comments = sum(1 for l in body if l.strip().startswith('#') or l.strip().startswith('//'))
        counts.append(comments)
    return sum(counts) / len(counts)


def inline_comment_ratio(code):
    """Ratio of inline comments (code + comment) vs total comment lines."""
    lines = code.split('\n')
    total_comments = 0
    inline_comments = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            total_comments += 1
        elif RE_INLINE_COMMENT.match(line):
            total_comments += 1
            inline_comments += 1
    return inline_comments / max(total_comments, 1)


def naming_uniformity(code):
    """1.0 = all identifiers follow one convention, 0.0 = mixed."""
    identifiers = [w for w in RE_IDENTIFIERS.findall(code)
                   if w.lower() not in MULTI_LANG_KEYWORDS and len(w) > 1 and not w.isdigit()]
    if not identifiers:
        return 0.5
    snake = sum(1 for w in identifiers if '_' in w and w.islower())
    camel = sum(1 for w in identifiers if RE_CAMEL.search(w) and '_' not in w)
    other = len(identifiers) - snake - camel
    total = len(identifiers)
    dominant = max(snake, camel, other)
    return dominant / total


def keyword_density(code):
    """Keyword count / total token count."""
    tokens = code.split()
    if not tokens:
        return 0.0
    kw_count = sum(1 for t in tokens if t in MULTI_LANG_KEYWORDS)
    return kw_count / len(tokens)


def avg_block_length(code):
    """Average lines per indented block."""
    lines = code.split('\n')
    blocks = []
    current_block = 0
    in_block = False
    for line in lines:
        indent = len(line) - len(line.lstrip()) if line.strip() else 0
        if indent > 0:
            current_block += 1
            in_block = True
        else:
            if in_block and current_block > 0:
                blocks.append(current_block)
            current_block = 0
            in_block = False
    if in_block and current_block > 0:
        blocks.append(current_block)
    return sum(blocks) / max(len(blocks), 1)


def cyclomatic_proxy(code):
    """Count of branching keywords as complexity proxy."""
    return len(RE_BRANCH_KW.findall(code))


def comment_word_count_avg(code):
    """Average word count per comment line."""
    lines = code.split('\n')
    word_counts = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            words = stripped[1:].split()
            word_counts.append(len(words))
        elif stripped.startswith('//'):
            words = stripped[2:].split()
            word_counts.append(len(words))
    if not word_counts:
        return 0.0
    return sum(word_counts) / len(word_counts)


def function_size_regularity(code):
    """Std of function sizes (lines). Low = regular, high = irregular."""
    lines = code.split('\n')
    ranges = _find_function_ranges(code)
    if len(ranges) < 2:
        return 0.0
    sizes = [end - start + 1 for start, end in ranges]
    mean_size = sum(sizes) / len(sizes)
    var = sum((s - mean_size) ** 2 for s in sizes) / (len(sizes) - 1)
    return math.sqrt(var)


def line_len_burstiness(code):
    """Burstiness of line lengths: (std - mean) / (std + mean)."""
    lines = code.split('\n')
    if len(lines) < 2:
        return 0.0
    lengths = [len(l) for l in lines if l.strip()]
    if len(lengths) < 2:
        return 0.0
    mean_len = sum(lengths) / len(lengths)
    std_len = math.sqrt(sum((l - mean_len) ** 2 for l in lengths) / (len(lengths) - 1))
    if std_len + mean_len == 0:
        return 0.0
    return (std_len - mean_len) / (std_len + mean_len)


def extract_style_features(code):
    """Extract all 20 style-only features for the hybrid pipeline."""
    lines = code.split('\n')
    line_count = max(len(lines), 1)
    non_empty = [l for l in lines if l.strip()]
    token_count = len(code.split())

    return {
        'comment_ratio': sum(1 for l in lines if l.strip().startswith('#') or l.strip().startswith('//')) / line_count,
        'blank_line_ratio': sum(1 for l in lines if not l.strip()) / line_count,
        'indentation_std': indent_consistency(code),
        'line_len_std': line_length_std(code),
        'style_consistency': style_consistency(code),
        'ttr': vocabulary_richness(code),
        'comment_completeness': comment_completeness(code),
        'blank_per_function': blank_per_function(code),
        'comment_per_function': comment_per_function(code),
        'trailing_ws_ratio': sum(1 for l in lines if l != l.rstrip()) / line_count,
        'naming_uniformity': naming_uniformity(code),
        'line_len_burstiness': line_len_burstiness(code),
        'token_entropy': shannon_entropy(code),
        'inline_comment_ratio': inline_comment_ratio(code),
        'keyword_density': keyword_density(code),
        'max_nesting_depth': max_nesting_depth(code),
        'avg_block_length': avg_block_length(code),
        'cyclomatic_proxy': cyclomatic_proxy(code),
        'comment_word_count_avg': comment_word_count_avg(code),
        'function_size_regularity': function_size_regularity(code),
    }


# =============================================================================
# PIPELINE RUNNER
# =============================================================================
def extract_all_features(codes, show_progress=False) -> pd.DataFrame:
    """
    Given a list or pd.Series of code strings, extract all Basic, Advanced and 
    Enhanced features to match `my_code.ipynb` output.
    Returns a unified pandas DataFrame with the features.
    """
    if not isinstance(codes, pd.Series):
        codes = pd.Series(codes)
    
    # Basic features are vectorized
    features_df = extract_code_features(codes)
    
    # Prepare individual lists
    adv_dict = {
        'shannon_entropy': [], 'compression_ratio': [], 'max_nesting_depth': [],
        'duplicate_line_ratio': [], 'line_length_std': [], 'indent_consistency': [],
        'avg_token_length': [], 'vocabulary_richness': [], 'avg_identifier_len': [],
        'snake_case_ratio': [], 'camel_case_ratio': [], 'token_count': [],
        'max_blank_streak': [],
        
        'id_avg_len': [], 'id_char_entropy': [], 'id_short_ratio': [], 'id_numeric_ratio': [],
        'style_consistency': [], 'eq_dirty_ratio': [], 'op_dirty_ratio': [], 'spacing_consistency': [],
        'has_human_marker': [], 'human_marker_count': [], 'burstiness': [], 'line_entropy_std': [],
        'line_entropy_mean': [], 'punct_entropy': [], 'punct_density': [], 'unique_indent_patterns': [],
        'indent_pattern_diversity': [], 'halstead_op_vocab': [], 'halstead_operand_vocab': [],
        'halstead_volume': [], 'halstead_difficulty': [], 'bigram_repetition': [],
        'trigram_repetition': [], 'tab_space_signal': [], 'has_docstring': [],
        'multiline_comment_count': [], 'func_density': [], 'class_density': []
    }
    
    it = codes.items()
    if show_progress:
        it = tqdm(it, desc="Extracting Adv/Enh Features", total=len(codes))
        
    for idx, code in it:
        # Advanced
        adv_dict['shannon_entropy'].append(shannon_entropy(code))
        adv_dict['compression_ratio'].append(compression_ratio(code))
        adv_dict['max_nesting_depth'].append(max_nesting_depth(code))
        adv_dict['duplicate_line_ratio'].append(duplicate_line_ratio(code))
        adv_dict['line_length_std'].append(line_length_std(code))
        adv_dict['indent_consistency'].append(indent_consistency(code))
        adv_dict['avg_token_length'].append(avg_token_length(code))
        adv_dict['vocabulary_richness'].append(vocabulary_richness(code))
        
        v1, v2, v3 = naming_features(code)
        adv_dict['avg_identifier_len'].append(v1)
        adv_dict['snake_case_ratio'].append(v2)
        adv_dict['camel_case_ratio'].append(v3)
        
        adv_dict['token_count'].append(len(code.split()))
        adv_dict['max_blank_streak'].append(max_blank_streak(code))
        
        # Enhanced
        v1, v2, v3, v4 = identifier_features(code)
        adv_dict['id_avg_len'].append(v1)
        adv_dict['id_char_entropy'].append(v2)
        adv_dict['id_short_ratio'].append(v3)
        adv_dict['id_numeric_ratio'].append(v4)
        
        adv_dict['style_consistency'].append(style_consistency(code))
        
        v1, v2, v3 = operator_spacing_features(code)
        adv_dict['eq_dirty_ratio'].append(v1)
        adv_dict['op_dirty_ratio'].append(v2)
        adv_dict['spacing_consistency'].append(v3)
        
        v1, v2 = human_marker_score(code)
        adv_dict['has_human_marker'].append(v1)
        adv_dict['human_marker_count'].append(v2)
        
        v1, v2, v3 = burstiness_features(code)
        adv_dict['burstiness'].append(v1)
        adv_dict['line_entropy_std'].append(v2)
        adv_dict['line_entropy_mean'].append(v3)
        
        v1, v2 = punctuation_entropy(code)
        adv_dict['punct_entropy'].append(v1)
        adv_dict['punct_density'].append(v2)
        
        v1, v2 = whitespace_pattern_features(code)
        adv_dict['unique_indent_patterns'].append(v1)
        adv_dict['indent_pattern_diversity'].append(v2)
        
        v1, v2, v3, v4 = halstead_features(code)
        adv_dict['halstead_op_vocab'].append(v1)
        adv_dict['halstead_operand_vocab'].append(v2)
        adv_dict['halstead_volume'].append(v3)
        adv_dict['halstead_difficulty'].append(v4)
        
        adv_dict['bigram_repetition'].append(ngram_repetition(code, 2))
        adv_dict['trigram_repetition'].append(ngram_repetition(code, 3))
        
        adv_dict['tab_space_signal'].append(tab_space_signal(code))
        
        v1, v2, v3, v4 = code_structure_ratios(code)
        adv_dict['has_docstring'].append(v1)
        adv_dict['multiline_comment_count'].append(v2)
        adv_dict['func_density'].append(v3)
        adv_dict['class_density'].append(v4)

    adv_df = pd.DataFrame(adv_dict, index=codes.index)
    return pd.concat([features_df, adv_df], axis=1)

# =============================================================================
# E. 24-FEATURE SET  (18 stylistic + 6 consistency)
# =============================================================================
# Stylistic (18):
#   comment_ratio, blank_line_ratio, indentation_std, line_len_std,
#   style_consistency, ttr, comment_completeness, blank_per_function,
#   comment_per_function, trailing_ws_ratio, naming_uniformity,
#   line_len_burstiness, token_entropy, inline_comment_ratio,
#   keyword_density, max_nesting_depth, avg_block_length, cyclomatic_proxy
#
# Consistency (6):
#   eq_dirty_ratio, op_dirty_ratio, spacing_consistency,
#   duplicate_line_ratio, bigram_repetition, function_size_regularity
# =============================================================================

STYLISTIC_FEATURES = [
    'comment_ratio', 'blank_line_ratio', 'indentation_std', 'line_len_std',
    'style_consistency', 'ttr', 'comment_completeness', 'blank_per_function',
    'comment_per_function', 'trailing_ws_ratio', 'naming_uniformity',
    'line_len_burstiness', 'token_entropy', 'inline_comment_ratio',
    'keyword_density', 'max_nesting_depth', 'avg_block_length', 'cyclomatic_proxy',
]

CONSISTENCY_FEATURES = [
    'eq_dirty_ratio', 'op_dirty_ratio', 'spacing_consistency',
    'duplicate_line_ratio', 'bigram_repetition', 'function_size_regularity',
]

HANDCRAFTED_24 = STYLISTIC_FEATURES + CONSISTENCY_FEATURES


def extract_24_features(code: str) -> dict:
    """
    Extract exactly 24 handcrafted features from a single code string.
    18 stylistic + 6 consistency. Fast, no external dependencies.
    """
    if not code or not code.strip():
        return {k: 0.0 for k in HANDCRAFTED_24}

    lines = code.split('\n')
    line_count = max(len(lines), 1)

    # ---- Stylistic ----
    comment_lines = sum(
        1 for l in lines if l.strip().startswith('#') or l.strip().startswith('//')
    )

    eq_d, op_d, sp_cons = operator_spacing_features(code)

    feats = {
        # 1
        'comment_ratio': comment_lines / line_count,
        # 2
        'blank_line_ratio': sum(1 for l in lines if not l.strip()) / line_count,
        # 3
        'indentation_std': indent_consistency(code),
        # 4
        'line_len_std': line_length_std(code),
        # 5
        'style_consistency': style_consistency(code),
        # 6
        'ttr': vocabulary_richness(code),
        # 7
        'comment_completeness': comment_completeness(code),
        # 8
        'blank_per_function': blank_per_function(code),
        # 9
        'comment_per_function': comment_per_function(code),
        # 10
        'trailing_ws_ratio': sum(1 for l in lines if l != l.rstrip()) / line_count,
        # 11
        'naming_uniformity': naming_uniformity(code),
        # 12
        'line_len_burstiness': line_len_burstiness(code),
        # 13
        'token_entropy': shannon_entropy(code),
        # 14
        'inline_comment_ratio': inline_comment_ratio(code),
        # 15
        'keyword_density': keyword_density(code),
        # 16
        'max_nesting_depth': max_nesting_depth(code),
        # 17
        'avg_block_length': avg_block_length(code),
        # 18
        'cyclomatic_proxy': cyclomatic_proxy(code),
        # ---- Consistency ----
        # 19
        'eq_dirty_ratio': eq_d,
        # 20
        'op_dirty_ratio': op_d,
        # 21
        'spacing_consistency': sp_cons,
        # 22
        'duplicate_line_ratio': duplicate_line_ratio(code),
        # 23
        'bigram_repetition': ngram_repetition(code, 2),
        # 24
        'function_size_regularity': function_size_regularity(code),
    }
    return feats


def extract_24_features_batch(codes, show_progress: bool = False) -> pd.DataFrame:
    """
    Batch extraction of 24 handcrafted features.

    Args:
        codes:         list or pd.Series of code strings
        show_progress: show tqdm progress bar

    Returns:
        pd.DataFrame (n_samples × 24) with HANDCRAFTED_24 columns
    """
    if not isinstance(codes, pd.Series):
        codes = pd.Series(codes)

    try:
        from tqdm import tqdm as _tqdm
        _tqdm_ok = True
    except ImportError:
        _tqdm_ok = False

    iterator = codes.items()
    if show_progress and _tqdm_ok:
        iterator = _tqdm(iterator, desc='24 features', total=len(codes))

    records = []
    for _, code in iterator:
        try:
            records.append(extract_24_features(code))
        except Exception:
            records.append({k: 0.0 for k in HANDCRAFTED_24})

    return pd.DataFrame(records, index=codes.index)


if __name__ == '__main__':
    sample_codes = [
        "def hello():\n    print('world')\n    # a comment\n",
        "for i in range(10): print(i)",
        "int main(){\n    int x=1;\n    return x;\n}",
    ]
    # Full feature set
    df_all = extract_all_features(sample_codes)
    print("extract_all_features shape:", df_all.shape)

    # 24-feature set
    df_24 = extract_24_features_batch(sample_codes, show_progress=False)
    print("extract_24_features_batch shape:", df_24.shape)
    print("Columns:", df_24.columns.tolist())
    print(df_24.round(3).to_string())

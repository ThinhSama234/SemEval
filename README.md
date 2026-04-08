# SemEval-2026 Task 13: Detecting Machine-Generated Code with Multiple Programming Languages, Generators, and Application Scenarios
## 🔍 Task Overview

The rise of generative models has made it increasingly difficult to distinguish machine-generated code from human-written code — especially across different programming languages, domains, and generation techniques. 

**SemEval-2026 Task 13** challenges participants to build systems that can **detect machine-generated code** under diverse conditions by evaluating generalization to unseen languages, generator families, and code application scenarios.

The task consists of **three subtasks**:

---

### Subtask A: Binary Machine-Generated Code Detection

**Goal:**  
Given a code snippet, predict whether it is:

- **(i)** Fully **human-written**, or  
- **(ii)** Fully **machine-generated**

**Training Languages:** `C++`, `Python`, `Java`  
**Training Domain:** `Algorithmic` (e.g., Leetcode-style problems)

**Evaluation Settings:**

| Setting                              | Language                | Domain                 |
|--------------------------------------|-------------------------|------------------------|
| (i) Seen Languages & Seen Domains    | C++, Python, Java       | Algorithmic            |
| (ii) Unseen Languages & Seen Domains | Go, PHP, C#, C, JS      | Algorithmic            |
| (iii) Seen Languages & Unseen Domains| C++, Python, Java       | Research, Production   |
| (iv) Unseen Languages & Domains      | Go, PHP, C#, C, JS      | Research, Production   |

**Dataset Size**: 
- Train - 500K samples (238K Human-Written | 262K Machine-Generated)
- Validation - 100K samples

**Target Metric** - Macro F1-score (we will build the leaderboard based on it), but you are free to use whatever works best for your approach during training.

---

## 📁 Data Format

- All data will be released via:
  - [Kaggle](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13)  
  - [HuggingFace Datasets](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
  - In this GitHub repo as `.parquet` file


## 📤 Submission Format

- Submit a `.csv` file with two columns:
  - `id`: unique identifier of the code snippet  
  - `label`: the **label ID** (not the string label)

- Sample submission files are available in each task’s folder  
- A **single scorer script** (`scorer.py`) is used for all subtasks  
- Evaluation measure: **macro F1** for all subtasks



## Citation
Our task is based on enriched data from our previous works. Please, consider citing them, when using data from this task

Droid: A Resource Suite for AI-Generated Code Detection
```
@inproceedings{orel2025droid,
  title={Droid: A resource suite for ai-generated code detection},
  author={Orel, Daniil and Paul, Indraneil and Gurevych, Iryna and Nakov, Preslav},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={31251--31277},
  year={2025}
}
```

CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings
```
@inproceedings{orel-etal-2025-codet,
    title = "{C}o{D}et-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings",
    author = "Orel, Daniil  and
      Azizov, Dilshod  and
      Nakov, Preslav",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.550/",
    pages = "10570--10593",
    ISBN = "979-8-89176-256-5",
}
```


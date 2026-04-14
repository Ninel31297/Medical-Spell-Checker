# 🧠 Medical Text Spell Checker (SymSpell + N-Gram + Gradio)

An adaptive **medical-domain spell checking system** combining:
- SymSpell for fast edit-distance correction
- N-gram language model for contextual ranking
- Dynamic dictionary updates from user input
- Interactive Gradio UI for real-time correction and learning

---

## 🚀 Features

- 🔤 Fast spelling correction using **SymSpell**
- 📊 Context-aware ranking using **N-gram language model**
- 🧠 Dynamic vocabulary expansion (online learning)
- 📂 Batch processing of text files
- 🧾 Single-word and phrase suggestions
- 🖥️ Web interface built with **Gradio**
- 💾 Persistent model saving/loading (pickle-based)

---

## 🏗️ System Architecture

The system consists of three core components:

### 1. Text Preprocessing
- Lowercasing
- Regex-based cleaning
- Tokenization
- Noise removal (punctuation, symbols)

### 2. SymSpell Module
- Dictionary-based candidate generation
- Edit distance search (fast approximate matching)
- Compound phrase correction

### 3. N-Gram Language Model
- Trigram-based probabilistic scoring
- Laplace smoothing
- Context-aware ranking of candidates

---

## 📦 Installation

```bash
pip install symspellpy nltk gradio pandas tqdm
````markdown
# Download required NLTK resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
````

---

## ▶️ How to Run

```bash
python app.py
```

Then open the browser interface automatically (or manually via Gradio link).

---

## 🧪 Usage

### 🔍 Check a Word

* Enter a word or phrase
* Get ranked spelling suggestions
* Correct matches highlighted

---

### 📄 Batch File Check

* Upload `.txt` file (one word per line)
* Receive correction table with suggestions

---

### ➕ Add to Dictionary

* Add single words manually
* Upload word lists
* Extract vocabulary from text files

---

### 💾 Save Models

* Persist updated dictionary and models for future sessions

---

## 🧠 Model Details

### SymSpell

* Maximum edit distance: 2
* Prefix-based pruning for speed

### N-Gram Model

* Supports unigram, bigram, trigram contexts
* Laplace smoothing for probability estimation

---

## 📁 Project Structure

```text
spell-checker/

├── spell_checker.py        # Core logic (SymSpell + NGram + utils)
├── app.py                  # Gradio interface
├── freq_dict.pkl           # Frequency dictionary
├── ngram_model.pkl         # Trained N-gram model
├── symspell.pkl            # SymSpell index
└── README.md
```

---

## ⚙️ Key Technologies

* Python 🐍
* SymSpellPy
* NLTK
* Scikit-learn (support utilities)
* Gradio (UI)
* N-gram Language Models

---

## 📊 Applications

* Medical text correction
* Scientific document preprocessing
* OCR post-processing
* Domain-specific spell checking systems

---

## 👤 Author

**Nina Galimullina**
Master’s Thesis Project — Medical NLP Systems

---

## 📜 License

This project is released under the MIT License (recommended for open academic projects).

---

## ⚠️ Notes

* Large pickle models are not included in the repository (add via `.gitignore`)
* Designed for research and educational use

```

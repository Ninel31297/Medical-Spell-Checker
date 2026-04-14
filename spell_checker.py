# spell_checker.py

import re
import sys
import os
import tempfile
import pickle
from collections import Counter, defaultdict
from itertools import islice, product
from symspellpy import SymSpell, Verbosity
from nltk.util import ngrams
from difflib import SequenceMatcher

# Предкомпилируем регулярки
CLEANER_RE = re.compile(r"[^\w\s’'-]")
TOKENIZER_RE = re.compile(r"\b[\w’-]+\b")

def clean_line(line):
    line = line.lower()
    line = CLEANER_RE.sub('', line)
    tokens = TOKENIZER_RE.findall(line)
    return ' '.join(tokens) if tokens else ''

def clean_file(input_path, output_path):
    # Подсчёт общего количества строк
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin, 1):
            cleaned = clean_line(line)
            if cleaned:
                fout.write(cleaned + '\n')

            # Прогресс
            percent = i / total_lines * 100
            sys.stdout.write(f"\rОбработано: {i}/{total_lines} строк ({percent:.2f}%)")
            sys.stdout.flush()

    print("\nГотово.")

# ===== Частотный словарь ===
def train_frequency(cleaned_file_path):
    freq_dict = defaultdict(int)

    with open(cleaned_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()  # строка уже очищена и токены разделены пробелами
            for word in words:
                freq_dict[word] += 1

    return freq_dict

def save_freq_dict_to_file(freq_dict, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for word, freq in freq_dict.items():
            f.write(f"{word} {freq}\n")


# step 3: SymSpell
class SymSpellWrapper:
    def __init__(self, max_edit_distance=2, prefix_length=7):
        # Инициализация SymSpell с заданным максимальным расстоянием редактирования и длиной префикса
        self.symspell = SymSpell(max_edit_distance, prefix_length)
        self.freq_dict = defaultdict(int)  # для хранения частот слов, если нужно отдельно

    def load_dictionary(self, dictionary_path, term_index=0, count_index=1, separator=" "):
        # Загрузка словаря из файла с частотами (обычно формат: слово<пробел>частота)
        self.symspell.load_dictionary(dictionary_path, term_index, count_index, separator)

    def add_word(self, word, frequency=1):
        # Добавление слова вручную с частотой
        self.freq_dict[word] += frequency
        self.symspell.create_dictionary_entry(word, self.freq_dict[word])

    def lookup(self, word, max_edit_distance=2, verbosity=Verbosity.ALL):
        # Получаем список кандидатов исправления
        suggestions = self.symspell.lookup(word, verbosity=verbosity, max_edit_distance=max_edit_distance)
        # Возвращаем список (слово, частота)
        return [(s.term, s.count) for s in suggestions]

    def lookup_compound(self, phrase, max_edit_distance=2):
        # Позволяет исправлять составные термины/фразы с учётом словарного контекста
        suggestions = self.symspell.lookup_compound(phrase, max_edit_distance=max_edit_distance)
        return [(s.term, s.count) for s in suggestions]

        # monkey patch для сохранения SymSpel
    def save_pickle(self, path):
        with open(path, 'wb') as f:
          pickle.dump(self, f)


class NGramModel:
    def __init__(self, n=3, vocab=None):
        """
        n - порядок модели (например, 3 для триграмм)
        vocab - необязательное начальное множество слов
        """
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set(vocab) if vocab else set()

    def update(self, text):
        # Предполагается, что text уже очищен и токены разделены пробелами
        words = text.lower().split()
        self.vocab.update(words)

        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n])
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def prob(self, context, word):
        context = tuple(context)
        ngram = context + (word,)
        vocab_size = max(len(self.vocab), 1)

        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        # Простой Laplace-сглажённый подсчёт вероятности
        return (ngram_count + 1) / (context_count + vocab_size)

    def train_from_file(self, cleaned_file_path):
        """
        Обучение модели по очищенному файлу с текстом,
        где каждая строка — уже очищенная и нормализованная.
        """
        with open(cleaned_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # пропускаем пустые строки
                    self.update(line)

def suggest_candidates(phrase, symspell_wrapper, freq_dict, ngram_model, max_terms=3, top_n=5, alpha=5):
    phrase = clean_line(phrase)
    terms = phrase.split()
    if len(terms) > max_terms:
        terms = terms[:max_terms]
    phrase_truncated = ' '.join(terms)

    # === Одно слово — возвращаем топ от SymSpell напрямую ===
    if len(terms) == 1:
        word = terms[0]
        symspell_results = symspell_wrapper.lookup(word, max_edit_distance=2)
        suggestions = [w for w, _ in symspell_results]
        if not suggestions:
            return [word]
        return suggestions[:top_n]

    # === Поиск похожих многословных фраз ===
    full_phrase_matches = [
        term for term in freq_dict
        if ' ' in term and SequenceMatcher(None, term, phrase_truncated).ratio() > 0.8
    ]
    full_phrase_matches = sorted(full_phrase_matches, key=lambda w: freq_dict.get(w, 0), reverse=True)
    if full_phrase_matches:
        return full_phrase_matches[:top_n]

    # === Кандидаты по словам ===
    all_term_candidates = []
    for word in terms:
        symspell_results = symspell_wrapper.lookup(word, max_edit_distance=2)
        candidates = [w for w, _ in symspell_results]
        if not candidates:
            candidates = [word]
        candidates = sorted(candidates, key=lambda w: freq_dict.get(w, 0), reverse=True)[:5]
        all_term_candidates.append(candidates)

    # Защита от пустых списков кандидатов
    if not all_term_candidates or any(len(c) == 0 for c in all_term_candidates):
        return [phrase_truncated]

    # === Комбинации ===
    combined_candidates = [' '.join(words) for words in product(*all_term_candidates)]

    def combo_score(phrase):
        words = phrase.split()
        base_score = freq_dict.get(phrase, 0)
        if base_score == 0:
            base_score = sum(freq_dict.get(w, 1) for w in words)
        if len(words) >= ngram_model.n:
            context = words[-(ngram_model.n - 1):-1]
            word = words[-1]
            context_score = ngram_model.prob(context, word)
            if context_score is None:
                context_score = 0
            return base_score + alpha * context_score
        return base_score

    combined_candidates = sorted(set(combined_candidates), key=combo_score, reverse=True)

    if not combined_candidates:
        return [phrase_truncated]

    return combined_candidates[:top_n]


def add_to_dictionary(text_or_path, freq_dict, symspell, ngram_model, frequency=15, max_ngram=3, status_callback=None):
    if status_callback: status_callback("🔄 Начинаем обновление словаря...")

    if isinstance(text_or_path, str) and os.path.isfile(text_or_path):
        if status_callback: status_callback("📁 Очистка файла...")

        with tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8', suffix='.txt') as temp_file:
            cleaned_path = temp_file.name
        clean_file(text_or_path, cleaned_path)

        if status_callback: status_callback("📊 Обучение N-gram модели...")
        ngram_model.train_from_file(cleaned_path)

        if status_callback: status_callback("📚 Обновление словарей...")
        with open(cleaned_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    freq_dict[word] += frequency
                    symspell.add_word(word, freq_dict[word])
                for n in range(2, max_ngram + 1):
                    for gram in ngrams(words, n):
                        phrase = ' '.join(gram)
                        freq_dict[phrase] += frequency

        os.remove(cleaned_path)
        if status_callback: status_callback("✅ Завершено: данные из файла обработаны.")
    else:
        if status_callback: status_callback("📝 Обработка строки...")
        cleaned_text = clean_line(text_or_path)
        words = cleaned_text.split()
        ngram_model.update(cleaned_text)
        for word in words:
            freq_dict[word] += frequency
            symspell.add_word(word, freq_dict[word])
        for n in range(2, max_ngram + 1):
            for gram in ngrams(words, n):
                phrase = ' '.join(gram)
                freq_dict[phrase] += frequency
        if status_callback: status_callback("✅ Завершено: текст добавлен.")
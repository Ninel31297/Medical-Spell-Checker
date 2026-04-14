import threading
import time
import os
import pickle
import gradio as gr
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from spell_checker import (
    suggest_candidates,
    add_to_dictionary, 
    SymSpellWrapper, 
    NGramModel)


# Вместо загрузки из pickle сделай так:
symspell = SymSpellWrapper()
ngram_model = NGramModel(n=3)


# === Загрузка моделей с прогресс-баром ===
def load_model(file_path):
    print(f"📦 Загрузка {file_path}...")
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return file_path, model

model_files = ['freq_dict.pkl', 'ngram_model.pkl', 'symspell.pkl']
models = {}

print("🚀 Загрузка моделей...\n")

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(load_model, model_files), total=len(model_files), desc="📊 Загрузка моделей"))

for file_path, model in results:
    model_name = file_path.split('.')[0]
    models[model_name] = model
    print(f"✅ {model_name} загружен.")

freq_dict = models['freq_dict']
ngram_model = models['ngram_model']
symspell = models['symspell']

print("\n✅ Все модели успешно загружены.\n")


# === Сохранение ===
def save_pickle_models():
    with open("freq_dict.pkl", "wb") as f:
        pickle.dump(freq_dict, f)
    with open("ngram_model.pkl", "wb") as f:
        pickle.dump(ngram_model, f)
    with open("symspell.pkl", "wb") as f:
        pickle.dump(symspell, f)

# === Gradio-функции ===
def gr_suggest_candidates(phrase):
    candidates = suggest_candidates(phrase, symspell, freq_dict, ngram_model)

    formatted = []
    for cand in candidates:
        if cand == phrase:
            formatted.append(f'<span style="color: red">{cand}</span>')
        else:
            formatted.append(cand)

    return ", ".join(formatted)

def check_words_from_file(file_path, symspell, freq_dict, ngram_model):
    if file_path is None:
        return "<p style='color: red'>Файл не загружен</p>"

    with open(file_path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f.read().splitlines() if w.strip()]

    table_rows = ["<tr><th>Слово</th><th>Кандидаты</th></tr>"]
    for word in words:
        candidates = suggest_candidates(word, symspell, freq_dict, ngram_model)

        formatted_candidates = []
        for cand in candidates:
            if cand == word:
                formatted_candidates.append(f'<span style="color: red">{cand}</span>')
            else:
                formatted_candidates.append(cand)

        row = f"<tr><td>{word}</td><td>{', '.join(formatted_candidates)}</td></tr>"
        table_rows.append(row)

    html_table = "<table border='1' style='border-collapse: collapse; width: 100%;'>" + "".join(table_rows) + "</table>"
    return html_table

def gr_add_single_word(word):
    if not word.strip():
        yield "⚠️ Пустое слово"
        return

    messages = []

    def status_callback(msg):
        messages.append(msg)

    add_to_dictionary(word.strip(), freq_dict, symspell, ngram_model, status_callback=status_callback)

    yield "\n".join(messages)
    yield f"✅ Добавлено: {word.strip()}"


def gr_add_word_list(file):
    if file is None:
        yield "⚠️ Файл не загружен"
        return

    try:
        with open(file.name, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

        total = len(words)
        if total == 0:
            yield "⚠️ Файл пуст"
            return

        for i, word in enumerate(words, 1):
            messages = []

            def status_callback(msg):
                messages.append(msg)

            add_to_dictionary(word, freq_dict, symspell, ngram_model, status_callback=status_callback)

            yield f"📌 {i}/{total}: {word}\n" + "\n".join(messages)

        yield f"✅ Добавлено {total} слов из списка"
    except Exception as e:
        yield f"❌ Ошибка: {e}"


def gr_add_text(file):
    if file is None:
        yield "⚠️ Файл не выбран"
        return

    messages = []

    def status_callback(msg):
        messages.append(msg)

    try:
        add_to_dictionary(file.name, freq_dict, symspell, ngram_model, status_callback=status_callback)
        yield "\n".join(messages)
        yield "✅ Добавлено из файла и словарь обновлён"
    except Exception as e:
        yield f"❌ Ошибка при обработке файла: {e}"


def gr_save_models():
    messages = []

    def log(msg):
        messages.append(msg)
        yield "\n".join(messages)

    try:
        yield from log("💾 Сохранение частотного словаря...")
        with open("freq_dict.pkl", "wb") as f:
            pickle.dump(freq_dict, f)

        yield from log("💾 Сохранение модели N-грамм...")
        with open("ngram_model.pkl", "wb") as f:
            pickle.dump(ngram_model, f)

        yield from log("💾 Сохранение SymSpell модели...")
        with open("symspell.pkl", "wb") as f:
            pickle.dump(symspell, f)

        yield from log("✅ Все словари успешно сохранены.")
    except Exception as e:
        yield f"❌ Ошибка при сохранении: {e}"

# === Интерфейс Gradio ===
with gr.Blocks(title="Проверка орфографии") as demo:
    gr.Markdown("## 📝 Проверка орфографии")

    with gr.Tab("Проверка одного слова"):
        gr.Markdown(
            "🔍 Введите одно слово или короткую фразу, чтобы получить список предложений на замену. "
            "Если слово совпадёт со словом в словаре — оно будет выделено **красным** в списке результатов."
        )
        word_input = gr.Textbox(label="Введите слово")
        word_output = gr.HTML(label="Предложения")  # HTML для подсветки
        word_button = gr.Button("Проверить")
        word_button.click(fn=gr_suggest_candidates, inputs=word_input, outputs=word_output)

    with gr.Tab("Проверка списка слов"):
        gr.Markdown(
            "📂 Загрузите `.txt` файл, где **каждое слово на новой строке**. "
            "Словарь подберёт кандидатов на замену для каждого слова. "
            "Если слово совпадёт со словом в словаре — оно будет выделено **красным** в списке результатов."
        )
        file_input = gr.File(file_types=[".txt"], label="Загрузите файл (.txt) со словами (по одному в строке)")
        file_output = gr.HTML(label="Результаты")  # HTML таблица вместо Dataframe
        file_button = gr.Button("Проверить файл")
        file_button.click(
            fn=lambda f: check_words_from_file(f, symspell, freq_dict, ngram_model),
            inputs=file_input,
            outputs=file_output
        )

    with gr.Tab("Обновление словаря"):
        gr.Markdown("### ➕ Добавление одного слова")
        gr.Markdown(
            "Введите редкое или отсутствующее в словаре слово, чтобы добавить его. "
            "Это полезно для терминов, имён или сокращений."
        )
        add_single = gr.Textbox(label="Слово")
        add_single_btn = gr.Button("Добавить")
        add_single_output = gr.Textbox(label="Результат", lines=10, interactive=True)
        add_single_btn.click(
            fn=gr_add_single_word,
            inputs=add_single,
            outputs=add_single_output,
            show_progress=True
        )

        gr.Markdown("### 📃 Добавление списка слов")
        gr.Markdown(
            "Загрузите `.txt` файл, где **по одному слову в строке**. "
            "Все слова из файла будут добавлены в словарь."
        )
        add_list = gr.File(file_types=[".txt"], label="Загрузите файл со словами")
        add_list_btn = gr.Button("Добавить список")
        add_list_output = gr.Textbox(label="Результат", lines=10, interactive=True)
        add_list_btn.click(
            fn=gr_add_word_list,
            inputs=add_list,
            outputs=add_list_output,
            show_progress=True
        )

        gr.Markdown("### 📄 Добавление слов из текста")
        gr.Markdown(
            "Загрузите текстовый файл — слова из него будут автоматически извлечены, очищены и добавлены в словарь. "
            "Подходит для добавления новых терминов из научных статей или описаний."
        )
        add_text_file = gr.File(file_types=[".txt"], label="Загрузите текстовый файл")
        add_text_btn = gr.Button("Добавить из файла")
        add_text_output = gr.Textbox(label="Результат", lines=10, interactive=True)
        add_text_btn.click(
            fn=gr_add_text,
            inputs=add_text_file,
            outputs=add_text_output,
            show_progress=True
        )

        gr.Markdown("### 💾 Сохранение изменений")
        gr.Markdown(
        "Обновляет все компоненты словаря и обновляет его после добавления новых слов. "
        "Обязательно сохраняйте изменения перед выходом, чтобы не потерять новую информацию."
        )
        save_btn = gr.Button("Сохранить словари", variant="primary")  # 🌟 Подсвеченная кнопка
        save_output = gr.Textbox(label="Результат", lines=10, interactive=True)
        save_btn.click(fn=gr_save_models, outputs=save_output, show_progress=True)




    status_output = gr.Markdown(visible=False)

    def stop_app():
        def _stop():
            time.sleep(0.3)
            demo.close()
            os._exit(0)
        threading.Thread(target=_stop).start()
        return gr.update(visible=True, value="**<span style='color: green; font-weight: bold;'>Выход выполнен ✅</span>**")

    with gr.Row(variant="compact", equal_height=True):
        stop_btn = gr.Button("Выход из приложения", variant="stop")

    stop_btn.click(fn=stop_app, inputs=None, outputs=status_output)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
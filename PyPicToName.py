import argparse
from transformers import pipeline, AutoTokenizer
import translators as ts
import os
import progressbar
import warnings
import random

# Игнорирование всех предупреждений
warnings.filterwarnings("ignore")

# Определяет количество одинаковых файлов
def count_files(directory, filename):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and name == filename])

def im_to_text(path):
    # Загружаем предобученный токенизатор и модель для описания изображений
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", tokenizer=tokenizer)
    # Получаем описание изображения
    text = captioner(path)
    # Переводим описание с английского на русский
    return ts.translate_text(text[0]['generated_text'], from_language='en', to_language='ru')

# Создаем парсер аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument('filepaths', nargs='+', type=str)

# Разбираем аргументы
args = parser.parse_args()

i = 0
maxval = 100
bar = progressbar.ProgressBar(maxval=maxval, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

bar.start()
for path in args.filepaths:
    name = path.split('\\')[-1]
    glav_name = name[:-4]
    result = im_to_text(path)
    try:
        # Переименовываем файл в соответствии с описанием изображения
        os.rename(name, f'{result}.jpg')
    except:
        # Если возникла ошибка, добавляем случайное число к имени файла
        os.rename(name, f'{result} {random.randint(0, 100)}.jpg')

    i = i + 1
    bar.update(maxval / len(args.filepaths) * i)
    #print(f' {i} / {len(args.filepaths)}')
bar.finish()

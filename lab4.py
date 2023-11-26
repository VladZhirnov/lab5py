import os
import pandas as pd
from PIL import Image, UnidentifiedImageError

df = pd.DataFrame(columns=["ClassName", "Directory"])

annotation_file = "annotation1.csv"

with open(annotation_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.split(",")
        df.loc[len(df.index)] = [line[2].strip(), line[0].strip()]

df['mark'] = 0

df.loc[df["ClassName"] == "tiger", "mark"] = 0
df.loc[df["ClassName"] == "leopard", "mark"] = 1

df['height'] = 0
df['width'] = 0
df['channels'] = 0

for index, row in df.iterrows():
    image_path = row['Directory']
    absolute_path = os.path.abspath(image_path)
    try:
        image = Image.open(absolute_path)
        height, width = image.size
        channels = len(image.getbands())
        df.at[index, 'height'] = height
        df.at[index, 'width'] = width
        df.at[index, 'channels'] = channels
    except FileNotFoundError as e:
        print(f"Ошибка при обработке файла {absolute_path}: {e}")
        continue
    except UnidentifiedImageError as e:
        print(f"Ошибка при обработке файла {absolute_path}: {e}")
        continue

print(df)

# Выводим статистическую информацию о размерах изображений
image_size_stats = df[['height', 'width', 'channels']].describe()

# Выводим статистическую информацию о метках класса
class_label_stats = df['mark'].value_counts()

# Определяем, является ли набор данных сбалансированным
is_balanced = class_label_stats.min() / class_label_stats.max() > 0.8 

print("Статистика размеров изображений:")
print(image_size_stats)
print("\nСтатистика по меткам класса:")
print(class_label_stats)
print("\nНабор данных является сбалансированным:", is_balanced)

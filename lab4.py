import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.DataFrame(columns=["class_name", "Path_to_image"])

def create_DataFrame():
    """Create DataFrame"""
    annotation_file = "annotation1.csv"

    with open(annotation_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.split(",")
            df.loc[len(df.index)] = [line[2].strip(), line[0].strip()]

    df['mark'] = 0

    df.loc[df["class_name"] == "tiger", "mark"] = 0
    df.loc[df["class_name"] == "leopard", "mark"] = 1
create_DataFrame()

def find_image_size():
    """Adds three columns of image characteristics to the DataFrame and fills them"""
    df['height'] = 0
    df['width'] = 0
    df['channels'] = 0
    for index, row in df.iterrows():
        image_path = row['Path_to_image']
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

find_image_size()


def check_balance(class_label_stats):
    """Checks if the set is balanced"""
    if (class_label_stats.min() == class_label_stats.max()):
        return True
    else:
        return False
    
def statistical_info():
    """Сalculate statistical information"""
    # Выводим статистическую информацию о размерах изображений
    image_size_stats = df[['height', 'width', 'channels']].describe()

    # Выводим статистическую информацию о метках класса
    class_label_stats = df['mark'].value_counts()

    # Определяем, является ли набор данных сбалансированным
    is_balanced = check_balance(class_label_stats)

    print("Статистика размеров изображений:")
    print(image_size_stats)
    print("\nСтатистика по меткам класса:")
    print(class_label_stats)
    print("\nНабор данных является сбалансированным:", is_balanced)

statistical_info()


def filter_dataframe_by_class(df, target_class):
    """Filters a DataFrame by class"""
    filtered_df = df[df['mark'] == target_class].copy()
    return filtered_df

def filter_DataFrame(target_class):
    """Calls the Filter DataFrame by class function and outputs it"""
    filtered_df = filter_dataframe_by_class(df, target_class)

    print("\nОтфильтрованный DataFrame для метки класса", target_class, ":")
    print(filtered_df)

filter_DataFrame(0)


def filter_dataframe_by_size_and_class(df, target_class, max_width, max_height):
    """Filters DataFrame by class and size"""
    filtered_df = df[(df['mark'] == target_class) & (df['width'] <= max_width) & (df['height'] <= max_height)].copy()
    return filtered_df

def filter_DataFrame_with_parameters(target_class, max_width, max_height):
    """Calls the Filter DataFrame by class and size function and outputs it"""
    filtered_df = filter_dataframe_by_size_and_class(df, target_class, max_width, max_height)

    print("\nОтфильтрованный DataFrame для метки класса", target_class, "и размеров (width <= {}, height <= {}):".format(max_width, max_height))
    print(filtered_df)

filter_DataFrame_with_parameters(1, 400, 300)


def group_and_find_pixel_values():
    """Groups a DataFrame by class, creates a column with the number of pixels, and finds the minimum, maximum, and average pixel values"""
    df['pixel_count'] = df['width'] * df['height']

    # Группируем DataFrame по метке класса
    grouped_df = df.groupby('mark')

    max_pixel_count = grouped_df['pixel_count'].max()
    min_pixel_count = grouped_df['pixel_count'].min()
    mean_pixel_count = grouped_df['pixel_count'].mean()

    result_df = pd.DataFrame({
        'max_pixel_count': max_pixel_count,
        'min_pixel_count': min_pixel_count,
        'mean_pixel_count': mean_pixel_count
    })

    print(df)
    print(result_df)

group_and_find_pixel_values()


def generate_histogram(df, target_class):
    """Calculates histograms for each image channel"""
    # Фильтруем DataFrame для выбора случайного изображения заданного класса
    filtered_df = df[df['mark'] == target_class]
    random_image_row = filtered_df.sample(n=1).iloc[0]

    image = cv2.imread(random_image_row['Path_to_image'])
        
    # Преобразуем изображение в пространство BGR
    b, g, r = cv2.split(image)

    # Вычисляем гистограммы для каждого канала
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Нормализуем гистограммы
    hist_b = hist_b / hist_b.sum()
    hist_g = hist_g / hist_g.sum()
    hist_r = hist_r / hist_r.sum()

    return hist_b, hist_g, hist_r


def create_histogram_graph(target_class):
    """Plot each histogram"""
    hist_b, hist_g, hist_r = generate_histogram(df, target_class)

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    axs[0].plot(hist_b)
    axs[0].set_title('Гистограмма по каналу B')
    axs[0].set_xlabel('Интенсивность пикселя')
    axs[0].set_ylabel('Относительная частота')

    axs[1].plot(hist_g)
    axs[1].set_title('Гистограмма по каналу G')
    axs[1].set_xlabel('Интенсивность пикселя')
    axs[1].set_ylabel('Относительная частота')

    axs[2].plot(hist_r)
    axs[2].set_title('Гистограмма по каналу R')
    axs[2].set_xlabel('Интенсивность пикселя')
    axs[2].set_ylabel('Относительная частота')

    plt.tight_layout()
    plt.show()

create_histogram_graph(1)
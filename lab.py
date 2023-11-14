import os
from time import sleep
import requests
from bs4 import BeautifulSoup


def createFolders(name : str) -> None:
    try:
        if not os.path.exists("dataset"):
            os.makedirs(os.path.join("dataset", name))
        elif  not os.path.exists(os.path.join("dataset", name)):
            os.mkdir(os.path.join("dataset", name))
    except Exception:
        print("Error creating folder!")


def saveImg(imgurl : str, filename : str) -> None:
    try:
        html = requests.get(imgurl, timeout = 10)
        if html.ok:
            with open(filename, 'wb') as file:
                file.write(html.content)
    except Exception:
        print(f'Error saving image: {imgurl}')  


def ImgParser(name : str, url : str) -> None:
    createFolders(name)
    list_length = len(os.listdir(os.path.join("dataset", name))) + 2
    for page in range(list_length // 30, 34):
        html = requests.get(url + f"&p={page}", timeout = 10)
        soup = BeautifulSoup(html.content, "lxml")
        images = soup.findAll("img", class_ = "serp-item__thumb justifier__thumb")
        if len(images) == 0:
            print(soup.text)
            break
        for i in range(list_length % 30, len(images)):
            try:
                url_with_tag = images[i].get("src") 
                filename = make_filename(list_length, name)
                saveImg("http:" + url_with_tag, filename)
                print(f'Image №: {list_length}')
                sleep(30)
                list_length += 1
                if list_length > 1000:
                    break
            except Exception:
                print(f'Error! Image: {list_length}')
                continue


def make_filename(i : int, name : str) -> str:
    filename = f"{i:04d}.jpg"
    return os.path.join("dataset", name, filename)


if __name__ == "__main__":
    ImgParser("tiger", "https://www.yandex.ru/images/search?text=tiger")
    ImgParser("leopard", "https://www.yandex.ru/images/search?text=leopard")
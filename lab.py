import os
from time import sleep
import requests
from bs4 import BeautifulSoup
from fake_headers import Headers


def createFolders(name:str)->None:
    if not os.path.exists("dataset"):
        os.makedirs(f"dataset/{name}")
    elif  not os.path.exists(f"dataset/{name}"):
        os.mkdir(f"dataset/{name}")


def saveImg(imgurl: str, filename: str) -> None:
    try:
        response = requests.get(imgurl, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    if chunk:
                        file.write(chunk)
    except (Exception, requests.exceptions.RequestException) as E:
        print('Ошибка при загрузке: ', imgurl, ':', str(E))


def ImgParser(name : str) -> []:
    createFolders(name)
    i = 0
    for page in range(20):
        url = f"https://yandex.ru/images/search?from=tabbar&text={name}&p={page}"
        headers = Headers( 
            browser="chrome",
            os="win",
            headers=True
            ).generate()
        html = requests.get(url, headers)
        soup = BeautifulSoup(html.content, "lxml")
        tags = soup.findAll("img",
                            class_ = "serp-item__thumb justifier__thumb"
                            )
        for tag in tags:
            try:
                tag = tag.get("src")
                str_i = str(i)
                if (i < 10):
                    leading_zeroes = '0' * 3
                else:
                    if (9 < i < 100):
                        leading_zeroes = '0' * 2
                    else:
                        if (i > 99):
                            leading_zeroes = '0' * 1
                formatted_i = leading_zeroes + str_i
                filename = f"dataset/{name}/{formatted_i}.jpg"
                saveImg("http:" + tag, filename)
                i += 1
                print(i)
                sleep(1)
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    ImgParser("tiger")
    ImgParser("leopard")
    
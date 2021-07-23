from bs4 import BeautifulSoup
import requests
import os

def listFD(url, ext=''):
    page = requests.get(url).text
    # print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    folders = os.listdir(os.path.join(dirname, "mozart_piano_sonatas"))
    print(os.listdir(os.path.join(dirname, "mozart_piano_sonatas", folders[0])))

    import dgl

    g = dgl.heterograph({('user', '+1', 'movie') : [(0, 0), (0, 1), (1, 0)], ('user', '-1', 'movie') : [(2, 1)]})

    print(g.edges("+1"))
    
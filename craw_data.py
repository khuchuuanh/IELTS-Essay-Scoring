import concurrent.futures
import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

def total_ieltsScore(scores):
    avg = sum(map(float, scores))/len(scores)
    return int(avg)+0.5 if avg %1 >= 0.25 and avg % 1 < 0.75 else int(avg) +1 if avg %1 >= 0.75 else int(avg)

def get_data(link):
    URL = 'https://writing9.com' + link
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    for div in soup.find_all('div', {'class':'jsx-1879403401 hint'}):
        div.decompose()
    topic = soup.find('h1', {'class':'jsx-3594441866 h4'})
    essay = soup.find('span',{'class': 'jsx-242105113 t18'})
    scores = [i.get_text() for i in soup.find_all('span', {'class': 'jsx-2601907021 caps'})][:4]


    score_names = []
    score_values = []
    for score in scores:
        score_name, score_value = score.split(':')
        score_names.append(score_name)
        score_values.append(score_value)

    return [topic.text, essay.text, score_values]

for x in tqdm(range(5000, 6000, 2)):
    num_page = [i for i in range(x, x+2)]
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = []
            for num in num_page:
                base_URL = 'https://writing9.com/band/0/' + str(num)
                page = requests.get(base_URL)
                soup = BeautifulSoup(page.content, 'html.parser')
                links =[a['href'] for a in soup.find_all('a') if a.get('href','').startswith('/text/')]

                future_to_link = {executor.submit(get_data, link): link for link in links}
                for future in concurrent.futures.as_completed(future_to_link):
                    data.append(future.result())
        
        with open('scores.csv','a', newline= '', encoding= 'utf-8') as file:
            writer = csv.writer(file)
            for row in data:
                topic, essay, score_values = row
                x = total_ieltsScore(score_values)
                writer.writerow([topic]+[essay]+[value for value in score_values] +[x])

    except:
        continue


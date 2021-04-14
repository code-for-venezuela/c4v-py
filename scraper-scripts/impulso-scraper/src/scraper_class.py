from bs4 import BeautifulSoup
import requests

class ServiciosScraper:
	def __init__(self, source_url):
		self.text = ""
		self.source_url = source_url
		self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'}

	def elimpulso_scraper(self):
		page = requests.get(self.source_url, headers = self.headers)
		soup = BeautifulSoup(page.content, 'html.parser')
		final_text = ""

		class_for_text = soup.find_all('div', {'class': 'td-post-content'})
		for div in class_for_text:
			for p in div.find_all('p'):
				final_text += p.text + " "
		return final_text
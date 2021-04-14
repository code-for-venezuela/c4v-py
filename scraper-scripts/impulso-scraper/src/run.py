from scraper_class import ServiciosScraper
import csv

def main():
	# hard code for https://www.elimpulso.com

	reference_url = "https://www.elimpulso.com";
	text_data = []

	with open('../assets/servicios_source.csv') as csvfile, open('../assets/servicios_source_modified.csv', mode='w', newline='') as writefile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		csv_writer = csv.writer(writefile, delimiter=',')
		
		impulso_items = 0

		for row in csv_reader:
			if reference_url in row[2]:
				test_object = ServiciosScraper(row[2])
				row = [row[0], test_object.elimpulso_scraper(), row[2]]
				csv_writer.writerow(row)
				impulso_items += 1;
			else: 
				row = [row[0], row[1], row[2]]
				csv_writer.writerow(row)


	# with open('../assets/servicios_source.csv', mode='w', newline='') as csvwrite:
	# 	csv_writer = csv.writer(csvfile, delimiter=',')

	# 	for row in csv_writer:
	# 		if reference_url in row[2]:
if __name__ == "__main__":			
	print("Starting to scrape...")
	main()
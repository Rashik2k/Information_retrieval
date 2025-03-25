## **Project detail**
This project consists of two tasks aimed at developing tools for document retrieval and classification:

### Task 1: Search Engine
The goal is to design and implement a vertical search engine tailored to retrieve academic papers and books specifically authored by members of Coventry University’s School of Economics, Finance, and Accounting. The system crawls the relevant web pages and retrieves information about all available publications. For each publication, it extracts available data (such as authors, publication year, and title) and the links to both the publication page and the author's profile ("pureportal" profile) page.

### Task 2: Text Classification
This task involves creating a document classification system that can categorize texts into three distinct topics: Politics, Business, and Health. The system uses data extracted from New York Times in a csv file of 120 documents in each category.

#### **File Structure**
```
├── requisite.py          # Handles crawling and indexing of data
├── classification.ipynb  #Trains the Multinomial NB with the available dataset
├── app.py                # uses flask to show the UI in a website, main script to run the search engine and text classification based on three Categories:                                     Politics, Business and Health
```


#### **How to Run**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rashik2k/Search_Engine-and-Subject_Classification.git
   ```
2. **Install libraries from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```
3. **If we need to do a clean run from beginning then**
   - First delete inverted_index.json, publications.json, classifier.sav, vectorizer.sav files
   - Then run python requisite.py
   - Then run each cells from jupyter notebook i.e Classification.ipynb
3. **Run the application in website**:
   ```bash
   python app.py
   ```

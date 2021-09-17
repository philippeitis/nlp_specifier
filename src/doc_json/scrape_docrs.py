import requests
import lxml

DOCRS_URL = "https://docs.rs/"


def get_all_links(page):
    # do some lxml parsing.
    pass


if __name__ == '__main__':
    crate_name = "finnhub-rs"
    page = requests.get(DOCRS_URL + crate_name, allow_redirects=True)
    # Base Case: read Structs and Functions section
    # Recursively read Modules sections
    print(page.content)

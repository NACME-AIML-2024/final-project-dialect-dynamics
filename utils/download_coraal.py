import os

# Define the path to the text file containing the URLs and the destination folder
url_file_path = '/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics/data/urls_coraal.txt'
destination_folder = '/Users/romerocruzsa/Workspace/aiml/final-project-dialect-dynamics/data'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the URLs from the text file
with open(url_file_path, 'r') as file:
    urls = file.readlines()

# Download each file using wget
for url in urls:
    url = url.strip()  # Remove any leading/trailing whitespace
    if url:
        os.system(f'wget -P {destination_folder} {url}')
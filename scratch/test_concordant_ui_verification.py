import os
import requests
from bs4 import BeautifulSoup

def run_test():
    url = "http://127.0.0.1:5000/predict"
    
    # 1. Test Concordant Negative (Uninfected smear + non-severe symptoms)
    uninfected_path = os.path.abspath("data/sample_uninfected.png")
    if not os.path.exists(uninfected_path):
        # find any uninfected image in cell_images
        import glob
        matches = glob.glob("data/cell_images/**/Uninfected/*.png", recursive=True)
        if matches:
            uninfected_path = matches[0]
            
    parasitized_path = os.path.abspath("data/sample_parasitized.png")
    if not os.path.exists(parasitized_path):
        import glob
        matches = glob.glob("data/cell_images/**/Parasitized/*.png", recursive=True)
        if matches:
            parasitized_path = matches[0]

    print(f"Using uninfected: {uninfected_path}")
    print(f"Using parasitized: {parasitized_path}")

    # Case 1: Concordant Negative (Uninfected smear + non-severe symptoms)
    with open(uninfected_path, "rb") as f:
        files = {"smear_image": ("test_uninf.png", f, "image/png")}
        data = {
            "mode": "mode_c",
            "age": "30",
            "sex": "0",
            "fever": "1",
            "headace": "1"
        }
        res = requests.post(url, files=files, data=data)
        soup = BeautifulSoup(res.text, "html.parser")
        top_consensus = soup.select_one(".consensus-card .verdict-box, .glass-card .verdict-box")
        print("\n--- CASE 1: Concordant Negative ---")
        print("Status code:", res.status_code)
        # Look for the top verdict banner
        top_box = soup.find(lambda tag: tag.name == "div" and "verdict-box" in tag.get("class", []) and ("verdict-negative" in tag.get("class", []) or "verdict-critical" in tag.get("class", [])) and "OVERALL DIAGNOSTIC OUTCOME" in tag.text)
        if top_box:
            print("Found Top Consensus Verdict Box:")
            print("  Classes:", top_box.get("class"))
            print("  Title:", top_box.select_one(".verdict-main-heading").get_text(strip=True))
            print("  Explanation:", top_box.select_one(".verdict-explanation").get_text(strip=True))
        else:
            print("Top Consensus Verdict Box NOT FOUND")

    # Case 2: Concordant Positive (Parasitized smear + high severe symptoms)
    with open(parasitized_path, "rb") as f:
        files = {"smear_image": ("test_para.png", f, "image/png")}
        data = {
            "mode": "mode_c",
            "age": "4",
            "sex": "1",
            "fever": "1",
            "Convulsion": "1",
            "Anemia": "1",
            "hypoglycemia": "1",
            "prostraction": "1",
            "hyperpyrexia": "1"
        }
        res = requests.post(url, files=files, data=data)
        soup = BeautifulSoup(res.text, "html.parser")
        top_box = soup.find(lambda tag: tag.name == "div" and "verdict-box" in tag.get("class", []) and ("verdict-negative" in tag.get("class", []) or "verdict-critical" in tag.get("class", [])) and "OVERALL DIAGNOSTIC OUTCOME" in tag.text)
        print("\n--- CASE 2: Concordant Positive ---")
        print("Status code:", res.status_code)
        if top_box:
            print("Found Top Consensus Verdict Box:")
            print("  Classes:", top_box.get("class"))
            print("  Title:", top_box.select_one(".verdict-main-heading").get_text(strip=True))
            print("  Explanation:", top_box.select_one(".verdict-explanation").get_text(strip=True))
        else:
            print("Top Consensus Verdict Box NOT FOUND")

    # Case 3: Conflict (Uninfected smear + high severe symptoms)
    with open(uninfected_path, "rb") as f:
        files = {"smear_image": ("test_uninf.png", f, "image/png")}
        data = {
            "mode": "mode_c",
            "age": "4",
            "sex": "1",
            "fever": "1",
            "Convulsion": "1",
            "Anemia": "1",
            "hypoglycemia": "1",
            "prostraction": "1",
            "hyperpyrexia": "1"
        }
        res = requests.post(url, files=files, data=data)
        soup = BeautifulSoup(res.text, "html.parser")
        conflict = soup.select_one(".conflict-box")
        print("\n--- CASE 3: Conflict Case ---")
        print("Status code:", res.status_code)
        if conflict:
            print("Conflict banner found:", conflict.select_one(".conflict-title").get_text(strip=True))
        else:
            print("Conflict banner NOT FOUND")

if __name__ == "__main__":
    run_test()

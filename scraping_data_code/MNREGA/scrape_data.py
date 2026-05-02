import time
import re
import pandas as pd

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ================== CONFIGURATION ==================
# Starting District URL for Bathinda
DISTRICT_URL = "https://mnregaweb2.dord.gov.in/netnrega/Homedist.aspx?flag_debited=&is_statefund=&lflag=eng&district_code=2611&district_name=BHATINDA&state_name=PUNJAB&state_Code=26"

STATE_NAME = "PUNJAB"
DISTRICT_NAME = "BATHINDA"
# ===================================================

def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # Uncomment the line below if you want the browser to run silently in the background
    # options.add_argument("--headless") 
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_links_via_js(driver, keyword):
    """Searches the main page AND all frames for specific links, ignoring case."""
    return driver.execute_script(f"""
        let extractedData = [];
        
        function searchLinks(doc) {{
            if(!doc) return;
            let links = doc.querySelectorAll('a');
            for(let i = 0; i < links.length; i++) {{
                let href = links[i].href || "";
                let text = links[i].innerText.trim();
                
                if(href.toLowerCase().includes('{keyword.lower()}') && text) {{
                    if(!extractedData.some(item => item.name === text)) {{
                        extractedData.push({{'name': text, 'url': href}});
                    }}
                }}
            }}
        }}

        searchLinks(document);
        
        let frames = document.querySelectorAll('frame, iframe');
        for(let i = 0; i < frames.length; i++) {{
            try {{ searchLinks(frames[i].contentDocument); }} 
            catch(e) {{ }} 
        }}
        
        return extractedData;
    """)

# --- TARGETED DATA EXTRACTOR ---

def extract_demographics(driver, link_url):
    """Extracts demographic data from the 'Registration Caste Wise' report."""
    data = {
        "Total Registered HH": "N/A", "Total Registered Persons": "N/A",
        "SC HH": "N/A", "SC Persons": "N/A",
        "ST HH": "N/A", "ST Persons": "N/A",
        "Male Persons": "N/A", "Female Persons": "N/A"
    }
    
    try:
        driver.get(link_url)
        time.sleep(3)
        body_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Matches the exact "Total:" row format found on the report pages
        match = re.search(r"Total:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", body_text)
        
        if match:
            data["Total Registered HH"] = match.group(1)
            data["Total Registered Persons"] = match.group(2)
            data["SC HH"] = match.group(3)
            data["SC Persons"] = match.group(4)
            data["ST HH"] = match.group(5)
            data["ST Persons"] = match.group(6)
            data["Male Persons"] = match.group(9)
            data["Female Persons"] = match.group(10)
    except Exception as e:
        print(f"      [!] Demographics extract error: {e}")
    return data

# ----------------------------------

def main():
    driver = setup_driver()
    all_data = []
    total_scraped = 0

    try:
        print(f"Loading MGNREGA District Dashboard for {DISTRICT_NAME}...")
        driver.get(DISTRICT_URL)
        time.sleep(5)

        # 1. Get all Blocks in the District
        blocks = get_links_via_js(driver, "block_code")
        if not blocks:
            print("❌ Failed to find Blocks. The site might be down. Exiting.")
            return
            
        print(f"\n✅ Found {len(blocks)} Blocks. Initiating high-speed extraction...\n")

        # 2. Iterate through every Block
        for blk in blocks:
            blk_name = blk['name']
            blk_url = blk['url']
            
            print(f"\n==================================================")
            print(f"🏢 ENTERING BLOCK: {blk_name}")
            print(f"==================================================")
            
            driver.get(blk_url)
            time.sleep(4)
            
            # 3. Get all Panchayats in the Block
            panchayats = get_links_via_js(driver, "panchayat_code")
            # Clean out structural generic links
            panchayats = [p for p in panchayats if p['name'].lower() != 'panchayat']
            
            if not panchayats:
                print(f"  [!] No Panchayats found for Block: {blk_name}")
                continue
                
            print(f"  -> Found {len(panchayats)} Panchayats in {blk_name}. Starting fetch...")
            
            # 4. Iterate through every Panchayat
            for gp in panchayats:
                gp_name = gp['name']
                gp_url = gp['url']
                
                print(f"    [{total_scraped + 1}] Fetching Demographics For: {gp_name}")
                
                # Setup base row structure
                row_data = {
                    'State Name': STATE_NAME,
                    'District Name': DISTRICT_NAME,
                    'Block Name': blk_name,
                    'Panchayat Name': gp_name
                }
                
                # Navigate to the GP Main Page
                driver.get(gp_url)
                time.sleep(3)
                
                # Find target report links on the flat Bootstrap page
                report_links = driver.find_elements(By.TAG_NAME, "a")
                caste_link = None
                
                for link in report_links:
                    try:
                        text = link.text.strip().lower()
                        href = link.get_attribute("href")
                        if href and "javascript" not in href:
                            if "registration caste wise" in text:
                                caste_link = href
                                break # Found it, no need to check the rest
                    except:
                        pass # Ignore stale elements dynamically changing
                
                # Extract Specific Data
                demographics = {"Total Registered HH": "N/A", "Total Registered Persons": "N/A", "SC HH": "N/A", "SC Persons": "N/A", "ST HH": "N/A", "ST Persons": "N/A", "Male Persons": "N/A", "Female Persons": "N/A"}
                
                if caste_link:
                    demographics = extract_demographics(driver, caste_link)
                
                # Combine and save
                row_data.update(demographics)
                all_data.append(row_data)
                total_scraped += 1
            
            # --- PROGRESSIVE SAVE AFTER EVERY BLOCK ---
            if all_data:
                temp_df = pd.DataFrame(all_data)
                # Ensure column order
                front_cols = ['State Name', 'District Name', 'Block Name', 'Panchayat Name']
                other_cols = [col for col in temp_df.columns if col not in front_cols]
                temp_df = temp_df[front_cols + other_cols]
                
                backup_file = f"mgnrega_{DISTRICT_NAME}_Backup_Fast.csv"
                temp_df.to_csv(backup_file, index=False)
                print(f"  [💾 Progress Saved: {len(all_data)} rows total up to block {blk_name}]")
                
    except KeyboardInterrupt:
        print("\n⚠️ Scraping manually interrupted by user! Generating CSV of current progress...")
    except Exception as e:
        print(f"\n❌ An unexpected critical error occurred: {e}")
    finally:
        print("\nClosing browser...")
        driver.quit()

        # --- FINAL FULL SAVE ---
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Organize columns cleanly
            front_cols = ['State Name', 'District Name', 'Block Name', 'Panchayat Name']
            other_cols = [col for col in df.columns if col not in front_cols]
            df = df[front_cols + other_cols]
            
            output_file = f"mgnrega_Demographics_{DISTRICT_NAME}_Fast.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ SUCCESS: Scraped {len(all_data)} Panchayats.")
            print(f"✅ Final Completed Data saved to: {output_file}")
        else:
            print("\n❌ Failed to extract any data. Nothing to save.")

if __name__ == "__main__":
    main()
import time
import re
import pandas as pd

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# ================== CONFIGURATION ==================
# Only State and District are needed now!
TARGET_STATE_VAL = "26"      
TARGET_DISTRICT_VAL = "380"  

# Human-readable names for your final CSV columns
STATE_NAME = "Punjab"
DISTRICT_NAME = "Patiala"
# ===================================================

# ---------------- DRIVER SETUP ----------------
def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ---------------- JAVASCRIPT HANDLERS ----------------
def force_select_via_js(driver, dropdown_partial_id, target_value, sleep_time=2):
    """Instantly sets dropdown values via JS, bypassing Selenium staleness."""
    success = driver.execute_script(f"""
        let dropdown = document.querySelector('select[id*="{dropdown_partial_id}" i]');
        if (dropdown) {{
            dropdown.value = '{target_value}';
            dropdown.dispatchEvent(new Event('change'));
            return true;
        }}
        return false;
    """)
    if success:
        time.sleep(sleep_time) # Wait for background AJAX to cascade
    return success

def get_dropdown_options_via_js(driver, dropdown_partial_id):
    """Scrapes available options from a dropdown instantly."""
    return driver.execute_script(f"""
        let dropdown = document.querySelector('select[id*="{dropdown_partial_id}" i]');
        if (!dropdown) return [];
        
        let valid_options = [];
        for (let i = 0; i < dropdown.options.length; i++) {{
            let val = dropdown.options[i].value;
            let text = dropdown.options[i].text.trim();
            if (val && val !== '0' && val !== '-1' && !text.toLowerCase().includes('select')) {{
                valid_options.push({{'text': text, 'value': val}});
            }}
        }}
        return valid_options;
    """)

def click_submit_via_js(driver):
    """Clicks the Show button via JS."""
    driver.execute_script("""
        let btn = document.querySelector('input[id*="btnShow" i]') || 
                  document.querySelector('input[value="Show" i]');
        if(btn) btn.click();
    """)
    # Give the server time to load the report data below the form
    time.sleep(3.5) 

# ---------------- JJM EXTRACTOR ----------------
def extract_jjm_metrics(driver):
    """Extracts data from the dynamically updated report section."""
    data = {}
    page_text = driver.execute_script("return document.body.innerText;")
    
    # 1. EXTRACT ABSTRACT TEXT DATA (Includes fixes for spaces and newlines)
    patterns = {
        "Total Population": r"Total population\s*:\s*([\d,]+)",
        "SC Population": r"SC\s*\(Scheduled castes\)\s*:\s*([\d,]+)",
        "ST Population": r"ST\s*\(Scheduled tribes\)\s*:\s*([\d,]+)",
        "GEN Population": r"GEN\s*\(General\)\s*:\s*([\d,]+)",
        "No. of Habitations": r"No[,.]\s*of habitations\s*:\s*([\d,]+)",
        "Is PWS Available": r"Is PWS\s*\(Piped water supply\)\s*available\s*\?\s*([A-Za-z]+)",
        "JJM Status": r"JJM status\s*:\s*([^\n]+)",
        "Har Ghar Jal Certificate": r"Har ghar jal certificate\s*:\s*([^\n]+)",
        "Total Households": r"No\.?\s*of housesholds[^:]*:\s*([\d,]+)",
        "Total Tap Connections": r"No\.?\s*of tap connections provided[^:]*:\s*([\d,]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()
        else:
            data[key] = "N/A"
            
    # 2. EXTRACT TABULAR DATA (Schools & Anganwadis)
    try:
        tables = driver.find_elements(By.XPATH, "//table")
        for table in tables:
            rows = table.find_elements(By.XPATH, ".//tr")
            if not rows: continue
            
            headers = [h.text.strip().lower() for h in rows[0].find_elements(By.XPATH, ".//th | .//td")]
            
            if "school name" in headers or "balwadi/ anganwadi" in headers:
                tap_col_idx = next((i for i, h in enumerate(headers) if "drinking water through tap" in h), -1)
                        
                if tap_col_idx != -1:
                    tap_yes_count = sum(1 for row in rows[1:] if len(row.find_elements(By.XPATH, ".//td")) > tap_col_idx 
                                        and row.find_elements(By.XPATH, ".//td")[tap_col_idx].text.strip().lower() == 'yes')
                            
                    prefix = "Schools" if "school name" in headers else "Anganwadis"
                    data[f"Total {prefix}"] = len(rows) - 1
                    data[f"{prefix} with Tap Water"] = tap_yes_count
    except Exception as e:
        pass 
        
    return data

# ---------------- MAIN PIPELINE ----------------
def main():
    driver = setup_driver()
    all_data = []

    try:
        print("Loading JJM Portal...")
        driver.get("https://ejalshakti.gov.in/JJM/JJMReports/profiles/rpt_VillageProfile.aspx")
        time.sleep(5) 

        # 1. Set State and District
        print(f"Setting State ({STATE_NAME}) and District ({DISTRICT_NAME})...")
        force_select_via_js(driver, "state", TARGET_STATE_VAL, sleep_time=3)
        force_select_via_js(driver, "district", TARGET_DISTRICT_VAL, sleep_time=4)

        # 2. Fetch all BLOCKS dynamically
        blocks = get_dropdown_options_via_js(driver, "block")
        print(f"\n✅ Found {len(blocks)} Blocks in {DISTRICT_NAME}. Starting district-wide extraction...\n")

        for blk in blocks:
            blk_val = blk['value']
            blk_text = blk['text']
            print(f"==================================================")
            print(f"🏢 ENTERING BLOCK: {blk_text}")
            print(f"==================================================")
            
            # Select the Block and wait for its Panchayats to load
            force_select_via_js(driver, "block", blk_val, sleep_time=3)
            
            # 3. Fetch all PANCHAYATS dynamically
            panchayats = get_dropdown_options_via_js(driver, "panchayat")
            if not panchayats:
                print(f"  [!] No Panchayats found for Block: {blk_text}")
                continue
                
            for gp in panchayats:
                gp_val = gp['value']
                gp_text = gp['text']
                
                # Select the Panchayat and wait for its Villages to load
                force_select_via_js(driver, "panchayat", gp_val, sleep_time=2.5)
                
                # 4. Fetch all VILLAGES dynamically
                villages = get_dropdown_options_via_js(driver, "village")
                if not villages:
                    continue
                
                for vill in villages:
                    vill_val = vill['value']
                    vill_text = vill['text']
                    
                    print(f"  -> Processing: {vill_text} (GP: {gp_text})")
                    
                    # Select Village and Submit
                    force_select_via_js(driver, "village", vill_val, sleep_time=1)
                    click_submit_via_js(driver)
                    
                    # Extract the metrics
                    metrics_data = extract_jjm_metrics(driver)
                    
                    row_data = {
                        'State Name': STATE_NAME,
                        'District Name': DISTRICT_NAME,
                        'Block Name': blk_text,
                        'Panchayat Name': gp_text,
                        'Village Name': vill_text,
                        **metrics_data
                    }
                    
                    all_data.append(row_data)

    except KeyboardInterrupt:
        print("\nScraping manually interrupted by user!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        driver.quit()

        # --- SAVE TO CSV ---
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Reorder columns to make geographic location read nicely from Left to Right
            front_cols = ['Village Name', 'Panchayat Name', 'Block Name', 'District Name', 'State Name']
            existing_front_cols = [col for col in front_cols if col in df.columns]
            other_cols = [col for col in df.columns if col not in front_cols]
            
            df = df[existing_front_cols + other_cols]
            
            safe_district = DISTRICT_NAME.replace(' ', '_').strip()
            output_file = f"jjm_data_{safe_district}_District.csv"
            
            df.to_csv(output_file, index=False)
            print(f"\n✅ SUCCESS: Fast-Scraped {len(all_data)} villages across {len(blocks)} blocks.")
            print(f"✅ Data saved to: {output_file}")
        else:
            print("\n❌ Failed to extract any data. Nothing to save.")

if __name__ == "__main__":
    main()
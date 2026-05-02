import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException

# ================== CONFIGURATION ==================
TARGET_STATE_VAL = "19"      # Punjab
TARGET_DISTRICT_VAL = "382"  # Bathinda 
TARGET_BLOCK_VAL = "227"     # Target Block
# ===================================================

# ---------------- DRIVER SETUP ----------------
def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ---------------- DROPDOWN HANDLERS ----------------
def get_dropdown_options(driver, wait, element_id, max_retries=3):
    """Fetches valid options, handling AJAX stale elements and network delays."""
    time.sleep(2) 
    
    for attempt in range(max_retries):
        try:
            dropdown = wait.until(EC.presence_of_element_located((By.ID, element_id)))
            select = Select(dropdown)
            options = []
            seen_vals = set() 
            
            for opt in select.options:
                val = opt.get_attribute("value")
                text = opt.text.strip()
                
                if not val or val in ["", "0", "-1"]: continue
                if not text or "select" in text.lower(): continue
                if val in seen_vals: continue
                
                seen_vals.add(val)
                options.append((val, text))
                
            return options
            
        except StaleElementReferenceException:
            print(f"  [Network Delay] Dropdown {element_id} is refreshing. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(1.5)
        except Exception as e:
            print(f"Error getting options for {element_id}: {e}")
            return []
            
    return []

def select_dropdown(driver, wait, element_id, val, max_retries=3):
    """Waits for the specific option to load via AJAX, securely sets it, and handles staleness."""
    for attempt in range(max_retries):
        try:
            wait.until(lambda d: len(d.find_elements(By.XPATH, f"//select[@id='{element_id}']/option[@value='{val}']")) > 0)
            
            dropdown = driver.find_element(By.ID, element_id)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", dropdown)
            
            current_val = driver.execute_script("return arguments[0].value;", dropdown)
            if current_val == val:
                return True 
                
            driver.execute_script(f"arguments[0].value = '{val}';", dropdown)
            driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", dropdown)
            time.sleep(2) # Give extra time for cascading dropdowns to load
            return True
            
        except StaleElementReferenceException:
            print(f"  [AJAX Refresh] Dropdown {element_id} updated mid-action. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(1.5)
        except Exception as e:
            print(f"Timeout Error: Could not find value '{val}' in {element_id}. Site might be slow.")
            return False
            
    return False

def get_selected_text(driver, element_id, max_retries=3):
    """Safely extracts the text of the selected dropdown option."""
    for attempt in range(max_retries):
        try:
            return Select(driver.find_element(By.ID, element_id)).first_selected_option.text
        except StaleElementReferenceException:
            time.sleep(1)
    return "Unknown"

# ---------------- METRIC VALIDATION ----------------
def is_valid_key(k):
    k = str(k).strip()
    k_lower = k.lower()
    
    bad_words = ['assets', 'sn', 's.no', 'total', 'indicator', 'items', 'particulars', 'count', 'action', 'village name']
    if not k or k_lower in bad_words: return False
    if not any(c.isalpha() for c in k): return False
    if len(k) < 3: return False 
    
    return True

# ---------------- EXTRACTOR ----------------
def extract_table_metrics(driver):
    data = {}
    try:
        odf_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'declared as ODF-Plus')]")
        data['ODF Declaration Status'] = odf_elem.text.strip()
    except: pass

    tables = driver.find_elements(By.TAG_NAME, "table")
    for table in tables:
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            cols = row.find_elements(By.XPATH, ".//td | .//th")
            col_texts = [c.text.strip() for c in cols if c.text.strip() != ""]
            
            if not col_texts: continue
            
            if len(col_texts) == 2:
                k, v = col_texts[0], col_texts[1]
                if is_valid_key(k): data[k] = v
                    
            elif len(col_texts) >= 3:
                if col_texts[0].replace('.', '').isdigit():
                    k = col_texts[1].replace('\n', ' ')
                    v = col_texts[2]
                    if is_valid_key(k): data[f"Total - {k}"] = v
                else:
                    k = col_texts[0].replace('\n', ' ')
                    v = col_texts[1]
                    if is_valid_key(k): data[f"Total - {k}"] = v
                        
    return data

# ---------------- MAIN PIPELINE ----------------
def main():
    driver = setup_driver()
    wait = WebDriverWait(driver, 15)
    all_data = []
    
    state_text = ""
    district_text = ""
    block_text = ""

    try:
        driver.get("https://sbm.gov.in/sbmgdashboard/VillageDetailDashboard.aspx")
        
        # Clear Popups
        try:
            driver.find_element(By.XPATH, "//button[contains(@class,'close')]").click()
            driver.execute_script("document.querySelectorAll('.modal-backdrop').forEach(e=>e.remove())")
        except: pass

        print("Switching to Village Progress Report tab...")
        try:
            tab = wait.until(EC.element_to_be_clickable((By.ID, "MstrVPR")))
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(3) 
        except: pass

        # 1. Target Specific State, District, and Block
        print("Initializing Scraper...")
        
        if not select_dropdown(driver, wait, "ddlState", TARGET_STATE_VAL): raise Exception("Failed to load State.")
        if not select_dropdown(driver, wait, "ddlDistrict", TARGET_DISTRICT_VAL): raise Exception("Failed to load District.")
        if not select_dropdown(driver, wait, "ddlBlock", TARGET_BLOCK_VAL): raise Exception("Failed to load Block.")
        
        # Safely extract text
        state_text = get_selected_text(driver, "ddlState")
        district_text = get_selected_text(driver, "ddlDistrict")
        block_text = get_selected_text(driver, "ddlBlock")

        print(f"\n--- Navigating to Target Block: {block_text} ---")

        # 2. Iterate GPs
        gps = get_dropdown_options(driver, wait, "ddlGP")
        print(f"Found {len(gps)} valid Gram Panchayats.")
        
        for gp_val, gp_text in gps:
            
            select_dropdown(driver, wait, "ddlState", TARGET_STATE_VAL)
            select_dropdown(driver, wait, "ddlDistrict", TARGET_DISTRICT_VAL)
            select_dropdown(driver, wait, "ddlBlock", TARGET_BLOCK_VAL)
            select_dropdown(driver, wait, "ddlGP", gp_val)
            
            # 3. Iterate Villages
            villages = get_dropdown_options(driver, wait, "ddlVillage")
            if not villages:
                print(f"Warning: No valid villages found for GP {gp_text}.")
                continue
            
            for vill_val, vill_text in villages:
                print(f"Processing Village: {vill_text} (GP: {gp_text})")
                
                select_dropdown(driver, wait, "ddlState", TARGET_STATE_VAL)
                select_dropdown(driver, wait, "ddlDistrict", TARGET_DISTRICT_VAL)
                select_dropdown(driver, wait, "ddlBlock", TARGET_BLOCK_VAL)
                select_dropdown(driver, wait, "ddlGP", gp_val)
                select_dropdown(driver, wait, "ddlVillage", vill_val)
                
                geo_data = {
                    'State Name': state_text,
                    'District Name': district_text,
                    'Block Name': block_text,
                    'Panchayat Name': gp_text,
                    'Village Name': vill_text
                }
                
                # Submit
                submit_btn = wait.until(EC.presence_of_element_located((By.ID, "btnSubmit")))
                driver.execute_script("arguments[0].click();", submit_btn)
                time.sleep(4) 
                
                # Extract
                metrics_data = extract_table_metrics(driver)
                row_data = {**geo_data, **metrics_data}
                all_data.append(row_data)
                
                print(f"  -> Extracted {len(metrics_data)} metrics.")

                # Reset Page
                print("  -> Returning to VPR Tab for next iteration...")
                try:
                    driver.execute_script("document.querySelectorAll('.modal-backdrop').forEach(e=>e.remove())")
                    tab = wait.until(EC.element_to_be_clickable((By.ID, "MstrVPR")))
                    driver.execute_script("arguments[0].click();", tab)
                    time.sleep(3) 
                except Exception as e:
                    print(f"  -> Warning: Could not click VPR tab to reset: {e}")

    except KeyboardInterrupt:
        print("\nScraping manually interrupted by user!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        driver.quit()

        # --- SAVE TO CSV ---
        if all_data:
            df = pd.DataFrame(all_data)
            
            front_cols = ['Village Name', 'Panchayat Name', 'Block Name', 'District Name', 'State Name', 'ODF Declaration Status']
            existing_front_cols = [col for col in front_cols if col in df.columns]
            other_cols = [col for col in df.columns if col not in front_cols]
            
            df = df[existing_front_cols + other_cols]
            
            safe_block = block_text.replace(' ', '_').replace('(', '').replace(')', '').strip() if block_text else "Unknown_Block"
            output_file = f"sbm_data_{safe_block}.csv"
            
            df.to_csv(output_file, index=False)
            print(f"\n✅ SUCCESS: Scraped {len(all_data)} villages.")
            print(f"✅ Data saved to: {output_file}")
        else:
            print("\n❌ Failed to extract any data. Nothing to save.")

if __name__ == "__main__":
    main()
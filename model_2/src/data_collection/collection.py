import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import warnings
import pandas as pd
import sys
from pathlib import Path
warnings.filterwarnings("ignore")



seasons = {
    2025: "250044",
    2024: "240050",
    2023: "230054",
    2022: "220069",
    2021: "210064",
    2020: "200061",
    2019: "190075",
    2018: "180084",
    2017: "170099",
    2016: "160058",
    2015: "150059",
    2014: "140052", 
    2013: "130034",
    2012: "120002",
    2011: "110002",
    2010: "100002",
    2009: "090002",
    2008: "080002",
    2007: "070002"
}

# Use absolute path from this file's location
SAVE_DIR = Path(__file__).parent.parent.parent / "data" / "raw"  


class SoFIFAScraper:
    def __init__(self, seasons, base_url, save_dir):
        self.seasons = seasons
        self.base_url = base_url
        self.save_dir = save_dir

    def initialize_driver(self):        
        chrome_options = Options()

        chrome_options.add_experimental_option("detach", True)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])

        # Add user agent to avoid bot detection
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Performance optimizations
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        
        chrome_options.page_load_strategy = "eager"

        driver = webdriver.Chrome(options=chrome_options)
        print("Chrome WebDriver initialized")
        
        return driver


    def get_team_stats(self, driver, season_code, year):
        """
        Scrape player stats for a given season.
        
        Args:
            driver: Selenium WebDriver instance
            season_code: SoFIFA season code (e.g., "240050")
            year: Calendar year (e.g., 2024)
            
        Returns:
            DataFrame with all players for the season
        """
        data_list = []  # Use list instead of DataFrame for efficiency
        offset = 0
        
        while True:
            try:
                url = f"{self.base_url}r={season_code}&set=true&offset={offset}"
                driver.get(url)
                time.sleep(3)  # Wait for page load
                
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
            except TimeoutException:
                print(f"No more data at offset {offset}. Scraping complete!")
                break
            except Exception as e:
                print(f"Error at offset {offset}: {e}")
                break
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table = soup.find('table')
            
            if not table:
                print("ERROR: No table found!")
                break
            
            # Get headers only once
            if offset == 0:
                headers = [th.get_text(strip=True) for th in table.select("thead th")]
            
            # Extract rows
            for row in table.select("tbody tr"):
                cols = row.find_all(['th', 'td'])
                if cols:
                    data_list.append([col.get_text(strip=True) for col in cols])

            offset += 60
            
            if offset % 600 == 0:
                print(f"  {offset} players collected so far...")
        
        # Create DataFrame once at the end
        df = pd.DataFrame(data_list, columns=headers)
        
        # Add season information
        df['season'] = year
        df['season_code'] = season_code

        print(f"\n{'='*20}")
        print(f"Total rows collected: {len(df)}")
        print(f"{'='*20}")
        return df

    def close_driver(self, driver):
        """Close the WebDriver."""
        driver.quit()
        print("Chrome WebDriver closed")
    
    def scrape_all_combined(self):
        """Scrape all seasons and write directly to one combined file."""
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.save_dir / "sofifa_players.csv"
        driver = self.initialize_driver()
        
        # first_year = True  If you run the first time
        first_year = False # If you run the next ime
        total_rows = 0
        
        try:
            for year, season_code in self.seasons.items():
                print(f"\n{'='*20}")
                print(f"Starting year {year} (season code: {season_code})")
                print(f"{'='*20}")
                
                # Clear cookies between seasons to prevent caching issues
                driver.delete_all_cookies()
                
                df = self.get_team_stats(driver, season_code, year)
                
                # Write to combined file (append after first year)
                
                df.to_csv(
                    output_path,
                    mode='w' if first_year else 'a',
                    header=first_year,
                    index=False
                )
                
                total_rows += len(df)
                print(f"✓ {year} data ({len(df)} rows) appended to combined file\n")
                
                # Clear memory
                del df
                first_year = False
        
        finally:
            self.close_driver(driver)
            
        print(f"\n{'='*20}")
        print(f"✓ All data combined: {total_rows} total rows")
        print(f"✓ Saved to: {output_path}")
        print(f"\n{'='*20}")

class FBREFAPI:
    def __init__(self, years, list_stats, league, save_dir):
        self.years = years
        self.list_stats = list_stats
        self.league = league
        self.save_dir = save_dir

    def individual_stats(self, year):
        """Collect individual player statistics for each year."""

        fbref = sd.FBref(
            leagues=[self.league], seasons=[year])

        total_others = pd.DataFrame()

        for i in range(len(self.list_stats)):
            stat = self.list_stats[i]

            pl = fbref.read_player_season_stats(stat_type=stat)
            pl.columns = ['_'.join(col).strip() if col[1] else col[0]
                            for col in pl.columns.values]
            pl = pl.reset_index()

            others = pl[pl['pos'] != "GK"]

            if i == 0:
                total_others = others
                print("Data Collected:", stat)
                continue

            others_columns = [
                'player'] + [col for col in others.columns if col not in total_others.columns and col != 'player']
            others = others[others_columns]
            total_others = total_others.merge(
                others, how='left', on=['player'])
            total_others = total_others[total_others['team'].notna()]

            print("Data Collected:", stat)

        total_others = total_others.drop_duplicates(
            subset=['player', 'season']).reset_index(drop=True)

        print("All Individual Data Collected")

        # Use config path instead of hard-coded path
        print("Data Saved for Year:", year)
        
        return total_others
    
    def combine_total(self):
        """Combine all yearly data into one total file."""
        output_path = self.save_dir / "fbref_players.csv"
        
        # Check if file exists and has data to determine append mode
        first_year = not output_path.exists()

        for year in self.years:
            df = FBREFAPI.individual_stats(self, year)
            df.to_csv(
                output_path,
                mode='w' if first_year else 'a',
                header=first_year,
                index=False
            )
            first_year = False
            print(f"✓ {year} data ({len(df)} rows) appended to combined file")
            del df

list_stats = ["standard", "shooting", "passing",
                "passing_types", "goal_shot_creation", "defense", "possession", "playing_time", "misc"]

years = [2020, 2021, 2022, 2023] # list(range(1995, 2025))
league = 'Big 5 European Leagues Combined'

if __name__ == "__main__":
    # BASE_URL = "https://sofifa.com/players?type=all&lg%5B0%5D=13&lg%5B1%5D=31&lg%5B2%5D=19&lg%5B3%5D=53&lg%5B4%5D=16&showCol%5B0%5D=ae&showCol%5B1%5D=oa&showCol%5B2%5D=pt&showCol%5B3%5D=vl&showCol%5B4%5D=wg&showCol%5B5%5D=tt&showCol%5B6%5D=pi&showCol%5B7%5D=wi&showCol%5B8%5D=pf&showCol%5B9%5D=bo&showCol%5B10%5D=hi&showCol%5B11%5D=bp&showCol%5B12%5D=jt&showCol%5B13%5D=le&showCol%5B14%5D=gu&showCol%5B15%5D=rc&showCol%5B16%5D=cp&showCol%5B17%5D=at&showCol%5B18%5D=ps2&showCol%5B19%5D=ps1&showCol%5B20%5D=t2&showCol%5B21%5D=t1&showCol%5B22%5D=phy&showCol%5B23%5D=def&showCol%5B24%5D=pas&showCol%5B25%5D=pac&showCol%5B26%5D=dri&showCol%5B27%5D=hc&showCol%5B28%5D=bt&showCol%5B29%5D=ir&showCol%5B30%5D=aw&showCol%5B31%5D=sk&showCol%5B32%5D=dw&showCol%5B33%5D=bs&showCol%5B34%5D=sho&showCol%5B35%5D=gd&showCol%5B36%5D=sa&showCol%5B37%5D=td&showCol%5B38%5D=cj&showCol%5B39%5D=wk&showCol%5B40%5D=tc&showCol%5B41%5D=ma&showCol%5B42%5D=cm&showCol%5B43%5D=vi&showCol%5B44%5D=pe&showCol%5B45%5D=in&showCol%5B46%5D=ar&showCol%5B47%5D=po&showCol%5B48%5D=te&showCol%5B49%5D=st&showCol%5B50%5D=ju&showCol%5B51%5D=so&showCol%5B52%5D=tp&showCol%5B53%5D=sr&showCol%5B54%5D=ln&showCol%5B55%5D=ag&showCol%5B56%5D=ba&showCol%5B57%5D=re&showCol%5B58%5D=ac&showCol%5B59%5D=to&showCol%5B60%5D=sp&showCol%5B61%5D=lo&showCol%5B62%5D=cu&showCol%5B63%5D=dr&showCol%5B64%5D=fr&showCol%5B65%5D=ts&showCol%5B66%5D=he&showCol%5B67%5D=fi&showCol%5B68%5D=ta&showCol%5B69%5D=sh&showCol%5B70%5D=cr&showCol%5B71%5D=vo&showCol%5B72%5D=bl&"
    # scraper = SoFIFAScraper(seasons, BASE_URL, SAVE_DIR)
    # scraper.scrape_all_combined()
    
    
    SAVE_DIR = Path(__file__).parent.parent.parent / "data" / "raw"  
    fbref_api = FBREFAPI(years, list_stats, league, SAVE_DIR)
    fbref_api.combine_total()


"""

{premier_leaguge: 13, }





"""
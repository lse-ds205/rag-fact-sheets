# Climate Data Web Scraping

Welcome to the Climate Data Web Scraping repository! This project is part of the [DS205 course](https://lse-dsi.github.io/DS205) and is designed to help you learn and practice ethical web scraping techniques using the Climate Action Tracker dataset.

## Getting Started

### 1. Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/lse-ds205/climate-data-web-scraping.git
cd climate-data-web-scraping
```

### 2. Set Up Virtual Environment
It's strongly recommended to use a dedicated virtual environment for this project to avoid conflicts with other Python packages. This is especially important as you might be working on multiple DS205 projects simultaneously (e.g., [ascor-api](https://github.com/lse-ds205/ascor-api)).

```bash
# Create a virtual environment
python -m venv scraping-env

# Activate the virtual environment
# On Mac/Linux:
source scraping-env/bin/activate
# On Windows:
scraping-env\Scripts\activate
```

### 3. Install Dependencies
With your virtual environment activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Spider
Navigate to the project directory and run the spider:
```bash
cd climate_tracker
scrapy crawl climate_action_tracker -O output.json
```

Visit the [Climate Action Tracker](https://climateactiontracker.org/) website to understand the data source.

## Data Usage Notice

Data and extracted textual content from the Climate Action Tracker website are copyrighted © 2009-2025 by Climate Analytics and NewClimate Institute. All rights reserved.

## Ethical Web Scraping

This project follows ethical web scraping practices:
- Respects robots.txt
- Implements appropriate delays between requests
- Properly identifies the spider with user agent information
- Only scrapes publicly available data

## Collaborator Access

Students who are currently enrolled in the DS205 course (or auditing) are eligible to contribute to this repository. To be granted push permission on this repository, please send a message to Jon on Slack with your GitHub username. Once approved, you'll receive an invite to contribute.

## Need Help?
For issues or questions:
- Post in the `#help` channel on Slack
- Check out the [Scrapy Documentation](https://docs.scrapy.org/)
- Contact Jon directly if you face persistent issues
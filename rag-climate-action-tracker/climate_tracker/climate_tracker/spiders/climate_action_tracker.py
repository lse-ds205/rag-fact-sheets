import scrapy
import os
import json
from datetime import datetime
from urllib.parse import urljoin
from climate_tracker.items import ClimateTrackerItem

class ClimateActionTrackerSpider(scrapy.Spider):
    name = "climate_action_tracker_fulltext"
    allowed_domains = ["climateactiontracker.org"]
    start_urls = ["https://climateactiontracker.org/countries/"]
    version = "1.0"  # Version tracking for the scraper
    
    def __init__(self, *args, **kwargs):
        super(ClimateActionTrackerSpider, self).__init__(*args, **kwargs)
        # Dictionary to store country data by country_slug
        self.countries_data = {}
        
        # Dictionary to store unstructured data by country_slug
        self.unstructured_data = {}
        
        # Create the directory structure if it doesn't exist
        self.md_dir = os.path.abspath("climate_tracker/data/full_text/MD")
        self.json_dir = os.path.abspath("climate_tracker/data/full_text/structured")
        self.unstructured_dir = os.path.abspath("climate_tracker/data/full_text/unstructured")
        
        # Create directories
        os.makedirs(self.md_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.unstructured_dir, exist_ok=True)
        
    def parse(self, response):
        country_links = response.css("a::attr(href)").getall()
        country_urls = [
            response.urljoin(link) for link in country_links if "/countries/" in link and link.count("/") == 3
        ]
        for url in set(country_urls):
            yield scrapy.Request(url=url, callback=self.parse_country)
    
    def parse_country(self, response):
        country_slug = response.url.split("/countries/")[-1].strip("/")
        # Try to extract country name from page title or heading
        country_name = response.css("h1::text").get() or response.css("title::text").get().split(" | ")[0].strip()
        
        # Initialize country data in the dictionary
        self.countries_data[country_slug] = {
            "country_name": country_name,
            "sections": {}
        }
        
        # Initialize unstructured data in the dictionary
        self.unstructured_data[country_slug] = {
            "country_slug": country_slug,
            "country_name": country_name,
            "content": [],
            "urls": {},
            "timestamp": datetime.now().isoformat(),
            "version": self.version
        }
        
        # Extract the Summary section directly from this page before proceeding
        summary_content = self.extract_section_text(response)
        self.save_section_md(country_slug, "Summary", summary_content)
        
        # Save to country data structure (structured format)
        self.countries_data[country_slug]["sections"]["Summary"] = {
            "content": summary_content,
            "url": response.url
        }
        
        # Add to unstructured data
        self.add_to_unstructured(country_slug, "Summary", summary_content, response.url)
        
        # Export JSON for structured format
        self.export_json(country_slug)
        
        # Export unstructured format
        self.export_unstructured_json(country_slug)
        
        # Extract links for other sections
        section_links = response.css('.nav.nav-pills a')
        sections = []
        for link in section_links:
            section_title = link.css("::text").get().strip()
            section_url = urljoin(response.url, link.attrib['href'])
            sections.append((section_title, section_url))
        
        section_order = ["Summary", "Targets", "Policies & Actions", "Net Zero Targets", "Assumptions", "Sources"]
        sorted_sections = sorted(sections, key=lambda x: section_order.index(x[0]) if x[0] in section_order else len(section_order))
        
        for title, url in sorted_sections:
            if title != "Summary":  # Skip Summary as we've already processed it
                yield scrapy.Request(
                    url=url, 
                    callback=self.parse_section, 
                    meta={
                        "country": country_slug, 
                        "country_name": country_name,
                        "section": title
                    }
                )
    
    def parse_section(self, response):
        country_slug = response.meta["country"]
        country_name = response.meta["country_name"]
        section_title = response.meta["section"]
        content = self.extract_section_text(response)
        self.save_section_md(country_slug, section_title, content)
        
        # Save to country data structure (structured format)
        if country_slug not in self.countries_data:
            self.countries_data[country_slug] = {
                "country_name": country_name,
                "sections": {}
            }
        
        self.countries_data[country_slug]["sections"][section_title] = {
            "content": content,
            "url": response.url
        }
        
        # Add to unstructured data
        self.add_to_unstructured(country_slug, section_title, content, response.url)
        
        # Export JSON files after each section update
        self.export_json(country_slug)
        self.export_unstructured_json(country_slug)
    
    def add_to_unstructured(self, country_slug, section_title, content, url):
        """Add section data to the unstructured format"""
        if country_slug not in self.unstructured_data:
            # This should not happen, but just in case
            self.unstructured_data[country_slug] = {
                "country_slug": country_slug,
                "country_name": self.countries_data[country_slug]["country_name"],
                "content": [],
                "urls": {},
                "timestamp": datetime.now().isoformat(),
                "version": self.version
            }
        
        # Add section title as a header
        self.unstructured_data[country_slug]["content"].append(f"# {section_title}")
        
        # Add all content paragraphs
        self.unstructured_data[country_slug]["content"].extend(content)
        
        # Add a separator between sections
        self.unstructured_data[country_slug]["content"].append("\n---\n")
        
        # Store the URL for this section
        self.unstructured_data[country_slug]["urls"][section_title] = url
    
    def extract_section_text(self, response):
        elements = response.css('.content-section__content p, .content-section__content li, .content-section__content h3, .content-section__left-side h3')
        content = []
        for element in elements:
            tag = element.root.tag
            # Extract HTML content and explicitly replace <b> and <strong> tags with Markdown bold syntax
            raw_html = element.get()
            selector = scrapy.Selector(text=raw_html)
            # Replace bold tags with Markdown syntax
            for bold_tag in selector.css('b, strong'):
                bold_text = ''.join(bold_tag.css('::text').getall())
                bold_html = bold_tag.get()
                markdown_bold_text = f"**{bold_text}**"
                raw_html = raw_html.replace(bold_html, markdown_bold_text)
            # Convert cleaned HTML to plain text
            text_selector = scrapy.Selector(text=raw_html)
            text = ''.join(text_selector.css('::text').getall()).strip()
            if text:
                if tag == "h3":
                    if element.attrib.get("id"):
                        content.append(f"\n## {text}\n")
                    else:
                        content.append(f"\n### {text}\n")
                elif tag == "li":
                    content.append(f"- {text}")
                else:
                    content.append(f"{text}\n")
        return content
    
    def save_section_md(self, country_slug, section_title, content):
        """Save section to a Markdown file"""
        # Use the MD directory path defined in __init__
        file_path = os.path.join(self.md_dir, f"{country_slug}.md")
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n# {section_title}\n\n")
            for paragraph in content:
                f.write(f"{paragraph}\n")
        self.logger.info(f"Saved {section_title} for {country_slug} to Markdown")
    
    def export_json(self, country_slug):
        """Export data for a single country to a JSON file (structured format)"""
        if country_slug not in self.countries_data:
            self.logger.warning(f"No data found for country {country_slug}")
            return
        
        # Use the structured JSON directory path defined in __init__
        file_path = os.path.join(self.json_dir, f"{country_slug}.json")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.countries_data[country_slug], f, ensure_ascii=False, indent=2)
        self.logger.info(f"Exported structured JSON data for {country_slug}")
    
    def export_unstructured_json(self, country_slug):
        """Export unstructured data for a country to a JSON file"""
        if country_slug not in self.unstructured_data:
            self.logger.warning(f"No unstructured data found for country {country_slug}")
            return
        
        # Use the unstructured directory path defined in __init__
        file_path = os.path.join(self.unstructured_dir, f"{country_slug}.json")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.unstructured_data[country_slug], f, ensure_ascii=False, indent=2)
        self.logger.info(f"Exported unstructured JSON data for {country_slug}")
    
    def closed(self, reason):
        """Export all country JSON files one final time when spider closes"""
        for country_slug in self.countries_data:
            self.export_json(country_slug)
            self.export_unstructured_json(country_slug)
        self.logger.info(f"Spider closed: {reason}")
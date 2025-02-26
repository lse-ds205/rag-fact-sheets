# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from collections import defaultdict
from itemadapter import ItemAdapter
from urllib.parse import urlparse
import scrapy
import csv
import os
from scrapy.pipelines.files import FilesPipeline
from scrapy.utils.project import get_project_settings


from climate_tracker.items import CountryDataFiles, RatingsOverview, RatingsDescription, CountryTargets, PolicyAction, NetZeroTargets, Assumptions
import pandas as pd

class ClimateTrackerPipeline:
    def process_item(self, item, spider):
        return item

class CountryDataPipeline(FilesPipeline):
    def process_item(self, item, spider):
        if isinstance(item, CountryDataFiles):
            country_name = item['country_name']
            if 'xlsx_file' in item:
                xlsx_file = item['xlsx_file']
                self.save_file(xlsx_file, 'xlsx_files', country_name)
            if 'png_file' in item:
                png_file = item['png_file']
                self.save_file(png_file, 'png_files', country_name)
        return item

    def save_file(self, file, folder, country_name):
        file_name = file['file_name']
        file_content = file['file_content']
        file_path = os.path.join(folder, f"{country_name}_{file_name}")
        os.makedirs(folder, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        print(f"Saved file {file_path}")

#This is the Pipeline for the Country Summary Page
class RatingsPipeline:
    def open_spider(self, spider):
        self.overview_file = open('ratings_overview.csv', 'w', newline='', encoding='utf-8')
        self.overview_writer = csv.writer(self.overview_file)
        self.overview_writer.writerow([
            'country_name', 'overall_rating', 'policies_action_domestic', 
            'ndc_target_domestic', 'ndc_target_fair', 'climate_finance', 
            'net_zero_target_year', 'net_zero_target_rating', 'land_forestry_use'
        ])

    def close_spider(self, spider):
        self.overview_file.close()

    def process_item(self, item, spider):
        if isinstance(item, RatingsOverview):
            self.overview_writer.writerow([
                item.get('country_name', 'NA'), item.get('overall_rating', 'NA'), 
                item.get('policies_action_domestic', 'NA'), item.get('ndc_target_domestic', 'NA'), 
                item.get('ndc_target_fair', 'NA'), item.get('climate_finance', 'NA'), 
                item.get('net_zero_target_year', 'NA'), item.get('net_zero_target_rating', 'NA'), 
                item.get('land_forestry_use', 'NA')
            ])
        return item

class RatingsDescriptionPipeline:
    def open_spider(self, spider):
        self.writer = pd.ExcelWriter('ratings_descriptions.xlsx', engine='xlsxwriter')
        self.data = defaultdict(list)

    def close_spider(self, spider):
        try:
            for header, items in self.data.items():
                sheet_name = header[:30]  # Ensure sheet name does not exceed 30 characters
                df = pd.DataFrame(items)
                df.to_excel(self.writer, sheet_name=sheet_name, index=False)
        finally:
            self.writer.close()

    def process_item(self, item, spider):
        if isinstance(item, RatingsDescription):
            header = item.get('header', 'Unknown Rating')
            self.data[header].append({
                'country_name': item.get('country_name', 'NA'),
                'description': item.get('content_text', 'NA')
            })
        return item

    

class CountryTargetsPipeline:
    def open_spider(self, spider):
        self.writer = pd.ExcelWriter('country_targets.xlsx', engine='xlsxwriter')
        self.data = defaultdict(list)

    def close_spider(self, spider):
        try:
            for target, items in self.data.items():
                sheet_name = target[:30]  # Ensure sheet name does not exceed 30 characters
                df = pd.DataFrame(items)
                df.to_excel(self.writer, sheet_name=sheet_name, index=False)
        finally:
            self.writer.close()

    def process_item(self, item, spider):
        if isinstance(item, CountryTargets):
            target = item.get('target', 'Unknown Target')
            self.data[target].append({
                'country_name': item.get('country_name', 'NA'),
                'target_description': item.get('target_description', 'NA')
            })
        return item
    
class PolicyActionPipeline:
    def open_spider(self, spider):
        self.writer = pd.ExcelWriter('policy_actions.xlsx', engine='xlsxwriter')
        self.data = defaultdict(list)

    def close_spider(self, spider):
        try:
            for policy, items in self.data.items():
                sheet_name = policy[:30]  # Ensure sheet name does not exceed 30 characters
                if sheet_name in self.writer.sheets:
                    existing_df = pd.read_excel(self.writer, sheet_name=sheet_name)
                    new_df = pd.DataFrame(items)
                    combined_df = pd.concat([existing_df, new_df])
                    combined_df.to_excel(self.writer, sheet_name=sheet_name, index=False)
                else:
                    df = pd.DataFrame(items)
                    df.to_excel(self.writer, sheet_name=sheet_name, index=False)
        finally:
            self.writer.close()

    def process_item(self, item, spider):
        if isinstance(item, PolicyAction):
            policy = item.get('policy', 'Unknown Policy')
            self.data[policy].append({
                'country_name': item.get('country_name', 'NA'),
                'action_description': item.get('action_description', 'NA')
            })
        return item


class NetZeroTargetsPipeline:
    def open_spider(self, spider):
        self.writer = pd.ExcelWriter('net_zero_targets.xlsx', engine='xlsxwriter')
        self.data = defaultdict(list)

    def close_spider(self, spider):
        try:
            for target, items in self.data.items():
                sheet_name = target[:30]  # Ensure sheet name does not exceed 30 characters
                if sheet_name in self.writer.sheets:
                    suffix = 1
                    while f"{sheet_name}_{suffix}" in self.writer.sheets:
                        suffix += 1
                    sheet_name = f"{sheet_name}_{suffix}"
                df = pd.DataFrame(items)
                df.to_excel(self.writer, sheet_name=sheet_name, index=False)
        finally:
            self.writer.close()

    def process_item(self, item, spider):
        if isinstance(item, NetZeroTargets):
            target = item.get('target', 'Unknown Target')
            self.data[target].append({
                'country_name': item.get('country_name', 'NA'),
                'target_description': item.get('target_description', 'NA')
            })
        return item

class AssumptionsPipeline:
    def open_spider(self, spider):
        self.writer = pd.ExcelWriter('assumptions.xlsx', engine='xlsxwriter')
        self.data = defaultdict(list)

    def close_spider(self, spider):
        try:
            for assumption, items in self.data.items():
                sheet_name = assumption[:30]  # Ensure sheet name does not exceed 30 characters
                df = pd.DataFrame(items)
                if sheet_name in self.writer.sheets:
                    existing_df = pd.read_excel(self.writer, sheet_name=sheet_name)
                    df = pd.concat([existing_df, df], ignore_index=True)
                df.to_excel(self.writer, sheet_name=sheet_name, index=False)
        finally:
            self.writer.close()

    def process_item(self, item, spider):
        if isinstance(item, Assumptions):
            assumption = item.get('assumption', 'Unknown Assumption')
            self.data[assumption].append({
                'country_name': item.get('country_name', 'NA'),
                'assumption_description': item.get('assumption_description', 'NA')
            })
        return item
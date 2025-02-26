# Climate Data Web Scraping Project

## Spider
The Functions within the Spider used are:

### 1. `parse`

- **Purpose**: Collects all country-specific URLs from the main page.
- **Process**:
  - Logs the discovery of country URLs.
  - Extracts country links using XPath.
  - Follows each country URL and calls the `parse_country` function.

### 2. `parse_country`

- **Purpose**: Extracts data from each country-specific page.
- **Process**:
  - Extracts the country name and various ratings.
  - Handles errors and logs any issues encountered.
  - Extracts links to data files (PNG and XLSX) and downloads them.
  - Stores the downloaded files in the `CountryDataFiles` item.
  - Follows links to other pages (targets, policies, net-zero targets, assumptions) and calls respective parsing functions.

### 3. `save_file`

- **Purpose**: Downloads and saves files from the extracted URLs.
- **Process**:
  - Extracts the country name and file content from the response.
  - Stores the file content in the `CountryDataFiles` item.

### 4. `parse_country_target`

- **Purpose**: Extracts target-related data from the country-specific target page.
- **Process**:
  - Extracts target descriptions and NDC tables.
  - Extracts images and downloads them.
  - Stores the data in the `CountryTargets` item.

### 5. `save_image`

- **Purpose**: Downloads and saves images from the extracted URLs.
- **Process**:
  - Extracts the country name and image content from the response.
  - Stores the image content in the `CountryTargets` item.

### 6. `parse_tables`

- **Purpose**: Extracts table data from the containers.
- **Process**:
  - Finds all styled tables within the container.
  - Extracts table titles, subheadings, and values.
  - Stores the table data in a dictionary.

### 7. `parse_policies_action`

- **Purpose**: Extracts policy action-related data from the country-specific policies action page.
- **Process**:
  - Extracts policy descriptions.
  - Extracts images and downloads them.
  - Stores the data in the `PolicyAction` item.

### 8. `parse_net_zero_targets`

- **Purpose**: Extracts net-zero target-related data from the country-specific net-zero targets page.
- **Process**:
  - Extracts net-zero target descriptions.
  - Extracts images and downloads them.
  - Stores the data in the `NetZeroTargets` item.

### 9. `parse_assumptions`

- **Purpose**: Extracts assumption-related data from the country-specific assumptions page.
- **Process**:
  - Extracts assumption descriptions.
  - Stores the data in the `Assumptions` item.

### 10. `extract_with_default`

- **Purpose**: Extracts data using a CSS selector and provides a default value if the data is missing.
- **Process**:
  - Extracts the value using the provided CSS selector.
  - Returns the value or the default value if the data is missing.

## Items

Items are used to define the structure of the data being collected. The main items involved are:

### 1. `RatingsOverview`

- **Fields**:
  - `country_name`
  - `overall_rating`
  - `policies_action_domestic`
  - `ndc_target_domestic`
  - `ndc_target_fair`
  - `climate_finance`
  - `net_zero_target_year`
  - `net_zero_target_rating`
  - `land_forestry_use`

### 2. `RatingsDescription`

- **Fields**:
  - `country_name`
  - `header`
  - `rating`
  - `content_text`

### 3. `CountryTargets`

- **Fields**:
  - `country_name`
  - `target`
  - `target_description`
  - `ndc_data`
  - `images`

### 4. `PolicyAction`

- **Fields**:
  - `country_name`
  - `policy`
  - `action_description`
  - `images`

### 5. `NetZeroTargets`

- **Fields**:
  - `country_name`
  - `target`
  - `target_description`
  - `images`

### 6. `Assumptions`

- **Fields**:
  - `country_name`
  - `assumption`
  - `assumption_description`

### 7. `CountryDataFiles`

- **Fields**:
  - `country_name`
  - `xlsx_file`
  - `png_file`

## Pipelines

Pipelines are used to process the extracted data and save it in the desired format. The main pipelines involved are:

### 1. `RatingsPipeline`

- **Purpose**: Processes the ratings overview data and writes it to a CSV file.
- **Process**:
  - Opens a CSV file for writing the ratings overview data.
  - Writes the data to the CSV file.
  - Closes the CSV file when the spider finishes.

### 2. `RatingsDescriptionPipeline`

- **Purpose**: Processes the ratings description data and writes it to an Excel file.
- **Process**:
  - Opens an Excel file for writing the ratings description data.
  - Writes the data to the Excel file.
  - Closes the Excel file when the spider finishes.

### 3. `CountryTargetsPipeline`

- **Purpose**: Processes the country targets data and writes it to an Excel file.
- **Process**:
  - Opens an Excel file for writing the country targets data.
  - Writes the data to the Excel file.
  - Saves any images collected in the `images` field into the "Country Ratings Overview/png_files" folder.
  - Closes the Excel file when the spider finishes.

### 4. `PolicyActionPipeline`

- **Purpose**: Processes the policy action data and writes it to an Excel file.
- **Process**:
  - Opens an Excel file for writing the policy action data.
  - Writes the data to the Excel file.
  - Saves any images collected in the `images` field into the "Country Ratings Overview/png_files" folder.
  - Closes the Excel file when the spider finishes.

### 5. `NetZeroTargetsPipeline`

- **Purpose**: Processes the net-zero targets data and writes it to an Excel file.
- **Process**:
  - Opens an Excel file for writing the net-zero targets data.
  - Writes the data to the Excel file.
  - Saves any images collected in the `images` field into the "Country Ratings Overview/png_files" folder.
  - Closes the Excel file when the spider finishes.

### 6. `AssumptionsPipeline`

- **Purpose**: Processes the assumptions data and writes it to an Excel file.
- **Process**:
  - Opens an Excel file for writing the assumptions data.
  - Writes the data to the Excel file.
  - Closes the Excel file when the spider finishes.

### 7. `CountryDataPipeline`

- **Purpose**: Processes the downloaded data files and saves them into separate folders.
- **Process**:
  - Saves the downloaded PNG and XLSX files into the "Country Ratings Overview/png_files" and "Country Ratings Overview/xlsx_files" folders, respectively.
  - Renames the files to include the country name for easy identification.



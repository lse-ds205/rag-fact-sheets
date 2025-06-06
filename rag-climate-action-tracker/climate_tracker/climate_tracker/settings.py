import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables explicitly from project root FIRST
# __file__ refers to settings.py
# .parent.parent.parent should navigate to the project root directory
dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"DEBUG (settings.py @ top): Loaded .env from {dotenv_path}. DATABASE_URL: {os.getenv('DATABASE_URL')}") # For debugging
else:
    print(f"DEBUG (settings.py @ top): .env file not found at {dotenv_path}") # For debugging

# Scrapy settings for climate_tracker project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "climate_tracker"

SPIDER_MODULES = ["climate_tracker.spiders"]
NEWSPIDER_MODULE = "climate_tracker.spiders"


# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = "climate_tracker (+http://www.yourdomain.com)"
USER_AGENT = 'LSE DS205 Student Spider (GitHub: @your-username) (+https://lse-dsi.github.io/DS205)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True


# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "climate_tracker.middlewares.ClimateTrackerSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    "climate_tracker.middlewares.ClimateTrackerDownloaderMiddleware": 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
#    "climate_tracker.pipelines.ClimateTrackerPipeline": 300, # Original JSON saving pipeline
     "climate_tracker.pipelines.CountryDataPostgreSQLPipeline": 300, # New pipeline for CountryModel
#    "climate_tracker.pipelines.PostgreSQLPipeline": 300, # This was for NDCDocumentModel
#    "climate_tracker.pipelines.TextExtractionPipeline": 400,
#    "climate_tracker.pipelines.WordEmbeddingPipeline": 500, # Uses Word2Vec
#    "climate_tracker.pipelines.TransformerPipeline": 800, # Uses all-mpnet-base-v2
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 3
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 10
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Project and directory setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
STRUCTURED_JSON_DIR = os.path.join(DATA_DIR, 'full_text', 'structured')

# Logging setup
LOG_FILE = os.path.join(PROJECT_ROOT, 'scrapy.log')
LOG_ENABLED = True
LOG_LEVEL = 'DEBUG'
LOG_FILE_APPEND = True
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'

# Create only the necessary data directories
os.makedirs(STRUCTURED_JSON_DIR, exist_ok=True)

# Standard Scrapy settings
BOT_NAME = "climate_tracker"
SPIDER_MODULES = ["climate_tracker.spiders"]
NEWSPIDER_MODULE = "climate_tracker.spiders"

USER_AGENT = 'LSE DS205 Student Spider (GitHub: @your-username) (+https://lse-dsi.github.io/DS205)'
ROBOTSTXT_OBEY = True
FEED_EXPORT_ENCODING = "utf-8"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Embedding model settings (used downstream)
EMBEDDING_SETTINGS = {
    'model_name': os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
    'batch_size': 32,
    'chunk_size': 1000,
    'overlap': 200,
}

# Optional Word2Vec or FastText if ever used (you can drop this if unused)
WORD_EMBEDDING_SETTINGS = {
    'vector_size': 100,
    'window': 5,
    'min_count': 5,
}

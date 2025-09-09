"""
Advanced Web Scraper Tool for LLM Agents
=========================================

A comprehensive web scraping tool with support for multiple parsers,
content extraction methods, and article processing. Designed for seamless
integration with LLM agent frameworks.

Requirements:
    - requests (install with: pip install requests)
    - beautifulsoup4 (install with: pip install beautifulsoup4)
    - lxml (install with: pip install lxml)
    - html5lib (install with: pip install html5lib)
    - chardet (install with: pip install chardet)
    - selenium (optional: pip install selenium)
    - webdriver-manager (optional: pip install webdriver-manager)
    - trafilatura (optional: pip install trafilatura)

Author: LLMFlow Framework
Version: 2.0.0
"""

import os
import re
import time
import json
import logging
import subprocess
import sys
from typing import Dict, Optional, Union, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import Counter

# Check and install required packages
packages = {
    'requests': 'requests',
    'bs4': 'beautifulsoup4',
    'lxml': 'lxml',
    'html5lib': 'html5lib',
    'chardet': 'chardet'
}

optional_packages = {
    'selenium': 'selenium',
    'webdriver_manager': 'webdriver-manager',
    'trafilatura': 'trafilatura'
}

for import_name, install_name in packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"{install_name} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Import required packages
import requests
from bs4 import BeautifulSoup, Comment
import chardet

# Try to import optional packages
HAS_SELENIUM = False
HAS_TRAFILATURA = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    pass

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    pass

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator

# Setup logging
scraper_logger = logging.getLogger(__name__)
if not scraper_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    scraper_logger.addHandler(handler)
    scraper_logger.setLevel(logging.INFO)


# Data structures
class ParserType(Enum):
    """Available HTML parsers."""
    HTML_PARSER = "html.parser"
    LXML = "lxml"
    HTML5LIB = "html5lib"


class ContentType(Enum):
    """Types of content to extract."""
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    LINKS = "links"
    IMAGES = "images"
    METADATA = "metadata"
    TABLES = "tables"
    ARTICLE = "article"
    STRUCTURED = "structured"
    ALL = "all"


class ExtractionMode(Enum):
    """Mode of extraction."""
    FULL = "full"
    MAIN = "main"
    ARTICLE = "article"


@dataclass
class ScrapeOptions:
    """Configuration options for web scraping."""
    headless: bool = True
    timeout: int = 30
    wait_for_js: float = 2.0
    wait_selector: Optional[str] = None
    wait_timeout: int = 20
    max_text_length: int = 50000
    extract_links: bool = True
    extract_images: bool = True
    extract_metadata: bool = True
    extract_tables: bool = False
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.TEXT])
    extraction_mode: ExtractionMode = ExtractionMode.MAIN
    parser: ParserType = ParserType.LXML
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    rate_limit: float = 1.0
    use_selenium: bool = False
    extract_article_content: bool = False
    verbose: bool = False
    retry_attempts: int = 3
    window_size: tuple = (1920, 1080)
    remove_scripts: bool = True
    remove_styles: bool = True


@dataclass
class WebPageMetadata:
    """Metadata extracted from a web page."""
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    author: Optional[str] = None
    language: Optional[str] = None
    published_date: Optional[str] = None
    modified_date: Optional[str] = None
    canonical_url: Optional[str] = None
    og_data: Dict[str, str] = field(default_factory=dict)
    twitter_data: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedContent:
    """Container for all extracted content from a web page."""
    url: str
    title: Optional[str] = None
    text: str = ""
    main_text: Optional[str] = None
    full_text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    metadata: Optional[WebPageMetadata] = None
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    headings: List[Dict[str, str]] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    article_data: Optional[Dict[str, Any]] = None
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    parser_used: Optional[str] = None
    encoding: Optional[str] = None
    word_count: int = 0
    language: Optional[str] = None


@dataclass
class ScrapeResponse:
    """Response from a scrape operation."""
    success: bool
    url: str
    content: Optional[ExtractedContent] = None
    data: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    load_time: float = 0.0
    error: Optional[str] = None
    message: Optional[str] = None
    cached: bool = False
    timestamp: Optional[datetime] = None


class WebScraper:
    """
    Advanced web scraping class with comprehensive extraction capabilities.
    """
    
    def __init__(
        self,
        options: Optional[ScrapeOptions] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the web scraper."""
        self.options = options or ScrapeOptions()
        self.logger = logger or scraper_logger
        self.session = self._setup_session()
        self.selenium_driver = None
        self.last_request_time: Dict[str, float] = {}
        
    def _setup_session(self) -> requests.Session:
        """Setup requests session with headers."""
        session = requests.Session()
        user_agent = self.options.user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36 LLMFlow-Scraper/2.0'
        )
        
        session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        if self.options.proxy:
            session.proxies = {
                'http': self.options.proxy,
                'https': self.options.proxy
            }
            
        return session
    
    def _setup_selenium_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with options."""
        if not HAS_SELENIUM:
            raise ImportError("Selenium not installed. Install with: pip install selenium webdriver-manager")
            
        chrome_options = Options()
        
        if self.options.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument(f"--window-size={self.options.window_size[0]},{self.options.window_size[1]}")
        
        if self.options.user_agent:
            chrome_options.add_argument(f"user-agent={self.options.user_agent}")
        
        if self.options.proxy:
            chrome_options.add_argument(f"--proxy-server={self.options.proxy}")
        
        # Additional options for stability
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        import platform
        
        # Special handling for Apple Silicon Macs
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except:
                driver_manager = ChromeDriverManager()
                driver_path = driver_manager.install()
                service = Service(driver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            try:
                driver_manager = ChromeDriverManager()
                driver_path = driver_manager.install()
                service = Service(driver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except:
                driver = webdriver.Chrome(options=chrome_options)
        
        driver.set_page_load_timeout(self.options.timeout)
        
        return driver
    
    def _respect_rate_limit(self, domain: str) -> None:
        """Enforce rate limiting for respectful scraping."""
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < self.options.rate_limit:
                sleep_time = self.options.rate_limit - elapsed
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    def _detect_encoding(self, response: requests.Response) -> str:
        """Detect the encoding of the response."""
        if response.encoding and response.encoding.lower() != 'iso-8859-1':
            return response.encoding
            
        detected = chardet.detect(response.content)
        if detected['encoding'] and detected['confidence'] > 0.7:
            return detected['encoding']
            
        return 'utf-8'
    
    def fetch_page(
        self,
        url: str,
        use_selenium: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """Fetch web page content."""
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                self.logger.error(f"Invalid URL: {url}")
                return None, None
            
            domain = self._get_domain(url)
            self._respect_rate_limit(domain)
            
            if use_selenium and HAS_SELENIUM:
                return self._fetch_with_selenium(url)
            else:
                return self._fetch_with_requests(url)
                
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None, None
    
    def _fetch_with_requests(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch page using requests library."""
        try:
            response = self.session.get(url, timeout=self.options.timeout)
            response.raise_for_status()
            
            encoding = self._detect_encoding(response)
            
            try:
                content = response.content.decode(encoding)
            except UnicodeDecodeError:
                content = response.content.decode('utf-8', errors='replace')
                encoding = 'utf-8'
                
            return content, encoding
            
        except requests.RequestException as e:
            self.logger.error(f"Request error for {url}: {str(e)}")
            return None, None
    
    def _fetch_with_selenium(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch page using Selenium for JavaScript content."""
        if not HAS_SELENIUM:
            self.logger.warning("Selenium not available, falling back to requests")
            return self._fetch_with_requests(url)
        
        try:
            if not self.selenium_driver:
                self.selenium_driver = self._setup_selenium_driver()
            
            self.selenium_driver.get(url)
            
            if self.options.wait_for_js > 0:
                time.sleep(self.options.wait_for_js)
            
            # If a specific CSS selector is provided, wait for it; otherwise wait for <body>
            try:
                if self.options.wait_selector:
                    WebDriverWait(self.selenium_driver, self.options.wait_timeout).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, self.options.wait_selector))
                    )
                else:
                    WebDriverWait(self.selenium_driver, self.options.wait_timeout).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
            except TimeoutException:
                self.logger.warning(
                    f"Timeout waiting for selector '{self.options.wait_selector or 'body'}' on {url}"
                )
            
            # Small scroll to trigger lazy content
            try:
                self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
                time.sleep(0.5)
                self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
            except Exception:
                pass
            
            content = self.selenium_driver.page_source
            
            return content, 'utf-8'
            
        except Exception as e:
            self.logger.error(f"Selenium error for {url}: {str(e)}")
            return None, None
    
    def extract_metadata(self, soup: BeautifulSoup) -> WebPageMetadata:
        """Extract metadata from HTML."""
        metadata = WebPageMetadata()
        
        title_tag = soup.find('title')
        if title_tag:
            metadata.title = title_tag.get_text(strip=True)
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if not content:
                continue
            
            if name == 'description':
                metadata.description = content
            elif name == 'keywords':
                metadata.keywords = [k.strip() for k in content.split(',')]
            elif name == 'author':
                metadata.author = content
            elif name == 'language':
                metadata.language = content
            elif name in ['published_time', 'article:published_time']:
                metadata.published_date = content
            elif name in ['modified_time', 'article:modified_time']:
                metadata.modified_date = content
            
            if property_attr.startswith('og:'):
                metadata.og_data[property_attr] = content
            
            if name.startswith('twitter:'):
                metadata.twitter_data[name] = content
        
        canonical = soup.find('link', {'rel': 'canonical'})
        if canonical:
            metadata.canonical_url = canonical.get('href')
        
        html_tag = soup.find('html')
        if html_tag and not metadata.language:
            metadata.language = html_tag.get('lang')
        
        return metadata
    
    def extract_text(self, soup: BeautifulSoup, clean: bool = True) -> str:
        """Extract text content from HTML."""
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        text = soup.get_text()
        
        if clean:
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        if self.options.remove_scripts:
            for script in soup(['script', 'noscript']):
                script.decompose()
        
        if self.options.remove_styles:
            for style in soup(['style', 'link']):
                style.decompose()
        
        main_selectors = [
            'main', 'article', '[role="main"]', '.main-content',
            '#main-content', '.content', '#content', '.post-content',
            '.entry-content', '.article-content', '.post', '.article-body'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            for element in main_content.select('nav, header, footer, aside, .sidebar, .navigation, .menu, .advertisement'):
                element.decompose()
            
            return main_content.get_text(separator='\n', strip=True)
        
        return soup.get_text(separator='\n', strip=True)
    
    def extract_article(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract article content using trafilatura."""
        if not HAS_TRAFILATURA:
            self.logger.warning("Trafilatura not available for article extraction")
            return None
        
        try:
            article_text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                deduplicate=True
            )
            
            if not article_text:
                return None
            
            metadata = trafilatura.extract_metadata(html, url)
            
            article_data = {
                'content': article_text,
                'title': metadata.title if metadata else None,
                'author': metadata.author if metadata else None,
                'date': metadata.date if metadata else None,
                'description': metadata.description if metadata else None
            }
            
            return article_data
            
        except Exception as e:
            self.logger.error(f"Article extraction error: {str(e)}")
            return None
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        base_domain = self._get_domain(base_url)
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue
            
            absolute_url = urljoin(base_url, href)
            
            if not absolute_url.startswith(('http://', 'https://')):
                continue
            
            link_domain = self._get_domain(absolute_url)
            is_internal = link_domain == base_domain
            
            link_data = {
                'url': absolute_url,
                'text': link.get_text(strip=True),
                'title': link.get('title', ''),
                'is_internal': is_internal,
                'domain': link_domain
            }
            
            links.append(link_data)
        
        return links
    
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all images from the page."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if not src:
                continue
            
            absolute_url = urljoin(base_url, src)
            
            image_data = {
                'url': absolute_url,
                'src': absolute_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            }
            
            images.append(image_data)
        
        return images
    
    def extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables as nested lists."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            
            for row in table.find_all('tr'):
                row_data = []
                cells = row.find_all(['th', 'td'])
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    row_data.append(cell_text)
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all headings from the page."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text(strip=True),
                    'tag': f'h{level}'
                })
        return headings
    
    def convert_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert HTML to Markdown format."""
        markdown_lines = []
        
        def process_element(element):
            if element.name == 'h1':
                markdown_lines.append(f"# {element.get_text(strip=True)}")
            elif element.name == 'h2':
                markdown_lines.append(f"## {element.get_text(strip=True)}")
            elif element.name == 'h3':
                markdown_lines.append(f"### {element.get_text(strip=True)}")
            elif element.name == 'h4':
                markdown_lines.append(f"#### {element.get_text(strip=True)}")
            elif element.name == 'h5':
                markdown_lines.append(f"##### {element.get_text(strip=True)}")
            elif element.name == 'h6':
                markdown_lines.append(f"###### {element.get_text(strip=True)}")
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(text)
                    markdown_lines.append("")
            elif element.name == 'ul':
                for li in element.find_all('li', recursive=False):
                    markdown_lines.append(f"- {li.get_text(strip=True)}")
            elif element.name == 'ol':
                for i, li in enumerate(element.find_all('li', recursive=False), 1):
                    markdown_lines.append(f"{i}. {li.get_text(strip=True)}")
            elif element.name == 'blockquote':
                for line in element.get_text(strip=True).split('\n'):
                    markdown_lines.append(f"> {line}")
            elif element.name == 'code':
                markdown_lines.append(f"`{element.get_text(strip=True)}`")
            elif element.name == 'pre':
                markdown_lines.append("```")
                markdown_lines.append(element.get_text(strip=True))
                markdown_lines.append("```")
            elif element.name == 'hr':
                markdown_lines.append("---")
        
        main_content = soup.find('body') or soup
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'blockquote', 'pre', 'hr']):
            process_element(element)
        
        return '\n'.join(markdown_lines)
    
    def extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, Microdata, etc.)."""
        structured_data = {}
        
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for i, script in enumerate(json_ld_scripts):
            try:
                data = json.loads(script.string)
                structured_data[f'json_ld_{i}'] = data
            except:
                pass
        
        return structured_data
    
    def scrape(
        self,
        url: str,
        options: Optional[ScrapeOptions] = None
    ) -> ScrapeResponse:
        """Main scraping method."""
        if options:
            original_options = self.options
            self.options = options
        
        try:
            self.logger.info(f"Scraping {url}")
            
            # Fetch page
            html, encoding = self.fetch_page(url, self.options.use_selenium)
            if not html:
                return ScrapeResponse(
                    success=False,
                    url=url,
                    error='Failed to fetch page content'
                )
            
            # Parse HTML
            try:
                soup = BeautifulSoup(html, self.options.parser.value)
            except Exception as e:
                self.logger.warning(f"Parser {self.options.parser.value} failed, trying html.parser")
                soup = BeautifulSoup(html, 'html.parser')
                parser = ParserType.HTML_PARSER
            
            # Create content container
            content = ExtractedContent(
                url=url,
                parser_used=self.options.parser.value,
                encoding=encoding
            )
            
            # Handle content types
            content_types = self.options.content_types
            if ContentType.ALL in content_types:
                content_types = [
                    ContentType.TEXT, ContentType.METADATA,
                    ContentType.LINKS, ContentType.IMAGES, ContentType.TABLES
                ]
            
            # Extract content based on types
            if ContentType.METADATA in content_types:
                content.metadata = self.extract_metadata(soup)
                content.title = content.metadata.title
            
            if ContentType.TEXT in content_types:
                if self.options.extraction_mode == ExtractionMode.MAIN:
                    content.main_text = self.extract_main_content(soup)
                    content.text = content.main_text
                else:
                    content.full_text = self.extract_text(soup)
                    content.text = content.full_text
                content.word_count = len(content.text.split())
            
            if ContentType.HTML in content_types:
                content.html = str(soup.prettify())
            
            if ContentType.MARKDOWN in content_types:
                content.markdown = self.convert_to_markdown(soup)
            
            if ContentType.LINKS in content_types:
                content.links = self.extract_links(soup, url)
            
            if ContentType.IMAGES in content_types:
                content.images = self.extract_images(soup, url)
            
            if ContentType.TABLES in content_types:
                content.tables = self.extract_tables(soup)
            
            if ContentType.STRUCTURED in content_types:
                content.structured_data = self.extract_structured_data(soup)
            
            # Always extract headings
            content.headings = self.extract_headings(soup)
            
            # Article extraction if requested
            if self.options.extract_article_content or ContentType.ARTICLE in content_types:
                content.article_data = self.extract_article(html, url)
            
            # Format response data
            response_data = {
                'success': True,
                'url': content.url,
                'title': content.title,
                'text': content.text[:self.options.max_text_length] if content.text else "",
                'word_count': content.word_count,
                'encoding': content.encoding,
                'parser': content.parser_used,
                'extraction_time': content.extraction_timestamp.isoformat()
            }
            
            if content.metadata:
                response_data['metadata'] = {
                    'description': content.metadata.description,
                    'keywords': content.metadata.keywords,
                    'author': content.metadata.author,
                    'language': content.metadata.language,
                    'published_date': content.metadata.published_date,
                    'og_data': content.metadata.og_data,
                    'twitter_data': content.metadata.twitter_data
                }
            
            if content.links:
                response_data['links'] = {
                    'total': len(content.links),
                    'internal': sum(1 for l in content.links if l['is_internal']),
                    'external': sum(1 for l in content.links if not l['is_internal']),
                    'links': content.links[:50]
                }
            
            if content.images:
                response_data['images'] = {
                    'total': len(content.images),
                    'images': content.images[:50]
                }
            
            if content.tables:
                response_data['tables'] = {
                    'total': len(content.tables),
                    'tables': content.tables[:10]
                }
            
            if content.article_data:
                response_data['article'] = content.article_data
            
            if content.markdown:
                response_data['markdown'] = content.markdown[:5000]
            
            if content.html:
                response_data['html'] = content.html[:5000]
            
            if content.structured_data:
                response_data['structured_data'] = content.structured_data
            
            response_data['headings'] = content.headings[:20]
            
            return ScrapeResponse(
                success=True,
                url=url,
                content=content,
                data=response_data,
                message=f'Successfully scraped {url}'
            )
            
        except Exception as e:
            self.logger.error(f"Scraping error for {url}: {str(e)}")
            return ScrapeResponse(
                success=False,
                url=url,
                error=f'Scraping failed: {str(e)}'
            )
        finally:
            if options:
                self.options = original_options
    
    def scrape_with_retry(
        self,
        url: str,
        max_retries: Optional[int] = None,
        options: Optional[ScrapeOptions] = None
    ) -> ScrapeResponse:
        """Scrape with automatic retry on failure."""
        max_retries = max_retries or self.options.retry_attempts
        last_error = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{max_retries}")
                self.close()
            
            response = self.scrape(url, options)
            
            if response.success:
                return response
            
            last_error = response.error
        
        return ScrapeResponse(
            success=False,
            url=url,
            error=f"Failed after {max_retries} retries. Last error: {last_error}",
            timestamp=datetime.now()
        )
    
    def scrape_multiple(
        self,
        urls: List[str],
        options: Optional[ScrapeOptions] = None
    ) -> List[ScrapeResponse]:
        """Scrape multiple URLs."""
        results = []
        for url in urls:
            self.logger.info(f"Scraping {len(results) + 1}/{len(urls)}: {url}")
            result = self.scrape(url, options)
            results.append(result)
        return results
    
    def close(self):
        """Close resources."""
        try:
            if self.selenium_driver:
                self.selenium_driver.quit()
                self.selenium_driver = None
        except Exception as e:
            self.logger.warning(f"Error closing WebDriver: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class WebScraperTool:
    """
    LLM Agent Tool wrapper for Web Scraper.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, default_options: Optional[ScrapeOptions] = None):
        """Initialize the tool."""
        self.scraper = WebScraper(default_options)
        self.name = "advanced_web_scraper"
        self.description = (
            "Advanced scraper for extracting content from web pages. "
            "Supports text, metadata, links, images, tables, articles, and structured data. "
            "Input should be a URL with optional parameters. "
            "Returns comprehensive extracted content from the page."
        )
    
    def run(self, url: Union[str, Dict[str, Any]], **kwargs) -> str:
        """
        Run the scraper tool.
        
        Args:
            url: URL to scrape or dict with url and options
            **kwargs: Additional options
            
        Returns:
            JSON string with scraped content
        """
        target_url = url
        if isinstance(url, dict):
            target_url = url.get("url", "")
            kwargs.update(url)
        
        # Parse kwargs into options
        options = ScrapeOptions()
        
        # Content types
        content_types = []
        if kwargs.get('extract_text', True):
            content_types.append(ContentType.TEXT)
        if kwargs.get('extract_metadata', True):
            content_types.append(ContentType.METADATA)
        if kwargs.get('extract_links', False):
            content_types.append(ContentType.LINKS)
        if kwargs.get('extract_images', False):
            content_types.append(ContentType.IMAGES)
        if kwargs.get('extract_tables', False):
            content_types.append(ContentType.TABLES)
        if kwargs.get('extract_article', False):
            content_types.append(ContentType.ARTICLE)
        if kwargs.get('extract_structured', False):
            content_types.append(ContentType.STRUCTURED)
        
        options.content_types = content_types
        
        # Other options
        if kwargs.get('use_selenium') is not None:
            options.use_selenium = kwargs['use_selenium']
        if kwargs.get('headless') is not None:
            options.headless = kwargs['headless']
        if kwargs.get('wait_selector') is not None:
            options.wait_selector = kwargs['wait_selector']
        if kwargs.get('wait_timeout') is not None:
            options.wait_timeout = int(kwargs['wait_timeout'])
        if kwargs.get('parser'):
            parser_map = {
                "lxml": ParserType.LXML,
                "html.parser": ParserType.HTML_PARSER,
                "html5lib": ParserType.HTML5LIB
            }
            options.parser = parser_map.get(kwargs['parser'].lower(), ParserType.LXML)
        
        # Auto-heuristics for JS-heavy domains when not explicitly set
        try:
            domain = urlparse(str(target_url)).netloc.lower()
        except Exception:
            domain = ""
        # Only apply when caller didn't specify use_selenium
        if 'use_selenium' not in kwargs:
            if any(d in domain for d in [
                'coinmarketcap.com',
                'tradingview.com',
                'medium.com',
                'linkedin.com',
                'twitter.com',
                'x.com'
            ]):
                options.use_selenium = True
                if not options.wait_selector:
                    options.wait_selector = 'main'
                if not options.wait_timeout:
                    options.wait_timeout = 25
        
        # Scrape
        response = self.scraper.scrape_with_retry(target_url, options=options)
        
        # Return data directly if success
        if response.success:
            return json.dumps(response.data, indent=2)
        else:
            return json.dumps({
                'success': False,
                'url': target_url,
                'error': response.error
            })
    
    async def arun(self, url: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Async version of run."""
        return self.run(url, **kwargs)
    
    def cleanup(self):
        """Clean up resources."""
        self.scraper.close()


# Registered tool functions
@register_tool(
    tags=["web", "scraper", "content", "extraction", "parser", "advanced"]
)
def scrape_webpage(
    url: str,
    extract_text: bool = True,
    extract_metadata: bool = True,
    extract_links: bool = False,
    extract_images: bool = False,
    extract_tables: bool = False,
    use_selenium: bool = False,
    parser: str = "lxml",
    wait_selector: Optional[str] = None,
    wait_timeout: int = 20,
) -> Dict[str, Any]:
    """Advanced webpage scraper with comprehensive extraction options.
    
    Args:
        url: The URL of the web page to scrape
        extract_text: Whether to extract text content
        extract_metadata: Whether to extract metadata
        extract_links: Whether to extract links
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        use_selenium: Use Selenium for JavaScript-rendered content
        parser: HTML parser to use ("lxml", "html.parser", "html5lib")
        
    Returns:
        Dictionary with scraped content
    """
    content_types = []
    if extract_text:
        content_types.append(ContentType.TEXT)
    if extract_metadata:
        content_types.append(ContentType.METADATA)
    if extract_links:
        content_types.append(ContentType.LINKS)
    if extract_images:
        content_types.append(ContentType.IMAGES)
    if extract_tables:
        content_types.append(ContentType.TABLES)
    
    parser_map = {
        "lxml": ParserType.LXML,
        "html.parser": ParserType.HTML_PARSER,
        "html5lib": ParserType.HTML5LIB
    }
    
    options = ScrapeOptions(
        content_types=content_types,
        parser=parser_map.get(parser.lower(), ParserType.LXML),
        use_selenium=use_selenium,
        wait_selector=wait_selector,
        wait_timeout=wait_timeout,
    )
    
    with WebScraper(options) as scraper:
        response = scraper.scrape(url)
        
        if response.success:
            return response.data
        else:
            return {
                'success': False,
                'url': url,
                'error': response.error
            }


@register_tool(
    tags=["web", "article", "content", "extraction", "readability"]
)
def extract_article(
    url: str,
    fallback_to_full_text: bool = True
) -> Dict[str, Any]:
    """Extract article content from a web page using advanced algorithms.
    
    Args:
        url: URL of the article to extract
        fallback_to_full_text: Use full text extraction if article extraction fails
        
    Returns:
        Dictionary with article content
    """
    options = ScrapeOptions(
        content_types=[ContentType.ARTICLE, ContentType.TEXT, ContentType.METADATA],
        extract_article_content=True
    )
    
    with WebScraper(options) as scraper:
        response = scraper.scrape(url)
        
        if response.success:
            data = response.data
            if data.get('article') and data['article'].get('content'):
                return {
                    'success': True,
                    'data': {
                        'content': data['article']['content'],
                        'title': data['article'].get('title'),
                        'author': data['article'].get('author'),
                        'date': data['article'].get('date'),
                        'description': data['article'].get('description'),
                        'word_count': len(data['article']['content'].split()),
                        'url': url
                    },
                    'message': f'Successfully extracted article from {url}'
                }
            elif fallback_to_full_text:
                return {
                    'success': True,
                    'data': {
                        'content': data.get('text', ''),
                        'title': data.get('title'),
                        'metadata': data.get('metadata'),
                        'word_count': data.get('word_count'),
                        'url': url
                    },
                    'message': f'Extracted text content from {url}'
                }
        
        return {
            'success': False,
            'error': response.error or 'Article extraction failed'
        }


@register_tool(
    tags=["web", "batch", "multiple", "scraper"]
)
def scrape_multiple_pages(
    urls: List[str],
    extract_text: bool = True,
    extract_metadata: bool = True,
    rate_limit: float = 1.0,
    max_workers: int = 1
) -> Dict[str, Any]:
    """Scrape multiple web pages with rate limiting.
    
    Args:
        urls: List of URLs to scrape
        extract_text: Whether to extract text content
        extract_metadata: Whether to extract metadata
        rate_limit: Minimum seconds between requests to same domain
        max_workers: Number of concurrent workers
        
    Returns:
        Dictionary with results for all URLs
    """
    if not urls:
        return {
            'success': False,
            'error': 'URLs list cannot be empty'
        }
    
    if len(urls) > 50:
        return {
            'success': False,
            'error': 'Maximum 50 URLs allowed per request'
        }
    
    content_types = []
    if extract_text:
        content_types.append(ContentType.TEXT)
    if extract_metadata:
        content_types.append(ContentType.METADATA)
    
    options = ScrapeOptions(
        content_types=content_types,
        rate_limit=rate_limit
    )
    
    with WebScraper(options) as scraper:
        responses = scraper.scrape_multiple(urls)
        
        successful = []
        failed = []
        
        for response in responses:
            if response.success:
                successful.append({
                    'url': response.url,
                    'data': response.data
                })
            else:
                failed.append({
                    'url': response.url,
                    'error': response.error
                })
        
        return {
            'success': len(successful) > 0,
            'data': {
                'successful': successful,
                'failed': failed,
                'total_requested': len(urls),
                'total_successful': len(successful)
            },
            'message': f'Scraped {len(successful)}/{len(urls)} pages successfully'
        }


# LangChain integration
try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    class ScrapeInput(BaseModel):
        """Input schema for scraper tool."""
        url: str = Field(description="URL to scrape")
        extract_links: bool = Field(default=False, description="Extract links from the page")
        extract_images: bool = Field(default=False, description="Extract images from the page")
        extract_tables: bool = Field(default=False, description="Extract tables from the page")
        use_selenium: bool = Field(default=False, description="Use Selenium for JavaScript content")
        wait_selector: Optional[str] = Field(default=None, description="CSS selector to wait for before parsing (helps with JS-heavy sites)")
        wait_timeout: int = Field(default=20, description="Seconds to wait for selector/body to appear")
    
    def get_langchain_scraper_tool() -> StructuredTool:
        """Returns a configured web scraper tool for LangChain."""
        tool_instance = WebScraperTool()
        
        def scrape_func(
            url: str,
            extract_links: bool = False,
            extract_images: bool = False,
            extract_tables: bool = False,
            use_selenium: bool = False,
            wait_selector: Optional[str] = None,
            wait_timeout: int = 20,
        ) -> str:
            """Scrape webpage and return formatted content."""
            try:
                result = tool_instance.run(
                    url,
                    extract_links=extract_links,
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    use_selenium=use_selenium,
                    wait_selector=wait_selector,
                    wait_timeout=wait_timeout,
                )
                
                data = json.loads(result)
                
                if data.get('success'):
                    formatted = f"📄 Scraped: {data['url']}\n\n"
                    formatted += f"**Title:** {data.get('title', 'No title')}\n\n"
                    
                    text = data.get('text', '')
                    if text:
                        display_text = text[:1000] + "..." if len(text) > 1000 else text
                        formatted += f"**Content:**\n{display_text}\n\n"
                    
                    if data.get('metadata'):
                        formatted += f"**Metadata:** Found\n"
                    
                    if extract_links and data.get('links'):
                        formatted += f"\n**Links found ({data['links']['total']} total)**\n"
                    
                    if extract_images and data.get('images'):
                        formatted += f"\n**Images found ({data['images']['total']} total)**\n"
                    
                    if extract_tables and data.get('tables'):
                        formatted += f"\n**Tables found ({data['tables']['total']} total)**\n"
                    
                    return formatted
                else:
                    return f"❌ Failed to scrape {data.get('url')}. Error: {data.get('error', 'Unknown error')}"
                    
            except Exception as e:
                return f"❌ Scraping error: {str(e)}"
        
        return StructuredTool.from_function(
            func=scrape_func,
            name="advanced_web_scraper",
            description=(
                "Advanced web scraper for extracting comprehensive content from web pages. "
                "Supports text, metadata, links, images, tables, and JavaScript-rendered content. "
                "Returns structured data from any web page."
            ),
            args_schema=ScrapeInput
        )
        
except ImportError:
    pass  # LangChain not installed


if __name__ == "__main__":
    # Example usage
    print("=== Advanced Web Scraper Tool Test ===\n")
    
    # Test basic scraping
    result = scrape_webpage("https://example.com")
    if result.get('success', False):
        print(f"✓ Successfully scraped: {result['url']}")
        print(f"  Title: {result.get('title', 'No title')}")
        print(f"  Word count: {result.get('word_count', 0)}")
    else:
        print(f"✗ Failed: {result.get('error')}")
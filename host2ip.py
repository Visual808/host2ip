#!/usr/bin/env python3
"""
CSV Domain Resolver

A utility script to resolve domain names to IP addresses within CSV files.
Reads a CSV file containing domain names, performs DNS resolution to get IP addresses,
and outputs an enhanced CSV with the resolved IP information.

Features:
- Concurrent DNS resolution for improved performance
- Flexible column detection and mapping
- Comprehensive error handling and validation
- Multiple output formats and filtering options
- Detailed logging and progress tracking

Version: 2.0
License: MIT
"""

import pandas as pd
import socket
import argparse
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import ipaddress
import re


class CSVDomainResolver:
    """
    A high-performance CSV domain resolver with advanced features for DNS resolution,
    data validation, and flexible output formatting.
    """
    
    def __init__(self, timeout: float = 5.0, max_workers: int = 50, 
                 retry_count: int = 2, validate_ips: bool = True):
        """
        Initialize the CSV Domain Resolver with configuration parameters.
        
        Args:
            timeout: DNS resolution timeout in seconds (default: 5.0)
            max_workers: Maximum concurrent threads for DNS resolution (default: 50)
            retry_count: Number of retries for failed DNS lookups (default: 2)
            validate_ips: Whether to validate resolved IP addresses (default: True)
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.validate_ips = validate_ips
        
        # DNS resolution statistics
        self.stats = {
            'total_domains': 0,
            'resolved': 0,
            'failed': 0,
            'invalid_domains': 0,
            'duplicate_domains': 0
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Common domain column name variations
        self.domain_column_names = [
            'Domain', 'domain', 'DOMAIN',
            'Domain Name', 'domain_name', 'DomainName',
            'Host', 'host', 'HOST',
            'Hostname', 'hostname', 'HostName',
            'URL', 'url', 'Website', 'website'
        ]
    
    def validate_domain(self, domain: str) -> Tuple[str, bool]:
        """
        Validate and clean domain name format.
        
        Performs basic validation and cleaning of domain names including:
        - Removing protocol prefixes (http://, https://)
        - Removing www. prefix if present
        - Removing paths and query parameters
        - Basic format validation
        
        Args:
            domain: Raw domain string from CSV
            
        Returns:
            Tuple of (cleaned_domain, is_valid)
            
        Examples:
            >>> resolver = CSVDomainResolver()
            >>> resolver.validate_domain("https://www.example.com/path")
            ('example.com', True)
            >>> resolver.validate_domain("invalid..domain")
            ('invalid..domain', False)
        """
        if not domain or not isinstance(domain, str):
            return ('', False)
        
        # Clean the domain string
        domain = str(domain).strip().lower()
        
        # Remove protocol prefixes
        for prefix in ['https://', 'http://', 'ftp://']:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove path and query parameters
        domain = domain.split('/')[0].split('?')[0].split('#')[0]
        
        # Basic validation checks
        if not domain:
            return ('', False)
        
        # Length check (RFC 1035)
        if len(domain) > 253:
            return (domain, False)
        
        # Basic format validation
        if '..' in domain or domain.startswith('.') or domain.endswith('.'):
            return (domain, False)
        
        # Must contain at least one dot (for TLD)
        if '.' not in domain:
            return (domain, False)
        
        # Basic character validation (simplified)
        if not re.match(r'^[a-z0-9.-]+$', domain):
            return (domain, False)
        
        return (domain, True)
    
    def resolve_domain_with_retry(self, domain: str) -> Dict[str, Union[str, bool, float]]:
        """
        Resolve a single domain with retry logic and detailed result information.
        
        Attempts DNS resolution with configurable retry logic and comprehensive
        error handling. Returns detailed information about the resolution attempt.
        
        Args:
            domain: Domain name to resolve
            
        Returns:
            Dictionary containing resolution results:
            - domain: Original domain name
            - ip_address: Resolved IP address (None if failed)
            - resolved: Boolean indicating success
            - resolution_time: Time taken for DNS resolution
            - error_type: Type of error if resolution failed
            
        Examples:
            >>> resolver = CSVDomainResolver()
            >>> result = resolver.resolve_domain_with_retry("google.com")
            >>> result['resolved']
            True
            >>> result['ip_address']
            '142.250.191.14'  # (example IP)
        """
        start_time = time.time()
        result = {
            'domain': domain,
            'ip_address': None,
            'resolved': False,
            'resolution_time': 0.0,
            'error_type': None
        }
        
        # Validate domain first
        cleaned_domain, is_valid = self.validate_domain(domain)
        if not is_valid:
            result['error_type'] = 'invalid_domain'
            result['resolution_time'] = time.time() - start_time
            return result
        
        # Attempt resolution with retries
        for attempt in range(self.retry_count + 1):
            try:
                # Set timeout for this resolution attempt
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(self.timeout)
                
                # Perform DNS lookup
                ip_address = socket.gethostbyname(cleaned_domain)
                
                # Validate IP address if requested
                if self.validate_ips:
                    try:
                        ipaddress.ip_address(ip_address)
                    except ValueError:
                        result['error_type'] = 'invalid_ip'
                        socket.setdefaulttimeout(old_timeout)
                        continue
                
                # Successful resolution
                result['ip_address'] = ip_address
                result['resolved'] = True
                result['resolution_time'] = time.time() - start_time
                
                socket.setdefaulttimeout(old_timeout)
                return result
                
            except socket.gaierror as e:
                # DNS resolution failed
                result['error_type'] = f'dns_error_{e.errno}' if hasattr(e, 'errno') else 'dns_error'
                socket.setdefaulttimeout(old_timeout)
                
            except socket.timeout:
                # Timeout occurred
                result['error_type'] = 'timeout'
                socket.setdefaulttimeout(old_timeout)
                if attempt < self.retry_count:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                    
            except Exception as e:
                # Unexpected error
                result['error_type'] = f'unknown_error: {type(e).__name__}'
                socket.setdefaulttimeout(old_timeout)
                self.logger.debug(f"Unexpected error resolving {cleaned_domain}: {e}")
                break
        
        result['resolution_time'] = time.time() - start_time
        return result
    
    def find_domain_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the domain column in the DataFrame.
        
        Searches for common domain column names and returns the first match.
        If multiple potential columns exist, prioritizes based on naming conventions.
        
        Args:
            df: Input DataFrame to search
            
        Returns:
            Name of the detected domain column
            
        Raises:
            ValueError: If no domain column is found
        """
        available_columns = df.columns.tolist()
        
        # Check for exact matches first
        for col_name in self.domain_column_names:
            if col_name in available_columns:
                self.logger.info(f"Found domain column: '{col_name}'")
                return col_name
        
        # Check for partial matches
        for col_name in available_columns:
            col_lower = col_name.lower()
            if any(keyword in col_lower for keyword in ['domain', 'host', 'url', 'website']):
                self.logger.info(f"Found potential domain column: '{col_name}'")
                return col_name
        
        # Show available columns for debugging
        self.logger.error(f"Available columns: {available_columns}")
        raise ValueError(
            f"No domain column found. Expected one of: {', '.join(self.domain_column_names[:5])}... "
            f"or columns containing 'domain', 'host', 'url', or 'website'"
        )
    
    def load_csv_data(self, input_file: Path) -> pd.DataFrame:
        """
        Load and validate CSV data with comprehensive error handling.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            Various exceptions for file loading issues
        """
        try:
            # Try different encodings if UTF-8 fails
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    if encoding == encodings[-1]:  # Last encoding failed
                        raise
                    continue
            
            # Validate DataFrame
            if df.empty:
                raise ValueError("CSV file is empty")
            
            self.logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or invalid")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file: {e}")
    
    def resolve_domains_concurrent(self, domains: List[str]) -> List[Dict]:
        """
        Resolve multiple domains concurrently for improved performance.
        
        Uses ThreadPoolExecutor to perform DNS resolution in parallel,
        significantly reducing processing time for large domain lists.
        
        Args:
            domains: List of domain names to resolve
            
        Returns:
            List of resolution result dictionaries
        """
        results = []
        
        # Filter out empty/invalid entries
        valid_domains = [d for d in domains if d and str(d).strip()]
        
        if not valid_domains:
            self.logger.warning("No valid domains to resolve")
            return results
        
        # Use ThreadPoolExecutor for concurrent DNS lookups
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all resolution tasks
            future_to_domain = {
                executor.submit(self.resolve_domain_with_retry, domain): domain 
                for domain in valid_domains
            }
            
            # Process results as they complete
            with tqdm(total=len(valid_domains), desc="Resolving domains", unit="domain") as pbar:
                for future in as_completed(future_to_domain):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update statistics
                        if result['resolved']:
                            self.stats['resolved'] += 1
                        else:
                            self.stats['failed'] += 1
                            if result['error_type'] == 'invalid_domain':
                                self.stats['invalid_domains'] += 1
                                
                    except Exception as e:
                        self.logger.error(f"Error processing domain: {e}")
                        self.stats['failed'] += 1
                    
                    pbar.update(1)
        
        return results
    
    def enhance_dataframe_with_results(self, df: pd.DataFrame, domain_column: str, 
                                     results: List[Dict], include_metadata: bool = False) -> pd.DataFrame:
        """
        Enhance the original DataFrame with DNS resolution results.
        
        Args:
            df: Original DataFrame
            domain_column: Name of the domain column
            results: List of resolution results
            include_metadata: Whether to include additional metadata columns
            
        Returns:
            Enhanced DataFrame with IP addresses and optional metadata
        """
        # Create a mapping from domain to results
        domain_to_result = {result['domain']: result for result in results}
        
        # Add IP addresses
        df['ip_address'] = df[domain_column].apply(
            lambda domain: domain_to_result.get(str(domain).strip(), {}).get('ip_address')
        )
        
        # Add metadata columns if requested
        if include_metadata:
            df['resolved'] = df[domain_column].apply(
                lambda domain: domain_to_result.get(str(domain).strip(), {}).get('resolved', False)
            )
            df['resolution_time'] = df[domain_column].apply(
                lambda domain: domain_to_result.get(str(domain).strip(), {}).get('resolution_time', 0.0)
            )
            df['error_type'] = df[domain_column].apply(
                lambda domain: domain_to_result.get(str(domain).strip(), {}).get('error_type')
            )
        
        return df
    
    def save_results(self, df: pd.DataFrame, output_file: Path, 
                    only_resolved: bool = False) -> None:
        """
        Save the enhanced DataFrame to output file.
        
        Args:
            df: Enhanced DataFrame with resolution results
            output_file: Path to output file
            only_resolved: If True, only save rows with resolved IP addresses
        """
        try:
            # Filter to only resolved domains if requested
            if only_resolved:
                original_count = len(df)
                df = df[df['ip_address'].notna()].copy()
                filtered_count = len(df)
                self.logger.info(f"Filtered to {filtered_count} resolved domains (from {original_count} total)")
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            
        except PermissionError:
            raise PermissionError(f"Permission denied writing to '{output_file}'")
        except Exception as e:
            raise Exception(f"Failed to save results to '{output_file}': {e}")
    
    def print_summary_statistics(self) -> None:
        """Print detailed summary statistics of the resolution process."""
        total = self.stats['total_domains']
        resolved = self.stats['resolved']
        failed = self.stats['failed']
        
        if total == 0:
            print("❌ No domains were processed")
            return
        
        success_rate = (resolved / total) * 100 if total > 0 else 0
        
        print(f"\n✅ DNS Resolution Summary:")
        print(f"   Total domains processed: {total}")
        print(f"   Successfully resolved: {resolved} ({success_rate:.1f}%)")
        print(f"   Failed to resolve: {failed}")
        print(f"   Invalid domain formats: {self.stats['invalid_domains']}")
        
        if self.stats['duplicate_domains'] > 0:
            print(f"   Duplicate domains found: {self.stats['duplicate_domains']}")
    
    def resolve_csv_domains(self, input_file: Path, output_file: Path,
                          domain_column: Optional[str] = None,
                          include_metadata: bool = False,
                          only_resolved: bool = False) -> None:
        """
        Main processing function to resolve domains in CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            domain_column: Name of domain column (auto-detected if None)
            include_metadata: Whether to include resolution metadata
            only_resolved: Whether to filter output to only resolved domains
        """
        try:
            # Load CSV data
            self.logger.info(f"Loading CSV data from {input_file}")
            df = self.load_csv_data(input_file)
            
            # Find domain column
            if domain_column:
                if domain_column not in df.columns:
                    raise ValueError(f"Specified domain column '{domain_column}' not found in CSV")
            else:
                domain_column = self.find_domain_column(df)
            
            # Extract domains for processing
            domains = df[domain_column].tolist()
            self.stats['total_domains'] = len(domains)
            
            # Check for duplicates
            unique_domains = list(set(str(d).strip().lower() for d in domains if d))
            self.stats['duplicate_domains'] = len(domains) - len(unique_domains)
            
            if not domains:
                print("❌ No domains found in the specified column")
                return
            
            print(f"📋 Processing {len(domains)} domains from column '{domain_column}'")
            
            # Resolve domains concurrently
            start_time = time.time()
            results = self.resolve_domains_concurrent(domains)
            end_time = time.time()
            
            # Enhance DataFrame with results
            enhanced_df = self.enhance_dataframe_with_results(
                df, domain_column, results, include_metadata
            )
            
            # Save results
            self.save_results(enhanced_df, output_file, only_resolved)
            
            # Print summary
            self.print_summary_statistics()
            print(f"   Processing time: {end_time - start_time:.2f} seconds")
            print(f"   Output saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            print(f"❌ Error: {e}")
            sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Resolve domain names to IP addresses in CSV files",
        epilog="""
Examples:
  %(prog)s domains.csv resolved_domains.csv
  %(prog)s -c "Domain Name" -m -t 10 input.csv output.csv
  %(prog)s --only-resolved --workers 100 domains.csv filtered.csv

Input CSV format:
  Must contain a column with domain names. Common column names are
  automatically detected: Domain, domain, Host, hostname, URL, etc.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file containing domain names"
    )
    
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV file with resolved IP addresses"
    )
    
    parser.add_argument(
        "-c", "--column",
        type=str,
        help="Name of the domain column (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=5.0,
        help="DNS resolution timeout in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=50,
        help="Maximum concurrent workers (default: 50)"
    )
    
    parser.add_argument(
        "-r", "--retries",
        type=int,
        default=2,
        help="Number of retries for failed lookups (default: 2)"
    )
    
    parser.add_argument(
        "-m", "--metadata",
        action="store_true",
        help="Include resolution metadata (resolved status, timing, errors)"
    )
    
    parser.add_argument(
        "--only-resolved",
        action="store_true",
        help="Only include rows with successfully resolved domains"
    )
    
    parser.add_argument(
        "--no-ip-validation",
        action="store_true",
        help="Skip IP address format validation"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="CSV Domain Resolver 2.0"
    )
    
    return parser


def main() -> None:
    """Main entry point for the CSV domain resolver utility."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.input_csv.exists():
        print(f"❌ Input file '{args.input_csv}' does not exist")
        sys.exit(1)
    
    if args.timeout <= 0:
        print("❌ Timeout must be positive")
        sys.exit(1)
    
    if args.workers <= 0 or args.workers > 200:
        print("❌ Workers must be between 1 and 200")
        sys.exit(1)
    
    # Initialize resolver
    resolver = CSVDomainResolver(
        timeout=args.timeout,
        max_workers=args.workers,
        retry_count=args.retries,
        validate_ips=not args.no_ip_validation
    )
    
    # Process CSV
    resolver.resolve_csv_domains(
        input_file=args.input_csv,
        output_file=args.output_csv,
        domain_column=args.column,
        include_metadata=args.metadata,
        only_resolved=args.only_resolved
    )


if __name__ == "__main__":
    main()

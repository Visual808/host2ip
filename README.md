# host2ip.py - CSV Domain Resolver

A high-performance Python utility for resolving domain names to IP addresses within CSV files. This tool reads CSV files containing domain names, performs concurrent DNS resolution, and outputs enhanced CSV files with IP address information.

## 🚀 Features

- **Concurrent DNS Resolution**: Multi-threaded processing for fast resolution of large domain lists
- **Flexible Column Detection**: Automatically detects domain columns or allows manual specification
- **Robust Error Handling**: Comprehensive validation and retry logic for reliable results
- **Multiple Output Formats**: Options for filtered output and detailed metadata inclusion
- **Progress Tracking**: Real-time progress bars and detailed statistics
- **Smart Domain Cleaning**: Handles URLs, removes protocols, and validates domain formats
- **Comprehensive Logging**: Detailed logging with configurable verbosity levels

## 📋 Requirements

- Python 3.6+
- Required packages:
  ```bash
  pandas
  tqdm
  ```

## 🔧 Installation

1. **Clone or download the script:**
   ```bash
   # Download the script
   wget https://example.com/host2ip.py  # Replace with actual URL
   # or
   curl -O https://example.com/host2ip.py
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas tqdm
   ```

3. **Make the script executable (Linux/Mac):**
   ```bash
   chmod +x host2ip.py
   ```

## 📖 Usage

### Basic Usage

```bash
python host2ip.py input.csv output.csv
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--column` | `-c` | Name of the domain column | Auto-detected |
| `--timeout` | `-t` | DNS resolution timeout (seconds) | 5.0 |
| `--workers` | `-w` | Maximum concurrent workers | 50 |
| `--retries` | `-r` | Number of retries for failed lookups | 2 |
| `--metadata` | `-m` | Include resolution metadata columns | False |
| `--only-resolved` | | Only output successfully resolved domains | False |
| `--no-ip-validation` | | Skip IP address format validation | False |
| `--verbose` | `-v` | Enable verbose logging | False |
| `--version` | | Show version information | - |

### Examples

**Basic domain resolution:**
```bash
python host2ip.py domains.csv resolved_domains.csv
```

**Specify domain column and include metadata:**
```bash
python host2ip.py -c "Domain Name" -m domains.csv output.csv
```

**High-performance processing with custom settings:**
```bash
python host2ip.py -w 100 -t 10 -r 3 domains.csv output.csv
```

**Filter to only resolved domains:**
```bash
python host2ip.py --only-resolved domains.csv filtered_output.csv
```

**Verbose logging for troubleshooting:**
```bash
python host2ip.py -v -m domains.csv debug_output.csv
```

## 📁 Input CSV Format

The input CSV file should contain a column with domain names. The script automatically detects common column names:

- `Domain`, `domain`, `DOMAIN`
- `Domain Name`, `domain_name`, `DomainName`
- `Host`, `host`, `HOST`
- `Hostname`, `hostname`, `HostName`
- `URL`, `url`, `Website`, `website`

### Example Input CSV:
```csv
Domain,Company,Category
google.com,Google Inc,Search
github.com,Microsoft,Development
stackoverflow.com,Stack Overflow,Q&A
invalid-domain-example,Test Company,Test
```

## 📊 Output Format

### Basic Output
The output CSV includes all original columns plus an `ip_address` column:

```csv
Domain,Company,Category,ip_address
google.com,Google Inc,Search,172.217.164.110
github.com,Microsoft,Development,140.82.112.4
stackoverflow.com,Stack Overflow,Q&A,151.101.193.69
invalid-domain-example,Test Company,Test,
```

### With Metadata (`--metadata` flag)
Additional columns provide detailed resolution information:

```csv
Domain,Company,Category,ip_address,resolved,resolution_time,error_type
google.com,Google Inc,Search,172.217.164.110,True,0.125,
github.com,Microsoft,Development,140.82.112.4,True,0.089,
stackoverflow.com,Stack Overflow,Q&A,151.101.193.69,True,0.156,
invalid-domain-example,Test Company,Test,,False,0.001,invalid_domain
```

## 🛠️ Advanced Features

### Domain Cleaning and Validation

The script automatically cleans and validates domain names:

- **Removes protocols**: `https://example.com` → `example.com`
- **Removes www prefix**: `www.example.com` → `example.com`
- **Removes paths**: `example.com/path` → `example.com`
- **Validates format**: Checks for valid domain structure

### Performance Optimization

- **Concurrent Processing**: Uses ThreadPoolExecutor for parallel DNS lookups
- **Configurable Workers**: Adjust concurrent threads based on your system
- **Retry Logic**: Automatically retries failed lookups with exponential backoff
- **Timeout Management**: Prevents hanging on unresponsive DNS queries

### Error Handling

The script provides detailed error information:

- `dns_error`: DNS resolution failed
- `timeout`: DNS query timed out
- `invalid_domain`: Domain format is invalid
- `invalid_ip`: Resolved IP address is invalid

## 📈 Performance Tips

1. **Adjust Worker Count**: For large files, increase workers (`-w 100`)
2. **Optimize Timeout**: Balance speed vs accuracy (`-t 3` for faster processing)
3. **Use SSD Storage**: I/O performance affects large file processing
4. **Monitor Memory**: Large CSV files may require significant RAM

### Performance Benchmarks

| Domain Count | Workers | Avg Time | Memory Usage |
|--------------|---------|----------|--------------|
| 1,000 | 50 | ~15 seconds | ~50 MB |
| 10,000 | 100 | ~90 seconds | ~200 MB |
| 100,000 | 100 | ~12 minutes | ~1.5 GB |

## 🐛 Troubleshooting

### Common Issues

**"No domain column found" error:**
```bash
# Specify the column name manually
python host2ip.py -c "YourColumnName" input.csv output.csv
```

**Slow processing:**
```bash
# Increase workers and reduce timeout
python host2ip.py -w 100 -t 3 input.csv output.csv
```

**Memory issues with large files:**
```bash
# Process in smaller batches or increase system RAM
# Consider splitting large CSV files into smaller chunks
```

**Permission denied on output:**
```bash
# Check write permissions for output directory
chmod 755 /path/to/output/directory
```

### Debug Mode

Enable verbose logging to troubleshoot issues:
```bash
python host2ip.py -v input.csv output.csv
```

## 📚 Additional Resources

- [Python Socket Documentation](https://docs.python.org/3/library/socket.html)
- [Pandas CSV Documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [DNS Resolution Best Practices](https://tools.ietf.org/html/rfc1035)

---

**Note**: This tool performs DNS lookups which may be rate-limited by some DNS servers. For very large datasets, consider using multiple DNS servers or implementing additional rate limiting.

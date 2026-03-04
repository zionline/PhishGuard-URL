# PhishGuard-URL: Feature Description (All 77 Features)

All features are **numeric** (float or integer). No missing values in the UNB dataset.

---

## Group F_lex — Lexical Composition (15 features)

| # | Feature Name | Type | Description |
|---|---|---|---|
| 1 | urlLen | int | Total character length of the full URL |
| 2 | domainlength | int | Length of the domain component |
| 3 | pathLength | int | Length of the URL path |
| 4 | subDirLen | int | Length of subdirectory path (path minus filename) |
| 5 | fileNameLen | int | Length of the filename component |
| 6 | ArgLen | int | Length of the query string |
| 7 | UrlLetterCount | int | Number of alphabetic characters in full URL |
| 8 | UrlDigitCount | int | Number of digit characters in full URL |
| 9 | UrlVowelCount | int | Number of vowels (a,e,i,o,u) in full URL |
| 10 | HostVowelCount | int | Number of vowels in host/netloc |
| 11 | DomainDigitCount | int | Number of digits in domain |
| 12 | NumberofDotsinURL | int | Count of '.' characters in full URL |
| 13 | SpecialCharCount | int | Non-alphanumeric chars excluding common URL chars |
| 14 | tld | int | Length of top-level domain (e.g., 3 for .com) |
| 15 | host_letter_count | int | Number of alphabetic characters in host |

---

## Group F_rat — Structural Ratios (16 features)

| # | Feature Name | Type | Description |
|---|---|---|---|
| 16 | pathurlRatio | float | pathLength / urlLen |
| 17 | domainUrlRatio | float | domainlength / urlLen |
| 18 | argDomainRatio | float | ArgLen / domainlength |
| 19 | argUrlRatio | float | ArgLen / urlLen |
| 20 | pathDomainRatio | float | pathLength / domainlength |
| 21 | subdirDomainRatio | float | subDirLen / domainlength |
| 22 | subdirUrlRatio | float | subDirLen / urlLen |
| 23 | filenameDomainRatio | float | fileNameLen / domainlength |
| 24 | filenameUrlRatio | float | fileNameLen / urlLen |
| 25 | ExtUrlRatio | float | Extension length / urlLen |
| 26 | ExtDomainRatio | float | Extension length / domainlength |
| 27 | queryArgRatio | float | Number of query variables / urlLen |
| 28 | digitLetterRatio | float | Digit count / letter count |
| 29 | vowelLetterRatio | float | Vowel count / letter count |
| 30 | dotRatioUrl | float | Dot count / urlLen |
| 31 | specialCharRatio | float | Special char count / urlLen |

---

## Group F_tok — Token Statistics (14 features)

Tokenisation uses delimiters: `.` `/` `-` `_` `~`

| # | Feature Name | Type | Description |
|---|---|---|---|
| 32 | domain_token_count | int | Number of tokens in domain (split on `.`) |
| 33 | domain_avg_token_length | float | Average token length in domain |
| 34 | LongestPathTokenLength | int | Length of longest token in path |
| 35 | path_token_count | int | Number of tokens in path |
| 36 | path_avg_token_length | float | Average token length in path |
| 37 | url_token_count | int | Number of tokens in full URL |
| 38 | url_avg_token_length | float | Average token length in full URL |
| 39 | subdomain_token_count | int | Number of tokens in subdomain |
| 40 | subdomain_max_token_len | int | Longest token in subdomain |
| 41 | domain_longest_token | int | Longest token in domain |
| 42 | avg_token_length_url | float | urlLen / token_count |
| 43 | token_count_ratio | float | domain_token_count / url_token_count |
| 44 | path_domain_token_ratio | float | path_token_count / domain_token_count |
| 45 | sub_domain_token_ratio | float | subdomain_token_count / domain_token_count |

---

## Group F_sym — Symbol and Delimiter Counts (19 features)

| # | Feature Name | Type | Description |
|---|---|---|---|
| 46 | SymbolCount_URL | int | Non-alphanumeric chars in full URL |
| 47 | SymbolCount_Domain | int | Non-alphanumeric chars in domain |
| 48 | SymbolCount_Path | int | Non-alphanumeric chars in path |
| 49 | SymbolCount_File | int | Non-alphanumeric chars in filename |
| 50 | SymbolCount_Ext | int | Non-alphanumeric chars in extension |
| 51 | SymbolCount_Query | int | Non-alphanumeric chars in query string |
| 52 | delimeter_url | int | Count of `-_.~` in full URL |
| 53 | delimeter_path | int | Count of `-_.~` in path |
| 54 | delimeter_domain | int | Count of `-.` in domain |
| 55 | NumberRate_URL | float | Digit ratio in full URL |
| 56 | NumberRate_Domain | float | Digit ratio in domain |
| 57 | NumberRate_Path | float | Digit ratio in path |
| 58 | Hyphen_URL | int | Count of `-` in full URL |
| 59 | Hyphen_Domain | int | Count of `-` in domain |
| 60 | Underscore_URL | int | Count of `_` in full URL |
| 61 | AtSign_URL | int | Count of `@` in full URL |
| 62 | Ampersand_URL | int | Count of `&` in full URL |
| 63 | sensitive_keyword | int | 1 if any k ∈ K appears in URL, else 0 |
| 64 | URLQueries_variable | int | Number of query key-value pairs |

---

## Group F_ent — Shannon Entropy (13 features)

H(s) = -∑ p(c) log₂ p(c) over character alphabet of string s.

| # | Feature Name | Type | Description |
|---|---|---|---|
| 65 | Entropy_URL | float | Entropy of full URL string |
| 66 | Entropy_Domain | float | Entropy of domain component |
| 67 | Entropy_Path | float | Entropy of path component |
| 68 | Entropy_File | float | Entropy of filename |
| 69 | Entropy_Ext | float | Entropy of file extension |
| 70 | Entropy_Query | float | Entropy of query string |
| 71 | Entropy_SubDir | float | Entropy of subdirectory |
| 72 | Entropy_Subdomain | float | Entropy of subdomain |
| 73 | Entropy_Host | float | Entropy of host (netloc) |
| 74 | Entropy_Netloc | float | Entropy of netloc (host + port) |
| 75 | Entropy_TLD | float | Entropy of TLD |
| 76 | Entropy_Scheme | float | Entropy of scheme (http/https) |
| 77 | Entropy_Fragment | float | Entropy of URL fragment (#...) |

---

## Sensitive Keyword Set K

Used to compute feature `sensitive_keyword` (binary indicator):

```
login, secure, verify, account, update, confirm, banking, password,
credential, signin, sign-in, paypal, ebay, amazon, apple, microsoft,
google, webscr, submit, checkout, billing, validate
```

Feature value: `x_kw = 1[∃ k ∈ K : k ∈ url]`

"""
feature_extraction.py
=====================
Algorithm 1: URL Feature Extraction Pipeline
PhishGuard-URL (Molefi, 2026)

Extracts all 77 features from a raw URL string, partitioned into:
  F_lex  (15 dims) -- lexical composition
  F_rat  (16 dims) -- structural ratios
  F_tok  (14 dims) -- token statistics
  F_sym  (19 dims) -- symbol and delimiter counts
  F_ent  (13 dims) -- Shannon entropy

Usage
-----
  from feature_extraction import extract_features, extract_features_batch
  x = extract_features("http://login-secure.example.com/verify?id=1")
  # returns dict with 77 keys

  import pandas as pd
  df = pd.read_csv("data/url_data.csv")
  features_df = extract_features_batch(df["url"])
"""

import re
import math
import urllib.parse
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Sensitive keyword set K (Definition used in Algorithm 1, Step 17)
# ------------------------------------------------------------------
SENSITIVE_KEYWORDS = {
    "login", "secure", "verify", "account", "update", "confirm",
    "banking", "password", "credential", "signin", "sign-in",
    "paypal", "ebay", "amazon", "apple", "microsoft", "google",
    "webscr", "submit", "checkout", "billing", "validate",
}

# ------------------------------------------------------------------
# Helper: Shannon entropy  H(s) = -sum p(c) log2 p(c)
# (Definition 5 in the paper)
# ------------------------------------------------------------------
def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of string s over its character alphabet."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ------------------------------------------------------------------
# Helper: safe division
# ------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# ------------------------------------------------------------------
# Helper: tokenise a string on URL delimiters
# ------------------------------------------------------------------
def _tokenise(s: str) -> List[str]:
    """Split on delimiters: . / - _ ~"""
    return [t for t in re.split(r"[.\-_/~]", s) if t]


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------
def extract_features(url: str) -> Dict[str, float]:
    """
    Extract all 77 URL features from a raw URL string.

    Parameters
    ----------
    url : str
        Raw URL string, e.g. "http://login-secure.example.com/path?q=1"

    Returns
    -------
    dict
        Ordered dictionary with 77 float-valued features.
    """
    features: Dict[str, float] = {}

    # ---------------------------------------------------------------
    # Parse URL into components
    # ---------------------------------------------------------------
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = urllib.parse.urlparse("")

    scheme    = parsed.scheme or ""
    netloc    = parsed.netloc or ""
    path      = parsed.path   or ""
    query     = parsed.query  or ""
    fragment  = parsed.fragment or ""

    # Separate subdomain / domain / TLD from netloc
    # Remove port if present
    host = netloc.split(":")[0] if ":" in netloc else netloc
    host_parts = host.split(".")
    tld        = host_parts[-1] if len(host_parts) >= 1 else ""
    domain     = host_parts[-2] if len(host_parts) >= 2 else host
    subdomain  = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""

    # Path components
    path_parts = [p for p in path.split("/") if p]
    filename   = path_parts[-1] if path_parts else ""
    # Extension
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    # Subdirectory (path without filename)
    subdir = "/".join(path_parts[:-1]) if len(path_parts) > 1 else ""

    # Query string tokens (key=value pairs)
    query_params = urllib.parse.parse_qs(query, keep_blank_values=True)
    n_query_vars = len(query_params)

    # ------------------------------------------------------------------
    # F_lex: Lexical composition features (15 dims)
    # ------------------------------------------------------------------
    url_len        = len(url)
    domain_length  = len(domain)
    path_length    = len(path)
    subdir_len     = len(subdir)
    filename_len   = len(filename)
    arg_len        = len(query)

    n_vowels_url    = sum(1 for c in url    if c.lower() in "aeiou")
    n_digits_url    = sum(1 for c in url    if c.isdigit())
    n_letters_url   = sum(1 for c in url    if c.isalpha())
    n_vowels_host   = sum(1 for c in host   if c.lower() in "aeiou")
    n_digits_domain = sum(1 for c in domain if c.isdigit())
    n_dots_url      = url.count(".")
    n_special_url   = sum(1 for c in url if not c.isalnum() and c not in ":/.-_~?#&=@%+")

    # TLD encoded as length (simple numeric proxy)
    tld_length = len(tld)

    features.update({
        "urlLen":           url_len,
        "domainlength":     domain_length,
        "pathLength":       path_length,
        "subDirLen":        subdir_len,
        "fileNameLen":      filename_len,
        "ArgLen":           arg_len,
        "UrlLetterCount":   n_letters_url,
        "UrlDigitCount":    n_digits_url,
        "UrlVowelCount":    n_vowels_url,
        "HostVowelCount":   n_vowels_host,
        "DomainDigitCount": n_digits_domain,
        "NumberofDotsinURL": n_dots_url,
        "SpecialCharCount": n_special_url,
        "tld":              tld_length,
        "host_letter_count": sum(1 for c in host if c.isalpha()),
    })

    # ------------------------------------------------------------------
    # F_rat: Structural ratio features (16 dims)
    # ------------------------------------------------------------------
    features.update({
        "pathurlRatio":      _safe_div(path_length,   url_len),
        "domainUrlRatio":    _safe_div(domain_length, url_len),
        "argDomainRatio":    _safe_div(arg_len,       domain_length),
        "argUrlRatio":       _safe_div(arg_len,       url_len),
        "pathDomainRatio":   _safe_div(path_length,   domain_length),
        "subdirDomainRatio": _safe_div(subdir_len,    domain_length),
        "subdirUrlRatio":    _safe_div(subdir_len,    url_len),
        "filenameDomainRatio": _safe_div(filename_len, domain_length),
        "filenameUrlRatio":  _safe_div(filename_len,  url_len),
        "ExtUrlRatio":       _safe_div(len(ext),      url_len),
        "ExtDomainRatio":    _safe_div(len(ext),      domain_length),
        "queryArgRatio":     _safe_div(n_query_vars,  url_len + 1),
        "digitLetterRatio":  _safe_div(n_digits_url,  n_letters_url + 1),
        "vowelLetterRatio":  _safe_div(n_vowels_url,  n_letters_url + 1),
        "dotRatioUrl":       _safe_div(n_dots_url,    url_len),
        "specialCharRatio":  _safe_div(n_special_url, url_len),
    })

    # ------------------------------------------------------------------
    # F_tok: Token statistics features (14 dims)
    # ------------------------------------------------------------------
    domain_tokens    = _tokenise(domain)
    path_tokens      = _tokenise(path)
    subdomain_tokens = _tokenise(subdomain)
    url_tokens       = _tokenise(url)

    def _token_stats(tokens):
        if not tokens:
            return 0, 0.0, 0
        lengths = [len(t) for t in tokens]
        return len(tokens), np.mean(lengths), max(lengths)

    domain_tc, domain_avg_tl, domain_max_tl  = _token_stats(domain_tokens)
    path_tc,   path_avg_tl,   path_max_tl    = _token_stats(path_tokens)
    url_tc,    url_avg_tl,    _              = _token_stats(url_tokens)
    sub_tc,    _,             sub_max_tl     = _token_stats(subdomain_tokens)

    features.update({
        "domain_token_count":        domain_tc,
        "domain_avg_token_length":   domain_avg_tl,
        "LongestPathTokenLength":    path_max_tl,
        "path_token_count":          path_tc,
        "path_avg_token_length":     path_avg_tl,
        "url_token_count":           url_tc,
        "url_avg_token_length":      url_avg_tl,
        "subdomain_token_count":     sub_tc,
        "subdomain_max_token_len":   sub_max_tl,
        "domain_longest_token":      domain_max_tl,
        "avg_token_length_url":      _safe_div(url_len, url_tc + 1),
        "token_count_ratio":         _safe_div(domain_tc, url_tc + 1),
        "path_domain_token_ratio":   _safe_div(path_tc, domain_tc + 1),
        "sub_domain_token_ratio":    _safe_div(sub_tc, domain_tc + 1),
    })

    # ------------------------------------------------------------------
    # F_sym: Symbol and delimiter count features (19 dims)
    # ------------------------------------------------------------------
    def _count_sym(s, sym): return s.count(sym)

    # Keyword presence indicator: x_kw = 1[exists k in K : k in u]
    url_lower   = url.lower()
    kw_flag     = int(any(k in url_lower for k in SENSITIVE_KEYWORDS))

    features.update({
        "SymbolCount_URL":      sum(1 for c in url    if not c.isalnum()),
        "SymbolCount_Domain":   sum(1 for c in domain if not c.isalnum()),
        "SymbolCount_Path":     sum(1 for c in path   if not c.isalnum()),
        "SymbolCount_File":     sum(1 for c in filename if not c.isalnum()),
        "SymbolCount_Ext":      sum(1 for c in ext    if not c.isalnum()),
        "SymbolCount_Query":    sum(1 for c in query  if not c.isalnum()),
        "delimeter_url":        sum(_count_sym(url,   s) for s in "-_.~"),
        "delimeter_path":       sum(_count_sym(path,  s) for s in "-_.~"),
        "delimeter_domain":     sum(_count_sym(domain, s) for s in "-_."),
        "NumberRate_URL":       _safe_div(n_digits_url, url_len + 1),
        "NumberRate_Domain":    _safe_div(n_digits_domain, domain_length + 1),
        "NumberRate_Path":      _safe_div(
                                    sum(1 for c in path if c.isdigit()),
                                    path_length + 1),
        "Hyphen_URL":           _count_sym(url,    "-"),
        "Hyphen_Domain":        _count_sym(domain, "-"),
        "Underscore_URL":       _count_sym(url,    "_"),
        "AtSign_URL":           _count_sym(url,    "@"),
        "Ampersand_URL":        _count_sym(url,    "&"),
        "sensitive_keyword":    kw_flag,
        "URLQueries_variable":  n_query_vars,
    })

    # ------------------------------------------------------------------
    # F_ent: Shannon entropy features (13 dims)
    # ------------------------------------------------------------------
    features.update({
        "Entropy_URL":      _shannon_entropy(url),
        "Entropy_Domain":   _shannon_entropy(domain),
        "Entropy_Path":     _shannon_entropy(path),
        "Entropy_File":     _shannon_entropy(filename),
        "Entropy_Ext":      _shannon_entropy(ext),
        "Entropy_Query":    _shannon_entropy(query),
        "Entropy_SubDir":   _shannon_entropy(subdir),
        "Entropy_Subdomain":_shannon_entropy(subdomain),
        "Entropy_Host":     _shannon_entropy(host),
        "Entropy_Netloc":   _shannon_entropy(netloc),
        "Entropy_TLD":      _shannon_entropy(tld),
        "Entropy_Scheme":   _shannon_entropy(scheme),
        "Entropy_Fragment": _shannon_entropy(fragment),
    })

    # Sanity check: exactly 77 features
    assert len(features) == 77, (
        f"Expected 77 features, got {len(features)}. "
        f"Missing: {set(range(77)) - set(range(len(features)))}"
    )

    return features


# ------------------------------------------------------------------
# Batch extraction (for a pandas Series of URLs)
# ------------------------------------------------------------------
def extract_features_batch(url_series: pd.Series) -> pd.DataFrame:
    """
    Extract features for an entire column of URLs.

    Parameters
    ----------
    url_series : pd.Series
        Series of raw URL strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n, 77), one row per URL.
    """
    records = [extract_features(url) for url in url_series]
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# CLI: test on a single URL
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 \
        else "http://login-secure-verify.paypal.com.phish.biz/account/update?id=12345"

    print(f"\nURL: {test_url}\n")
    feats = extract_features(test_url)
    print(f"{'Feature':<35} {'Value':>10}")
    print("-" * 48)
    for k, v in feats.items():
        print(f"{k:<35} {v:>10.4f}")
    print(f"\nTotal features extracted: {len(feats)}")

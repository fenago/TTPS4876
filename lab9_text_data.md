# Lab 9: Working with Text Data - Cleaning and Extracting Insights

## Lab Objectives

**🎯 IMPORTANT NOTE:** Text data is everywhere in business - property descriptions, customer reviews, product names, survey responses. Most real-world datasets are messy with inconsistent formatting, typos, and mixed cases. We'll master the essential skills to clean, standardize, and extract insights from text data efficiently.

By the end of this lab, you will be able to:

1. **Clean messy text data professionally** - Handle real-world inconsistencies and formatting issues
2. **Extract business intelligence from text** - Find patterns and insights hidden in unstructured data
3. **Standardize text for analysis** - Create consistent formats for reliable comparisons
4. **Apply string methods strategically** - Choose the right technique for each text problem
5. **Build text processing workflows** - Create repeatable processes for common text tasks

## 📚 Text Data Reality - Why This Matters

**💼 Business Context:** Text data in the real world is messy:
- **Property names**: "LUXURY Apt in MIDTOWN!!!" vs "Cozy midtown apartment"
- **Customer feedback**: Mixed languages, typos, inconsistent formatting
- **Product descriptions**: Inconsistent capitalization and spacing

**🎯 Our Mission:** Transform messy text into clean, analyzable data that reveals business insights.

---

## Step 1: Understanding Text Data Challenges

### 1.1 Real-World Text Problems

```python
import pandas as pd
import numpy as np

# Load Airbnb data to see real text challenges
url = "https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/AirBnB_NYC_2019.csv"
airbnb_df = pd.read_csv(url)

# Look at actual property names
sample_names = airbnb_df['name'].head(10)
print("🏠 Real Property Names (with all their messiness):")
for i, name in enumerate(sample_names):
    print(f"{i+1:2d}: '{name}'")
```

**🤔 Problems You Can See:**
- Mixed capitalization
- Extra spaces
- Special characters
- Inconsistent formatting

**💡 Why This Matters:** Inconsistent text makes analysis unreliable and insights misleading.

### 1.2 Text vs Numbers - Different Challenges

```python
# Text data characteristics
print("📊 Text Data Overview:")
print(f"Total properties: {len(airbnb_df):,}")
print(f"Properties with names: {airbnb_df['name'].notna().sum():,}")
print(f"Unique names: {airbnb_df['name'].nunique():,}")
print(f"Average name length: {airbnb_df['name'].str.len().mean():.1f} characters")
```

**🎯 Key Insight:** Text data requires different analysis approaches than numerical data.

---

## Step 2: Essential String Methods - Your Text Toolkit

### 2.1 Basic Cleaning Operations

**🧹 Standard Cleaning Workflow:**

```python
# Sample messy text for demonstration
messy_names = airbnb_df['name'].head(5)
print("Before cleaning:")
print(messy_names.tolist())

# Step 1: Remove extra whitespace
clean_step1 = messy_names.str.strip()

# Step 2: Standardize case
clean_step2 = clean_step1.str.title()

print("\nAfter basic cleaning:")
print(clean_step2.tolist())
```

**💡 Core Cleaning Methods:**
- `.str.strip()` - Remove leading/trailing spaces
- `.str.title()` - Proper case formatting
- `.str.lower()` - Lowercase for comparisons
- `.str.upper()` - Uppercase when needed

### 2.2 Text Length Analysis

**📏 Length-Based Insights:**

```python
# Analyze name lengths for business insights
name_lengths = airbnb_df['name'].str.len()

print("📏 Property Name Length Analysis:")
print(f"Average: {name_lengths.mean():.1f} characters")
print(f"Shortest: {name_lengths.min()} characters")
print(f"Longest: {name_lengths.max()} characters")

# Find extremes
shortest_name = airbnb_df.loc[name_lengths.idxmin(), 'name']
longest_name = airbnb_df.loc[name_lengths.idxmax(), 'name']

print(f"\nShortest: '{shortest_name}'")
print(f"Longest: '{longest_name[:60]}...'")
```

**🤔 Business Question:** Do longer property names correlate with higher prices?

```python
# Quick correlation check
correlation = airbnb_df['name'].str.len().corr(airbnb_df['price'])
print(f"Name length vs price correlation: {correlation:.3f}")
```

### 2.3 Text Replacement and Standardization

**🔄 Standardizing Common Variations:**

```python
# Common abbreviations in property names
sample_names = airbnb_df['name'].head(20)

# Standardize common terms
standardized = (sample_names
                .str.replace('&', 'and', regex=False)
                .str.replace('w/', 'with', regex=False)  
                .str.replace('NYC', 'New York City', regex=False)
                .str.replace('  ', ' ', regex=False))  # Remove double spaces

print("🔄 Before and after standardization:")
for original, clean in zip(sample_names.head(5), standardized.head(5)):
    if original != clean:
        print(f"Before: '{original}'")
        print(f"After:  '{clean}'\n")
```

**💼 Business Application:** Standardization enables better searching, grouping, and analysis.

---

## Step 3: Pattern Detection and Extraction

### 3.1 Finding Keywords and Themes

**🔍 Market Intelligence from Text:**

```python
# Analyze luxury indicators
luxury_keywords = ['luxury', 'premium', 'deluxe', 'executive', 'upscale']
luxury_pattern = '|'.join(luxury_keywords)

has_luxury = airbnb_df['name'].str.contains(luxury_pattern, case=False, na=False)
luxury_count = has_luxury.sum()

print(f"🏆 Properties with luxury keywords: {luxury_count:,} ({luxury_count/len(airbnb_df)*100:.1f}%)")

# Do they actually charge more?
luxury_avg_price = airbnb_df[has_luxury]['price'].mean()
regular_avg_price = airbnb_df[~has_luxury]['price'].mean()

print(f"💰 Luxury keyword average price: ${luxury_avg_price:.2f}")
print(f"💰 Regular property average price: ${regular_avg_price:.2f}")
print(f"💰 Premium: {(luxury_avg_price/regular_avg_price - 1)*100:.1f}%")
```

### 3.2 Location Extraction

**📍 Geographic Intelligence:**

```python
# Extract location mentions
location_keywords = ['manhattan', 'brooklyn', 'queens', 'bronx', 'times square', 'central park']

print("📍 Location Mentions in Property Names:")
for location in location_keywords:
    mentions = airbnb_df['name'].str.contains(location, case=False, na=False)
    count = mentions.sum()
    if count > 0:
        avg_price = airbnb_df[mentions]['price'].mean()
        print(f"{location.title()}: {count:,} mentions (avg price: ${avg_price:.0f})")
```

### 3.3 Text Classification

**📂 Categorizing Properties by Description:**

```python
# Create property categories based on text content
def categorize_property(name):
    if pd.isna(name):
        return 'Unknown'
    
    name_lower = name.lower()
    
    if any(word in name_lower for word in ['luxury', 'premium', 'deluxe']):
        return 'Luxury'
    elif any(word in name_lower for word in ['cozy', 'cute', 'charming']):
        return 'Cozy'
    elif any(word in name_lower for word in ['modern', 'new', 'renovated']):
        return 'Modern'
    elif any(word in name_lower for word in ['private', 'quiet', 'peaceful']):
        return 'Private'
    else:
        return 'Standard'

# Apply categorization
airbnb_df['property_category'] = airbnb_df['name'].apply(categorize_property)

# Analyze results
category_analysis = airbnb_df.groupby('property_category').agg({
    'price': ['mean', 'count'],
    'number_of_reviews': 'mean'
}).round(2)

print("🏷️ Property Categories Analysis:")
print(category_analysis)
```

---

## Step 4: Advanced Text Cleaning

### 4.1 Handling Special Characters

**🧹 Deep Cleaning for Analysis:**

```python
# Check for special characters that might cause issues
special_chars = airbnb_df['name'].str.contains('[^\w\s]', regex=True, na=False)
print(f"Properties with special characters: {special_chars.sum():,}")

# Clean special characters for analysis
def clean_for_analysis(text):
    if pd.isna(text):
        return text
    
    # Remove extra whitespace and special chars
    cleaned = (text.strip()
               .replace('\n', ' ')
               .replace('\t', ' ')
               .replace('  ', ' '))
    
    return cleaned

# Apply cleaning
airbnb_df['name_clean'] = airbnb_df['name'].apply(clean_for_analysis)

# Compare lengths
original_avg = airbnb_df['name'].str.len().mean()
cleaned_avg = airbnb_df['name_clean'].str.len().mean()
print(f"Average length before cleaning: {original_avg:.1f}")
print(f"Average length after cleaning: {cleaned_avg:.1f}")
```

### 4.2 Text Normalization Pipeline

**🔄 Systematic Text Processing:**

```python
def normalize_property_name(name):
    """Complete text normalization pipeline"""
    if pd.isna(name):
        return name
    
    # Step 1: Basic cleaning
    cleaned = name.strip().lower()
    
    # Step 2: Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    # Step 3: Standardize common abbreviations
    replacements = {
        '&': 'and',
        'w/': 'with',
        'nyc': 'new york city',
        'apt': 'apartment',
        'br': 'bedroom',
        'ba': 'bathroom'
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Step 4: Convert to title case
    return cleaned.title()

# Apply normalization
normalized_sample = airbnb_df['name'].head(10).apply(normalize_property_name)

print("🔄 Normalization Results:")
for original, normalized in zip(airbnb_df['name'].head(10), normalized_sample):
    print(f"Before: {original}")
    print(f"After:  {normalized}\n")
```

---

## Step 5: Text Analysis for Business Insights

### 5.1 Word Frequency Analysis

**📊 Most Common Terms:**

```python
# Extract all words from property names
all_text = ' '.join(airbnb_df['name'].dropna().str.lower())
words = all_text.split()

# Count word frequency
from collections import Counter
word_counts = Counter(words)

# Remove common stop words
stop_words = ['the', 'and', 'in', 'on', 'at', 'to', 'a', 'an', 'of', 'with', 'for']
business_words = {word: count for word, count in word_counts.items() 
                  if word not in stop_words and len(word) > 2}

# Top business-relevant words
top_words = Counter(business_words).most_common(15)

print("📊 Most Common Property Description Words:")
for word, count in top_words:
    print(f"{word.title()}: {count:,} times")
```

### 5.2 Host Name Analysis

**👤 Host Patterns:**

```python
# Analyze host name patterns
host_analysis = airbnb_df['host_name'].value_counts().head(10)

print("👤 Most Active Hosts:")
print(host_analysis)

# Analyze host business scale
host_property_counts = airbnb_df['host_name'].value_counts()
mega_hosts = host_property_counts[host_property_counts >= 10]

print(f"\n🏢 Business-Scale Hosts (10+ properties): {len(mega_hosts):,}")
print(f"🏢 Total properties managed by mega-hosts: {mega_hosts.sum():,}")
print(f"🏢 Percentage of market: {mega_hosts.sum()/len(airbnb_df)*100:.1f}%")
```

---

## Step 6: 🚀 Independent Challenge - Text Intelligence System

**Your Mission:** Build a comprehensive text analysis system that extracts business intelligence from property descriptions.

### 🎯 Challenge: Property Marketing Intelligence

**Business Scenario:** You're helping property owners optimize their listings by analyzing what language and keywords drive success.

**Your Tasks:**

**1. Marketing Keyword Analysis**
- Identify which keywords correlate with higher prices
- Find underutilized keywords that successful properties use
- Analyze keyword trends across different boroughs

**2. Property Description Optimization**
- Create categories based on description patterns
- Analyze optimal description length for different markets
- Identify successful marketing language patterns

**3. Competitive Text Intelligence**
- Compare text strategies across room types
- Find opportunities for better positioning
- Identify market gaps in property messaging

### 🛠️ Your Text Toolkit:

```python
# Starter framework - expand this
def analyze_keywords(df, keyword_list, price_col='price'):
    """Analyze keyword impact on pricing"""
    results = {}
    
    for keyword in keyword_list:
        has_keyword = df['name'].str.contains(keyword, case=False, na=False)
        if has_keyword.sum() > 0:
            results[keyword] = {
                'count': has_keyword.sum(),
                'avg_price': df[has_keyword][price_col].mean(),
                'percentage': has_keyword.sum() / len(df) * 100
            }
    
    return results

# Example usage
keywords_to_test = ['luxury', 'cozy', 'modern', 'private', 'spacious', 'clean', 'bright']
keyword_results = analyze_keywords(airbnb_df, keywords_to_test)
```

### 📊 Success Criteria:

- Find at least 3 actionable insights about marketing language
- Identify specific opportunities for property owners
- Create recommendations based on text analysis
- Build reusable functions for ongoing analysis

### 💡 Advanced Techniques to Try:

- **Sentiment analysis** of property descriptions
- **Text length optimization** for different markets
- **Competitive positioning** through language analysis
- **Success pattern recognition** in top-performing properties

---

## Step 7: What You've Mastered

**🎉 Excellent Work!** You've mastered the essential skills for working with text data in business contexts.

### ✅ **Text Analysis Skills:**
- Clean and standardize messy real-world text data
- Extract business insights from unstructured information
- Apply string methods strategically for specific problems
- Build systematic text processing workflows
- Identify patterns and opportunities in text content

### 🌟 **Business Applications:**
- **Marketing intelligence** from product descriptions
- **Customer sentiment** analysis from reviews
- **Competitive analysis** through text comparison
- **Content optimization** based on successful patterns
- **Data quality improvement** through text standardization

### 🎯 **Next Steps:**
- Apply text analysis to customer feedback data
- Combine text insights with numerical analysis
- Build automated text processing pipelines
- Explore advanced text analytics techniques

---

**🔤 Congratulations on mastering text data analysis!** You now have the skills to extract valuable business intelligence from the unstructured text data that's everywhere in modern business. These techniques will help you clean messy data, find hidden patterns, and generate insights that drive strategic decisions.

**The text toolkit is complete. Time to extract insights from the written world!** 🚀📝✨
# Lab 10: Data Visualization and Image Processing

## Lab Objectives

**🎯 IMPORTANT NOTE:** Visualizations turn data into insights that drive decisions. Charts communicate findings instantly where tables of numbers fail. We'll master Matplotlib for creating business-ready visualizations and explore PIL for basic image processing tasks that complement data analysis.

By the end of this lab, you will be able to:

1. **Create effective business visualizations** - Charts that communicate insights clearly
2. **Choose the right chart type** - Match visualization to business question
3. **Customize plots for professional presentation** - Clean, readable charts for stakeholders
4. **Handle image data basics** - Process and analyze visual content when needed
5. **Build visualization workflows** - Create repeatable chart generation processes

## 📚 Visualization Strategy - Charts That Drive Decisions

**📊 Why Visualizations Matter:**
- **Executive summary**: One chart replaces pages of analysis
- **Pattern recognition**: Human brain processes visuals 60,000x faster than text
- **Decision support**: Clear visuals drive faster, better business decisions

**🎯 Our Approach:** Create charts that answer specific business questions, not just display data.

---

## Step 1: Matplotlib Fundamentals - Your Visualization Engine

### 1.1 Setting Up Your Visualization Environment

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Load our data and set up plotting
url = "https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/AirBnB_NYC_2019.csv"
airbnb_df = pd.read_csv(url)

# Configure matplotlib for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print(f"📊 Ready to visualize {len(airbnb_df):,} properties")
```

### 1.2 Your First Business Chart

**💼 Business Question:** What's the price distribution across our market?

```python
# Create a basic histogram
plt.figure(figsize=(10, 6))
plt.hist(airbnb_df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('NYC Airbnb Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Price per Night ($)')
plt.ylabel('Number of Properties')
plt.grid(True, alpha=0.3)
plt.show()
```

**🤔 What This Chart Reveals:**
- Market concentration in specific price ranges
- Presence of luxury outliers
- Typical customer price points

### 1.3 Essential Plot Elements

**🎨 Professional Chart Components:**

```python
# Enhanced visualization with business context
plt.figure(figsize=(12, 6))

# Create the plot
plt.hist(airbnb_df['price'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')

# Add business context
median_price = airbnb_df['price'].median()
mean_price = airbnb_df['price'].mean()

plt.axvline(median_price, color='blue', linestyle='--', linewidth=2, 
           label=f'Median: ${median_price:.0f}')
plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: ${mean_price:.0f}')

# Professional formatting
plt.title('NYC Airbnb Market Analysis: Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Price per Night ($)', fontsize=12)
plt.ylabel('Number of Properties', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Add business insight as text
plt.text(0.7, 0.8, f'Total Properties: {len(airbnb_df):,}', 
         transform=plt.gca().transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()
```

---

## Step 2: Chart Types for Business Questions

### 2.1 Bar Charts - Comparing Categories

**📊 Business Question:** Which borough dominates the market?

```python
# Borough analysis
borough_counts = airbnb_df['neighbourhood_group'].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(borough_counts.index, borough_counts.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.title('Market Share by Borough', fontsize=14, fontweight='bold')
plt.xlabel('Borough')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Market share percentages
print("📊 Market Share Analysis:")
for borough, count in borough_counts.items():
    percentage = (count / len(airbnb_df)) * 100
    print(f"{borough}: {count:,} properties ({percentage:.1f}%)")
```

### 2.2 Scatter Plots - Relationships and Correlations

**🔍 Business Question:** Do more reviews correlate with higher prices?

```python
# Sample data for cleaner visualization
sample_data = airbnb_df.sample(2000)

plt.figure(figsize=(12, 6))

# Create scatter plot
plt.scatter(sample_data['number_of_reviews'], sample_data['price'], 
           alpha=0.6, s=30, color='darkblue')

plt.title('Price vs Review Count Relationship', fontsize=14, fontweight='bold')
plt.xlabel('Number of Reviews')
plt.ylabel('Price per Night ($)')
plt.grid(True, alpha=0.3)

# Add correlation info
correlation = airbnb_df['number_of_reviews'].corr(airbnb_df['price'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"💡 Insight: Correlation of {correlation:.3f} suggests {'strong' if abs(correlation) > 0.5 else 'weak'} relationship")
```

### 2.3 Box Plots - Distribution Comparisons

**📦 Business Question:** How do price ranges vary by room type?

```python
# Price distribution by room type
room_types = airbnb_df['room_type'].unique()
price_data = [airbnb_df[airbnb_df['room_type'] == rt]['price'] for rt in room_types]

plt.figure(figsize=(12, 6))
box_plot = plt.boxplot(price_data, labels=room_types, patch_artist=True)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
plt.xlabel('Room Type')
plt.ylabel('Price per Night ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Statistical summary
print("📊 Price Statistics by Room Type:")
for room_type in room_types:
    prices = airbnb_df[airbnb_df['room_type'] == room_type]['price']
    print(f"{room_type}: Median ${prices.median():.0f}, Mean ${prices.mean():.0f}")
```

---

## Step 3: Advanced Visualization Techniques

### 3.1 Subplots - Multiple Charts for Comprehensive Analysis

**📊 Business Dashboard:** Create a comprehensive market overview.

```python
# Create a 2x2 dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('NYC Airbnb Market Dashboard', fontsize=16, fontweight='bold')

# Chart 1: Price distribution
axes[0, 0].hist(airbnb_df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')

# Chart 2: Borough comparison
borough_avg = airbnb_df.groupby('neighbourhood_group')['price'].mean()
axes[0, 1].bar(borough_avg.index, borough_avg.values, color='lightcoral')
axes[0, 1].set_title('Average Price by Borough')
axes[0, 1].set_xlabel('Borough')
axes[0, 1].set_ylabel('Average Price ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Chart 3: Room type distribution
room_counts = airbnb_df['room_type'].value_counts()
axes[1, 0].pie(room_counts.values, labels=room_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title('Market Share by Room Type')

# Chart 4: Availability vs Price
sample = airbnb_df.sample(1000)
axes[1, 1].scatter(sample['availability_365'], sample['price'], alpha=0.6)
axes[1, 1].set_title('Availability vs Price')
axes[1, 1].set_xlabel('Days Available per Year')
axes[1, 1].set_ylabel('Price ($)')

plt.tight_layout()
plt.show()
```

### 3.2 Custom Styling for Professional Presentations

**🎨 Business-Ready Styling:**

```python
# Create a professional chart with custom styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate monthly trend (using last_review as proxy for booking activity)
airbnb_df['last_review'] = pd.to_datetime(airbnb_df['last_review'])
monthly_bookings = airbnb_df['last_review'].dt.month.value_counts().sort_index()

# Create styled line plot
ax.plot(monthly_bookings.index, monthly_bookings.values, 
        marker='o', linewidth=3, markersize=8, color='#2E86AB')
ax.fill_between(monthly_bookings.index, monthly_bookings.values, 
                alpha=0.3, color='#2E86AB')

# Professional styling
ax.set_title('Booking Activity by Month\n(Based on Last Review Date)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.grid(True, alpha=0.3)

# Month labels
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)

plt.tight_layout()
plt.show()
```

---

## Step 4: Introduction to PIL - Image Processing Basics

### 4.1 Creating Visual Data Summaries

**🖼️ Business Application:** Create visual summaries that complement your data analysis.

```python
# Create a simple data visualization as an image
from PIL import Image, ImageDraw, ImageFont
import io

def create_summary_image(data_dict, title="Data Summary"):
    """Create a visual summary of key metrics"""
    
    # Create blank image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        text_font = ImageFont.truetype("arial.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 20), title, fill='black', font=title_font)
    
    # Draw data
    y_position = 60
    for key, value in data_dict.items():
        text = f"{key}: {value}"
        draw.text((20, y_position), text, fill='blue', font=text_font)
        y_position += 25
    
    return img

# Create summary of key metrics
key_metrics = {
    'Total Properties': f"{len(airbnb_df):,}",
    'Average Price': f"${airbnb_df['price'].mean():.0f}",
    'Most Common Room Type': airbnb_df['room_type'].mode()[0],
    'Top Borough': airbnb_df['neighbourhood_group'].mode()[0]
}

summary_img = create_summary_image(key_metrics, "NYC Airbnb Summary")
summary_img.show()
print("📸 Created visual summary image")
```

### 4.2 Processing Chart Images

**🔄 Image Enhancement:** Basic image processing for better presentations.

```python
# Save a matplotlib chart and process it with PIL
fig, ax = plt.subplots(figsize=(10, 6))

# Create a chart
borough_avg = airbnb_df.groupby('neighbourhood_group')['price'].mean()
bars = ax.bar(borough_avg.index, borough_avg.values, color='lightblue')
ax.set_title('Average Price by Borough', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Price ($)')

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Load with PIL for processing
chart_image = Image.open(buffer)

# Basic image info
print(f"📊 Chart image size: {chart_image.size}")
print(f"📊 Chart image mode: {chart_image.mode}")

# Create a thumbnail version
thumbnail = chart_image.copy()
thumbnail.thumbnail((300, 200))
print(f"📊 Thumbnail size: {thumbnail.size}")

plt.close()  # Clean up the plot
```

### 4.3 Combining Multiple Visualizations

**🖼️ Creating Visual Reports:**

```python
def create_visual_report():
    """Create a combined visual report"""
    
    # Create multiple small charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Chart 1: Price distribution
    axes[0].hist(airbnb_df['price'], bins=20, alpha=0.7, color='skyblue')
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price ($)')
    
    # Chart 2: Borough comparison
    borough_counts = airbnb_df['neighbourhood_group'].value_counts()
    axes[1].bar(range(len(borough_counts)), borough_counts.values, color='lightcoral')
    axes[1].set_title('Properties by Borough')
    axes[1].set_xticks(range(len(borough_counts)))
    axes[1].set_xticklabels(borough_counts.index, rotation=45)
    
    # Chart 3: Room type pie chart
    room_counts = airbnb_df['room_type'].value_counts()
    axes[2].pie(room_counts.values, labels=room_counts.index, autopct='%1.1f%%')
    axes[2].set_title('Room Types')
    
    plt.tight_layout()
    
    # Save the combined chart
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    
    # Load as PIL image
    report_image = Image.open(buffer)
    plt.close()
    
    return report_image

# Create the visual report
visual_report = create_visual_report()
print(f"📊 Created visual report: {visual_report.size[0]}x{visual_report.size[1]} pixels")
```

---

## Step 5: 🚀 Independent Challenge - Interactive Dashboard Creation

**Your Mission:** Build a comprehensive visualization dashboard that tells the complete story of the NYC Airbnb market.

### 🎯 Challenge: Market Analysis Dashboard

**Business Scenario:** You're presenting to investors who need to understand the market opportunity in 5 minutes. Create visualizations that tell the complete story.

**Your Dashboard Must Include:**

**1. Market Overview Section**
- Overall market size and structure
- Price distribution and key statistics
- Geographic market concentration

**2. Competitive Analysis Section**
- Room type performance comparison
- Borough-by-borough competitive landscape
- Price positioning analysis

**3. Opportunity Identification Section**
- Underserved market segments
- Price-performance relationships
- Investment opportunity zones

### 🛠️ Your Visualization Toolkit:

```python
# Starter framework - expand this
def create_market_dashboard():
    """Create comprehensive market dashboard"""
    
    # Set up the dashboard layout
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout for multiple charts
    # Top row: 3 charts
    # Middle row: 2 charts  
    # Bottom row: 1 large chart
    
    # Your implementation here
    pass

def save_dashboard_as_report():
    """Save dashboard as professional image report"""
    
    # Create your visualization
    # Save as high-quality image
    # Add PIL enhancements if needed
    
    pass
```

### 📊 Advanced Techniques to Try:

**Visualization Enhancements:**
- Custom color schemes that reflect business insights
- Annotations that highlight key findings
- Multiple chart types in coordinated layouts
- Professional styling for stakeholder presentations

**PIL Integration:**
- Create summary cards with key metrics
- Combine multiple charts into report layouts
- Add logos or branding to visualizations
- Generate thumbnail summaries for different audiences

### 🏆 Success Criteria:

- Dashboard tells a clear business story
- Charts are professionally formatted and readable
- Key insights are highlighted visually
- Report could be presented to executives
- Code is organized and reusable

### 💡 Business Questions to Answer:

1. **Market Size:** How big is this opportunity?
2. **Competition:** Where is competition strongest/weakest?
3. **Pricing:** What pricing strategies are most effective?
4. **Geography:** Which areas offer best opportunities?
5. **Segments:** What customer segments are underserved?

---

## Step 6: What You've Mastered

**🎉 Outstanding Work!** You've developed essential visualization and image processing skills for business analysis.

### ✅ **Visualization Mastery:**
- Create professional charts that communicate business insights
- Choose appropriate chart types for different business questions
- Design comprehensive dashboards for stakeholder presentations
- Apply custom styling for professional appearance
- Combine multiple visualizations for complete analysis

### ✅ **Technical Skills:**
- Matplotlib fundamentals for business visualization
- PIL basics for image processing and enhancement
- Integration of visual and data analysis workflows
- Professional chart formatting and presentation

### 🌟 **Business Applications:**
- **Executive dashboards** for strategic decision-making
- **Market analysis** presentations for stakeholders
- **Performance monitoring** through visual KPIs
- **Opportunity identification** through pattern visualization
- **Report generation** with combined visual and text elements

### 🎯 **Next Steps:**
- Explore interactive visualization libraries (Plotly, Bokeh)
- Develop automated dashboard generation
- Integrate visualizations into business reporting systems
- Advanced image processing for specialized applications

---

**📊 Congratulations on mastering business visualization!** You now have the skills to transform data insights into compelling visual stories that drive business decisions. These visualization abilities will make your analysis more impactful and your recommendations more persuasive throughout your career.

**The visual storytelling toolkit is complete. Time to make data sing!** 🚀📈✨
# Week 3 – Exploratory Data Analysis and Visualization in Logistics

## 📌 Overview

Week 3 of the logistics data analysis internship focused on Exploratory Data Analysis and Data Visualization using Python.

After defining the logistics scenario and KPIs in Week 1 and completing data cleaning and preprocessing in Week 2 this weeks work focuses on understanding the prepared dataset through analysis and visualizations.

The main objective was to identify patterns related to delivery performance transportation costs order volume, performance and relationships between important logistics variables.

---

## 🎯 Objectives

The main objectives of Week 3 were:

- Perform Exploratory Data Analysis on logistics data.

- Calculate descriptive statistics.

- Analyze delivery-time and freight-cost distributions.

- Identify outliers and unusual observations.

- Analyze order-volume trends over time.

- Compare logistics performance across states.

- Study relationships between delivery time, freight cost and order value.

- Use visualizations to identify logistics bottlenecks.

- Generate meaningful business insights and recommendations.

---

## 📊 Dataset

The project uses the E-Commerce Public Dataset by Olist.

The dataset contains information related to:

- Customer orders

- Sellers

- Products

- Order dates

- Delivery dates

- Estimated delivery dates

- Freight costs

- Order values

- Customer locations

- Order status

The dataset was selected because it provides information that can be used to study e-commerce logistics and last-mile delivery performance.

---

## 🛠️ Technologies Used

- Python

- Pandas – Data manipulation and analysis

- NumPy – Numerical operations

- Matplotlib – Data visualization

- Seaborn – Statistical visualization

- Jupyter Notebook / Python

---

## 🔍 EDA Performed

The following analyses were performed:

### 1. Descriptive Statistics

Calculated:

- Mean

- Median

- Standard deviation

- Minimum

- Maximum

- Quartiles

for variables such as:

- Order value

- Freight value

- Delivery time

- Delay days

### 2. Delivery Time Analysis

A histogram and box plot were used to understand:

- delivery duration

- Distribution of delivery times

- Variation in delivery performance

- Potential extreme delivery times

### 3. Freight Cost Analysis

The distribution of freight costs was analyzed to understand:

- Typical transportation costs

- Cost variation

- High-cost shipments

- cost-related anomalies

### 4. Time-Series Analysis

Monthly order volume and average delivery time were analyzed to identify:

- Changes in demand

- High-volume periods

- Changes in delivery performance

- capacity-related issues

### 5. Geographic Analysis

Customer states were compared based on:

- Number of orders

- Late-delivery rate

This helps identify regions with demand and regions where delivery performance may require improvement.

### 6. Relationship Analysis

Scatter plots were created to investigate relationships between:

- Freight cost and delivery time

- Order value and freight cost

### 7. Correlation Analysis

A correlation heatmap was created to understand relationships among logistics variables.

---

## 📈 Visualizations

The project includes the following visualizations:

- Delivery Time Histogram

- Freight Cost Histogram

- Delivery Time Box Plot

- Monthly Order Volume Line Chart

- Average Delivery Time Line Chart

- Top States by Order Volume

- Late Delivery Rate by State

- Freight Cost vs Delivery Time Scatter Plot

- Order Value vs Freight Cost Scatter Plot

- Logistics Variable Correlation Heatmap

Each visualization was selected based on the type of logistics question being investigated.

---

## 💡 Key Insights

The EDA provides useful areas for investigation:

- Delivery times may show a long tail of unusually delayed shipments.

- Freight costs can vary considerably between shipments.

- Monthly order volume can reveal periods of increased pressure.

- Some regions may have late-delivery rates than others.

- High freight costs may be associated with certain shipment characteristics.

- Correlation analysis can help identify useful variables for future predictive models.

- Large variation in delivery time can indicate inconsistent logistics performance.

---

## 🚚 Logistics Bottlenecks Identified

Potential bottlenecks that can be investigated further include:

- Regions with high late-delivery rates

- High-demand periods

- Unusually expensive shipments

- Extremely long delivery times

- Differences in performance, between geographic regions

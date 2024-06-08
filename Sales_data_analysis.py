#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


#load dataset
df=pd.read_excel(r"C:\Users\jithu\Downloads\ECOMM DATA.xlsx")


# In[4]:


df.head()


# In[5]:


df.shape


# In[7]:


#summary of data set
df.info()


# In[8]:


#Check for missing value
df.isnull().sum()


# Total sales
# 

# In[9]:


total_sales = df['Sales'].sum()
print(f"Total sales: {total_sales}")


# Sales trends over time

# In[11]:


#grouping  sales by month and year
df['Month'] = pd.to_datetime(df['Ship Date']).dt.month
df['Year'] = pd.to_datetime(df['Ship Date']).dt.year

grouped_sales_year_month = df.groupby(['Year', 'Month'])['Sales'].sum()
print(grouped_sales_year_month)


# In[12]:


#Sales tends over time by year and month
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
grouped_sales_year_month.plot(kind='bar')
plt.xlabel('Year-month')
plt.ylabel('Sales')
plt.title('Sales Trends by Year and month')
plt.xticks(rotation=90)
plt.show()


# Best selling product

# In[13]:


#Top 10 best-selling products by calculating the sum of their sales
best_selling_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)

# Print the top 10 best-selling products
print(best_selling_products)


# In[14]:


#Bar chart to visualize the top 10 best-selling products
plt.figure(figsize=(12, 6))
best_selling_products.plot(kind='bar')
plt.xlabel('Product Name')
plt.ylabel('Total Sales')
plt.title('Top 10 Best-Selling Products')
plt.xticks(rotation=45)
plt.show()


# In[15]:


#Pie chart to visualize the contribution of each product to total sales
plt.figure(figsize=(12, 6))
plt.pie(best_selling_products, labels=best_selling_products.index, autopct="%1.1f%%")
plt.title('Contribution of Top 10 Best-Selling Products to Total Sales')
plt.show()


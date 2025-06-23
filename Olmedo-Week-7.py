import numpy as np
import matplotlib.pyplot as plt

sales_data = np.array ([
    [150, 200, 250, 300, 350, 400],
    [180, 220, 280, 310, 370, 430],
    [200, 240, 290, 320, 380, 450],
    [170, 210, 260, 310, 340, 400]
])

#product2_first3_months = sales_data[1, :3]
#print("Product 2 sales in the first three months: ", product2_first3_months)

#last3_months = sales_data[:, -3:]
#print("Sales data for all products in the last 3 months: \n", last3_months)

#growth = sales_data[:, -1] - sales_data[:, 0]
#print(growth)

adjustment = np.array([1, 1, 1, 1, 1.05, 1.05])
adjusted_sales = sales_data * adjustment
#print(adjusted_sales)

months = np.arange(1, 7)
for i in range (sales_data.shape[0]):
    plt.plot (months, sales_data[i], label = f'Product {i + 1}', marker = 'o')

'''plt.title("Monthly Sales Data for Each Product")
plt.xlabel("Month")
plt.ylabel("Sales (in thousand)")
plt.legend()
plt.grid(True)
plt.show()'''

'''total_sales_per_product = np.sum(sales_data, axis=1)
products = [f'Product {i + 1}' for i in range(sales_data.shape[0])]

plt.bar(products, total_sales_per_product, color='skyblue')
plt.title("Total Sales per Product")
plt.xlabel("Product")
plt.ylabel("Total Sales (in thousand)")
plt.show()

plt.bar(products, growth, color='coral')
plt.title("Growth in Sales from First to Last Month")
plt.xlabel("Product")
plt.ylabel("Sales Growth (in thousands)")
plt.show()'''

for i in range (sales_data.shape[0]):
    plt.plot(months[-3:], sales_data[i, -3:], label=f'Original Product {i + 1}', linestyle='--')
    plt.plot(months[-3:], adjusted_sales[i, -3:], label=f'Adjusted Product {i + 1}', marker='o')

plt.title("Sales Comparison Before and After Seasonal Adjustment")
plt.xlabel("Month")
plt.ylabel("Sales (in thousand)")
plt.legend()
plt.grid(True)
plt.show()


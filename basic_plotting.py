import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import mutual_info_classif
from matplotlib.pyplot import scatter, show, xlabel, ylabel, title, plot
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
full_x,full_y = np.array(data.data),np.array(data.target)
x,y = full_x[:2000,:],full_y[:2000].reshape(2000,1)

model = LinearRegression()
model.fit(x[:,:],y)
y_pred = model.predict(full_x)

title("house price vs size")
scatter(x[:,:1],y)
plot(x[:,:1],y_pred[:2000], color = "red")
xlabel("size m^2 * 100")
ylabel("price *100000")
show()

#print out first 10 y_pred next to y and also print the cost and accuracy.

print("---------FIRST 10 PREDICTIONS---------\n")
for i in range(10):
    print("\nactual: ",float(y[i]),"    predicted: ", float(y_pred[i]))


print("full_y.shape = ", full_y.shape, " y_pred.shape = ", y_pred.shape)
mse = mean_squared_error(full_y,y_pred)
print("\nerror_over_all_examples = ", mse,"\n")





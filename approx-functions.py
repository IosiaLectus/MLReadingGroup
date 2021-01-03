#############################  NECESARY MODULES ##############################
import numpy as np #- Used for mathematical manipulations.
import torch #- Used to make the neural network.
import matplotlib.pyplot as plt #- Used for making figures.
import os #- Used to find directories to  open and save files.
import time #- Used for timing the script.
from sklearn.metrics import r2_score, mean_squared_error #- Used for gauging how \
# good is my predition.
##############################################################################


############################## CUSTOM FUNCTIONS ##############################
def fun(x):
    return np.sin(x)


##############################################################################


################################## START #####################################
t0 = time.time() #- Start time.
num_in = 1 #- Number of inputs.
num_hidden = 120 #- Number of hidden nodes.
num_out = 1 #- Number of outputs

x_min_train = -3 #- Minimum x value used as input.
x_max_train = 3 #- Maximum x value used as input.
num_x_train = 50 #- Number of x values used for input.
x_train = np.linspace(x_min_train, x_max_train, num_x_train)
data_in = torch.tensor(x_train, dtype = torch.float)
data_in = data_in.view([num_x_train, num_in])
out_theory = fun(x_train)
out_theory = torch.tensor(out_theory, dtype = torch.float)
out_theory = out_theory.view([num_x_train, num_out])
##############################################################################


######################## NEURAL NETWORK ARCHITECTURE #########################
model = torch.nn.Sequential(torch.nn.Linear(num_in, num_hidden),\
                            torch.nn.Sigmoid(),\
                            torch.nn.Linear(num_hidden, num_out))
#- The model architecture.
num_epoch = int(2e4) # Maximum number of epochs we are training the network for.
learning_rate = 1e-3 #- The learning rate used in the optimizer.
#- We use the Adam optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
error = 10 #- Initialized the error value so it will enter the 
            # training while-loop.
error_threshold = 1e-8 #- Error threshold that above which the networl keeps 
                       # training 
loss_fn = torch.nn.MSELoss(reduction = 'mean')
##############################################################################


################################# TRAINING ###################################
epoch = 0 #- Current epoch.
print('EPOCH\tMSE')
while error > error_threshold:
    out = model(data_in)
    loss = loss_fn(out, out_theory )
    error = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('%d\t%.2E'% (epoch, loss.item()))
    if epoch >= num_epoch:
        break
    epoch += 1

##############################################################################

################################## FIGURES ###################################
x_min_test = -5
x_max_test = 5
num_x_test = 100
x_test = np.linspace(x_min_test, x_max_test, num_x_test)
data_in = x_test
data_in = torch.tensor(data_in, dtype = torch.float)
data_in = data_in.view([num_x_test, num_in])
out = model(data_in)
out = out.detach().numpy()

fun_real =fun(x_test)
fun_pred = out

plt.figure(0)
plt.scatter(x_test, fun_pred, c = 'orange', s = 6, label = 'Prediction')
plt.scatter(x_test, fun_real, s = 6, label = 'Theory')
plt.axvline(x = x_min_train, c = 'red')
plt.axvline(x = x_max_train, label = 'Training boundaries', c = 'red')
plt.legend()
plt.savefig(os.getcwd() + '/approx-fun.png', dpi = 600)
plt.show()
r2 = r2_score(fun_real, fun_pred)
mse = mean_squared_error(fun_real, fun_pred)
print('R^2\t%.3f' % r2)
print('MSE\t%.2E' % mse)
dt = time.time() - t0
print('Time needed for the script to execute is %.3f seconds!' % dt)

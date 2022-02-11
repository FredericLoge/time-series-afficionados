# UNIVARIATE TIME SERIES ====================================
#
#
#
#

# load libraries
library(keras)
library(tidyverse)
library(lubridate)

# prepare tibble with proper variables and shape
he <- tibble(value = as.numeric(AirPassengers)) %>%
  mutate(time = time(AirPassengers), 
         year = floor(time), 
         month = ((time-floor(time))*12 + 1) %>% round() %>% as.integer() %>% as.character(),
         month_ = case_when(
           nchar(month) == 1 ~ paste0('0', month),
           TRUE ~ month
         ),
         date = as.Date(paste0("01/", month_, "/", year), format = '%d/%m/%Y'))

# nice plot of the time series
gg = ggplot(data = he) +
  aes(x = date, y = value) +
  geom_point() +
  geom_line() 
gg

# prepare the data for a time series treatment
hep = he %>%
  select(date, month, year, value) %>%
  mutate(value_lag1 = value %>% lag(n=1),
         value_lag12 = value %>% lag(n=12),
         is_train = year < 1955)

# AR(p) model as baseline using lm()
ar1 = lm(value ~ value_lag1, data = hep %>% filter(is_train))
ar1_12 = lm(value ~ value_lag12 + value_lag1, data = hep %>% filter(is_train))
hep_ = hep %>%
  mutate(ar1_pred = predict(ar1, hep),
         ar1_12_pred = predict(ar1_12, hep))
ggplot(data=hep_ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=value, col='Series')) +
  geom_line(aes(y=ar1_pred, col='AR(1)')) +
  geom_line(aes(y=ar1_12_pred, col='AR(1,12)'))
ggplot(data=hep_ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=abs(ar1_pred - value), col='AR(1)')) +
  geom_line(aes(y=abs(ar1_12_pred - value), col='AR(1,12)'))

# replicate AR(p) model with keras basic model
model_0 <- keras_model_sequential() %>%
  layer_dense(input_shape = 2, units = 1, activation = 'linear')
model_0 %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.01),
  loss = 'mse'
)
train_ = hep %>% filter(is_train) %>% drop_na(value_lag1, value_lag12)
model_0 %>% fit(x = train_ %>% select(value_lag1, value_lag12) %>% as.matrix(),
                y = train_ %>% select(value) %>% as.matrix(),
                epochs = 100)

print(model_0$weights)
w = c('theta0' = model_0$weights[[2]]$numpy(),
  'theta1' = model_0$weights[[1]]$numpy()[1],
  'theta2' = model_0$weights[[1]]$numpy()[2])
print(w)
print(ar1_12)
model_0_pred_manual <- w[1] + w[2]*hep$value_lag1 + w[3]*hep$value_lag12   
model_0_pred_auto <- model_0 %>% 
  predict(hep %>% select(value_lag1, value_lag12) %>% as.matrix())
sum(abs(model_0_pred_auto - model_0_pred_manual), na.rm = T)

hep__ = hep_ %>% 
  mutate(model_0_pred_auto) 
hep__ %>%
  filter(!is_train) %>%
  summarise(my_stat(ar1_pred - value),
            my_stat(ar1_12_pred - value),
            my_stat(model_0_pred_auto - value)
  )
my_stat = function(e) mean(e^2)
my_stat = function(e) mean(abs(e))
ggplot(hep__ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=value, col='Series')) +
  geom_line(aes(y=ar1_pred, col='AR(1)')) +
  geom_line(aes(y=ar1_12_pred, col='AR(1,12)')) +
  geom_line(aes(y=model_0_pred_auto, col='KERAS'))
ggplot(hep__ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=abs(ar1_pred - value), col='AR(1)')) +
  geom_line(aes(y=abs(ar1_12_pred - value), col='AR(1,12)')) +
  geom_line(aes(y=abs(model_0_pred_auto - value), col='KERAS'))

# improve on previous model however you like
# adding callbacks
model_1 <- keras_model_sequential() %>%
  layer_dense(input_shape = 2, units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')
model_1 %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.01),
  loss = 'mse'
)
# TODO add callbacks
# TODO add validation set
# TODO work on larger data set
train_ = hep %>% filter(is_train) %>% drop_na(value_lag1, value_lag12)
model_1 %>% fit(x = train_ %>% select(value_lag1, value_lag12) %>% as.matrix(),
                y = train_ %>% select(value) %>% as.matrix(),
                epochs = 10)
hep__ = hep_ %>% 
  mutate(model_1_pred_auto = model_1 %>% 
           predict(hep %>% select(value_lag1, value_lag12) %>% as.matrix())) 
ggplot(hep__ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=value, col='Series')) +
  geom_line(aes(y=ar1_pred, col='AR(1)')) +
  geom_line(aes(y=ar1_12_pred, col='AR(1,12)')) +
  geom_line(aes(y=model_1_pred_auto, col='KERAS'))
ggplot(hep__ %>% filter(!is_train)) +
  aes(x = date) +
  geom_line(aes(y=abs(ar1_pred - value), col='AR(1)')) +
  geom_line(aes(y=abs(ar1_12_pred - value), col='AR(1,12)')) +
  geom_line(aes(y=abs(model_1_pred_auto - value), col='KERAS'))
hep__ %>%
  filter(!is_train) %>%
  summarise(my_stat(ar1_pred - value),
            my_stat(ar1_12_pred - value),
            my_stat(model_1_pred_auto - value)
  )
my_stat = function(e) mean(e^2)
my_stat = function(e) mean(abs(e))


# MULTIVARIATE TIME SERIES ====================================
#
#
#
#

# generator function
f1 <- function(x){
  theta1 <- c(3,-3,+6,-6)
  0.00*x + 0.00 * theta1[1 + (x%%4)] + rnorm(n=length(x), sd=0.2)
}

# generate some datapoints
he <- tibble(time_index = 1:1000) %>%
  mutate(seas = 1 + (time_index %% 4)) %>%
  mutate(s1 = time_index %>% f1) %>%
  mutate(s1_ = s1 + lag(s1, n = 10)) %>%
  mutate(s1_lag = lag(s1), 
         s1__lag = lag(s1_))

# plot time series
he %>% pull(s1) %>% plot.ts()
he %>% pull(s1_) %>% plot.ts()

# initiate LSTM model
model_0 <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = 2+1, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'linear')
model_0 %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.01),
  loss = 'mse'
)
model_1 <- keras_model_sequential() %>%
  layer_lstm(stateful = TRUE, units = 32, 
             batch_input_shape = c(1, 2+1, 1), activation = 'tanh') %>%
  layer_dense(units = 2, activation = 'linear')
model_1 %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.01),
  loss = 'mse'
)

mse_ <- function(e) mean(e^2)
rep(0, 1000) %>% f1() %>% mse_()

# TODO callbacks + validation set
# TODO <attention shuffle>
# TODO backtesting (package ?)
# TODO same architecture : wider/more depth ? consequences ?
# TODO learning rate (setting + scheduling)
# TODO architecture

# TODO work on larger data set

train_ = he %>% filter(time_index<700)
x_train = (train_ %>% select(seas, s1_lag, s1__lag) %>% as.matrix())[-(1:15),]
dim(x_train) = c(dim(x_train), 1)
any(is.na(x_train))
y_train = (train_ %>% select(s1, s1_) %>% as.matrix())[-(1:15),]
dim(y_train) = c(dim(y_train), 1)
any(is.na(y_train))

model_1 %>% fit(x = x_train, y = y_train, epochs = 1, batch_size=1, shuffle=FALSE)
model_1 %>% reset_states()

model_0 %>% fit(x = x_train[,,1], y = y_train[,,1], epochs = 1, batch_size=1)

y_pred = predict(model_1, x_train, batch_size = 1)
summary(y_pred)
summary(y_train[,,1])

plot.ts(y_train[1:50,1,1])
lines(y_pred[1:50,1], col='red')

plot(y_train[,1,1], y_pred[,1])
abline(0,1,col='red')
plot(y_train[,2,1], y_pred[,2])
abline(0,1,col='red')

test_ = he %>% filter(time_index>=700)
x_test = (test_ %>% select(s1_lag, s1__lag) %>% as.matrix())[-1,]
dim(x_test) = c(dim(x_test), 1)
y_test = (test_ %>% select(s1, s1_) %>% as.matrix())[-1,]
dim(y_test) = c(dim(y_test), 1)

y_pred = predict(model_1, x_test, batch_size = 1)
summary(y_pred)
summary(y_test[,,1])


library('ISLR') # набор данных
library('leaps') # функция regsubset() — отбор оптимального подмножества переменных
library('glmnet') # функция glmnet() — лассо
library('pls') # регрессия на главные компоненты — pcr() и частный МНК — plsr()

my.seed <- 1
names(Carseats)
# Пошаговое исключение----------------
regfit.bwd <- regsubsets(Sales ~ ., data = Carseats,
                         nvmax = 10, method = 'backward')
summary(regfit.bwd)
round(coef(regfit.bwd, 10), 3)

#k-кратная кросс-валидация----------------------
# отбираем 10 блоков наблюдений
k <- 10
set.seed(my.seed)
folds <- sample(1:k, nrow(Carseats), replace = T)
# заготовка под матрицу с ошибками
cv.errors <- matrix(NA, k, 10, dimnames = list(NULL, paste(1:10)))
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}
# заполняем матрицу в цикле по блокам данных
for (j in 1:k){
  best.fit <- regsubsets(Sales ~ ., data = Carseats[folds != j, ],
                         nvmax = 10)
  # теперь цикл по количеству объясняющих переменных
  for (i in 1:10){
    # модельные значения Sales
    pred <- predict(best.fit, Carseats[folds == j, ], id = i)
    # вписываем ошибку в матрицу
    cv.errors[j, i] <- mean((Carseats$Sales[folds == j] - pred)^2)
  }
}
# усредняем матрицу по каждому столбцу (т.е. по блокам наблюдений), 
#  чтобы получить оценку MSE для каждой модели с фиксированным 
#  количеством объясняющих переменных
mean.cv.errors <- apply(cv.errors, 2, mean)
round(mean.cv.errors, 2)
# на графике
plot(mean.cv.errors, type = 'b')
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)],
       col = 'red', pch = 20, cex = 2)
# перестраиваем модель с 7 объясняющими переменными на всём наборе данных
reg.best <- regsubsets(Sales ~ ., data = Carseats, nvmax = 10)
round(coef(reg.best, 7), 3)

#MSE модели на тестовой выборке--------------------------
set.seed(my.seed)
train <- sample(c(T, F), nrow(Carseats), rep = T)
test <- !train
# обучаем модели
regfit.best <- regsubsets(Sales ~ ., data = Carseats[train, ],
                          nvmax = 10)
# матрица объясняющих переменных модели для тестовой выборки
test.mat <- model.matrix(Sales ~ ., data = Carseats[test, ])
# вектор ошибок
val.errors <- rep(NA, 10)
# цикл по количеству предикторов
for (i in 1:10){
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[, names(coefi)] %*% coefi
  # записываем значение MSE на тестовой выборке в вектор
  val.errors[i] <- mean((Carseats$Sales[test] - pred)^2)
}
round(val.errors, 2)


#Регрессия на главные компоненты
# из-за синтаксиса glmnet() формируем явно матрицу объясняющих...
x <- model.matrix(Sales ~ ., Carseats)[, -1]
# и вектор значений зависимой переменной
y <- Carseats$Sales
set.seed(my.seed)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]
# кросс-валидация 
set.seed(2)   # непонятно почему они сменили зерно; похоже, опечатка
pcr.fit <- pcr(Sales ~ ., data = Carseats, scale = T, validation = 'CV')
summary(pcr.fit)
# график ошибок
validationplot(pcr.fit, val.type = 'MSEP')

#Подбор оптиального M: кросс-валидация на обучающей выборке
set.seed(my.seed)
pcr.fit <- pcr(Sales ~ ., data = Carseats, subset = train, scale = T,
               validation = 'CV')
validationplot(pcr.fit, val.type = 'MSEP')
# MSE на тестовой выборке
pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 7)
round(mean((pcr.pred - y.test)^2), 0)
# подгоняем модель на всей выборке для M = 7 
#  (оптимально по методу перекрёстной проверки)
pcr.fit <- pcr(y ~ x, scale = T, ncomp = 7)
summary(pcr.fit)

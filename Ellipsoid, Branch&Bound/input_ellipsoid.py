import numpy as np
from ellipsoid import find_feasible

n, m = map(int, input().split())
c = np.asarray(input().split(), dtype=np.float64)
A1 = []
for _ in range(m):
  A1.append(input().split())
A1 = np.asarray(A1, dtype=np.float64)
b1 = np.asarray(input().split(), dtype=np.float64)

# 2 2
# 2 3
# 3 6
# 3 1
# 24 9

# n,m = 2, 2
# c = np.array([2, 3], dtype=np.float64)
# A1 = np.array( [[3, 6], 
#               [3, 1]], dtype=np.float64)
# b1 = np.array([24, 9], dtype=np.float64)


# 2 3
# 5 7
# -2 4
# 1 -7
# 2 3
# -4 -7 10

# n,m = 2, 3
# c = np.array([5, 7], dtype=np.float64)
# A1 = np.array([[-2, 4], 
#               [1, -7],
#               [2, 3]], dtype=np.float64)
# b1 = np.array([-4, -7, 10], dtype=np.float64)

# n,m = 2, 3
# c = np.array([-3, -4], dtype=np.float64)
# A1 = np.array([[-1, -2], 
#               [-1, 1],
#               [3, -1]], dtype=np.float64)
# b1 = np.array([-14, -2, 0], dtype=np.float64)

# n,m = 2, 2
# c = np.array([2, 3], dtype=np.float64)
# A1 = np.array([[3, 6], 
#               [3, 1]], dtype=np.float64)
# b1 = np.array([24, 9], dtype=np.float64)

# 2 2
# -1 1
# -3 -5
# 1 2
# -3 2

# n,m = 2, 2
# c = np.array([-1, 1], dtype=np.float64)
# A1 = np.array([[-3, -5], 
#               [1, 2]], dtype=np.float64)
# b1 = np.array([-3, 2], dtype=np.float64)

# n,m = 2, 3
# c = np.array([1, 1], dtype=np.float64)
# A1 = np.array([[-1, -1], 
#               [2, 1],
#               [1, 2]], dtype=np.float64)
# b1 = np.array([-5, 3, 2], dtype=np.float64)

# A2 = np.eye(n)
# b2 = np.zeros(n)

# A = np.vstack([A1, A2])
# b = np.hstack([b1, b2])
# print('A: ', A)
# print('b: ', b)

A2 = np.eye(n)
b2 = np.zeros(n)

A = np.vstack([A1, A2])
b = np.hstack([b1, b2])

U = np.max([np.max(np.abs(A)), np.max(np.abs(b))])
v = (n**(-n))*((n*U)**(-(n**2)*(n+1)))
V = ((2*n)**(n))*((n*U)**(n**2))

s = np.zeros(n)
D = (n*((n*U)**(2*n)))*np.eye(n)

tstar = np.ceil((2*(n + 1))*np.log(V/v))

s, D, infeasible, t = find_feasible(tstar, n, s, D, A, b)

prevPt = s
tstar -= t
if infeasible:
  print('Infeasible')
  quit()
  
while not infeasible:

  A3 =  np.vstack([A, -c])
  b3 = np.hstack([b, 0])

  b3[-1] = np.inner(-c, s) + 0.000000001
  #print(A3)
  #print(b3)
  s, D, infeasible, t = find_feasible(tstar, n, s, D, A3, b3)
  tstar -= t
  # print("-----------",tstar)
  if not infeasible:
    prevPt = s

print(np.dot(c, prevPt))
for i in range(n):  
  print(prevPt[i], end=' ')


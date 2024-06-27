import numpy as np

def center_in_poly(st, A, b):

  k = A@st >= b
  if (k).all():
    return (True, None)
  else:
    idx = np.where(k == False)[0][0]
    return (False, idx)
  
def new_ellipse(n, st, Dt, a):

  snew = st + (1/(n+1))*(np.matmul(a, Dt)/np.sqrt(np.inner(np.matmul(a, Dt), a)))
  Dnew = (n**2/(n**2 - 1))*( Dt - (2/(n+1))*( np.matmul(np.matmul(a, Dt).reshape(-1,1), np.matmul(Dt, a).reshape(1,-1)) / np.inner(np.matmul(a, Dt), a) ) )

  return snew, Dnew

def find_feasible(tstar, n, s, D, A, b):

  infeasible = False
  t = 0
  for t in range(int(tstar)):
    #print('s:, ', s)
    #print('D:, ', D)
    y_n, idx = center_in_poly(s, A, b)
    if y_n:
      #print('y_n: ', y_n)
      break
    # print('y_n: ', y_n)
    # print('idx: ', idx)
    a = A[idx]
    s, D = new_ellipse(n, s, D, a)
    if t == int(tstar) -1:
      infeasible = True

  return s, D, infeasible, t


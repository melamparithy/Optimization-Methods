import numpy as np

def standardize(n, u, v, A, b, c):
  
  c = np.append(c, np.zeros(u + v, dtype=np.int64))
  I = np.eye(u+v, dtype=np.float64)                     # unit corrresponding to every inequation
  idx = np.arange(n + u + 2*v)
  b_idx = list(np.arange(n, n + u))

  if v != 0:
    I[:, -v:] *= -1                                   # -1 coeffficient for > equation
    I = np.concatenate((I, -1*I[:, -v:]), axis=1)     # +1 coefficient for artificaial variables corresponding to every > equation    
    b_idx.extend(np.arange(n + u + v, n + u + 2*v))

  n_idx = list(set(idx) - set(b_idx))
  A = np.concatenate((A, I), axis=1)

  return A, b, c, b_idx, n_idx

def simplex(A, b, c, b_idx, n_idx):

  cycling = False
  prev_bases = []

  while True:
    Ab = A[:, b_idx]
    An = A[:, n_idx]
    cb = c[b_idx]
    cn = c[n_idx]
    Ab_inv = np.linalg.inv(Ab)
    xb = Ab_inv @ b 
    
    y = Ab_inv @ A
    #print(cb.shape, y.shape)
    Z_C = (cb @ y) - c # Z_k - C_k
    Z_C = np.round(Z_C, 11)
    #print('Z_k - C_k:', Z_C)
    if (Z_C <= 0.).all():
      rhs = cb @ xb
      x = np.zeros(A.shape[1], dtype= np.float64)
      x[b_idx] = xb
      #print(rhs)
      #print(x)
      break

    if cycling:
      enter_v = np.where(Z_C > 0)[0][0]
    else:
      enter_v = np.argmax(Z_C)      
  
    if (y[:, enter_v] <= 0).all():
      rhs = np.inf
      print('Unbounded')
      #break     # return ??
      return y, None, b_idx, n_idx, xb, None

    # min ratio test ~ xb / y[:, enter_v]
    xb[y[:, enter_v] <= 0] = np.inf
    ratio = xb / y[:, enter_v]
    neg_vals = (ratio < 0).sum()    # number of negative ratios
    min_idx = np.argsort(ratio)[neg_vals]   # selecting smallest positive ratio
    leave_v = b_idx[min_idx]      

    i_e = np.where(n_idx == enter_v)[0][0]
    i_l = np.where(b_idx == leave_v)[0][0]
    b_idx[i_l] = enter_v
    n_idx[i_e] = leave_v 

    if not cycling:
      if list(b_idx) in prev_bases:
        print('Cycling detected')
        cycling = True

    prev_bases.append(list(b_idx)) 

  return y, rhs, b_idx, n_idx, xb, x

def check_infeasibility(n, u, v, A, b, b_idx, n_idx):

  idx = np.arange(n + u + 2*v)
  art_idx = idx[-v:]

  c_phase1 = np.zeros(n + u + 2*v)
  c_phase1[-v:] = 1

  newA, rhs, b_idx, n_idx, xb, x = simplex(A, b, c_phase1, b_idx, n_idx)

  if np.in1d(art_idx, n_idx).sum() == v:

    n_idx = list(np.setdiff1d(n_idx, art_idx))
    newA = np.delete(newA, art_idx, axis=1)

    return newA, None, b_idx, n_idx, xb, None
    #_ = simplex(newA, xb, c, b_idx, n_idx)

  else:

    if (xb > 0).all():
      print('Infeasible')
    else:
      print('Infeasible')

    return [None]*6
  
def twoPhase(n, u, v, A, b, c, b_idx, n_idx):

  idx = np.arange(n + u + 2*v)
  
  # A, b, c, b_idx, n_idx = standardize(n, u, v, A, b, c)

  if v == 0:
    # jumping to phase 2 if no artificial variables
    #print('direct to phase 2')
    y, rhs, b_idx, n_idx, xb, x = simplex(A, b, c, b_idx, n_idx)
    # return y, rhs, b_idx, n_idx, xb, x[:n]
    if rhs is not None:
      return y, rhs, b_idx, n_idx, xb, x[:n]
    else:
      return y, rhs, b_idx, n_idx, xb, x

  # phase 1
  newA, _, b_idx, n_idx, xb, _ = check_infeasibility(n, u, v, A, b, b_idx, n_idx)
  
  # phase 2
  if newA is not None:
    y, rhs, b_idx, n_idx, xb, x = simplex(newA, xb, c, b_idx, n_idx)
    if rhs is not None:
      return y, rhs, b_idx, n_idx, xb, x[:n]
    else:
      return y, rhs, b_idx, n_idx, xb, x
  else:
    return [None]*6
  
def find_eq(xb):

  f_idx = None
  for i in range(len(xb)):
    if not xb[i].is_integer():
      return i

  return -1
  
def gomory(n, u, v, A, b, c):

  A1, b1, c1, b_idx, n_idx = standardize(n, u, v, A, b, c)
  #y, rhs, b_idx, n_idx, xb, x = simplex(A1, b1, c1, b_idx, n_idx)
  y, rhs, b_idx, n_idx, xb, x = twoPhase(n, u, v, A1, b1, c1, b_idx, n_idx)

  vars = np.concatenate((np.eye(n), -A[:u]), axis=0)
  vars = np.concatenate((vars, A[u:]), axis=0)
  vals = np.concatenate((np.zeros(n), -b[:u]), axis=0)
  vals = np.concatenate((vals, b[u:]), axis=0)

  rhs, x = None, None
  while True:
    # find xb with fractional value
    eq_idx = find_eq(np.round(xb, 10))
    if eq_idx == -1:
      break
    eq = np.round(y, 10)[eq_idx]

    # create cut
    eq_coeff = np.floor(eq)
    eq_rhs = np.floor(xb[eq_idx])

    gcut_lhs = np.dot(eq_coeff, vars)
    gcut_rhs = np.dot(eq_coeff, vals) + eq_rhs

    # add new constraint
    A = np.insert(A, u, gcut_lhs, axis=0)
    b = np.insert(b, u, gcut_rhs)
    u = u + 1
    #print('A, b: ', A, b, sep='\n')

    A2, b2, c2, b_idx, n_idx = standardize(n, u, v, A, b, c)
    y, rhs, b_idx, n_idx, xb, x = twoPhase(n, u, v, A2, b2, c2, b_idx, n_idx)
    #print('xb', xb)

    n_s = np.expand_dims(-A2[u-1][:n], axis=0)
    vars = np.concatenate((vars, n_s), axis=0)
    vals = np.concatenate((vals, [-b[u-1]]), axis=0)

  if rhs is not None:
    return y, rhs, b_idx, n_idx, xb, x[:n]
  else:
    return y, rhs, b_idx, n_idx, xb, x
import numpy as np

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
    # print(cb.shape, y.shape)
    Z_C = (cb @ y) - c # Z_k - C_k
    Z_C = np.round(Z_C, 11)
    # print('Z_k - C_k:', Z_C)
    if (Z_C <= 0.).all():
      rhs = cb @ xb
      x = np.zeros(A.shape[1], dtype= np.float64)
      x[b_idx] = xb
      # print('b_idx:', b_idx)
      # print('rhs:', rhs)
      # print('x', x)
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

    # print('b_idx:', b_idx)

    if not cycling:
      if list(b_idx) in prev_bases:
        # print('Cycling detected')
        cycling = True

    prev_bases.append(list(b_idx)) 

  return y, rhs, b_idx, n_idx, xb,x, # x[:n]

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

def rem_art_vars(n, u, v, newA, b, b_idx, n_idx, art_idx):

  rem_var = np.intersect1d(art_idx, b_idx)
  while rem_var.size != 0:
    rem_var = np.intersect1d(art_idx, b_idx)
    non_art_nb = np.setdiff1d(n_idx, art_idx)
    # print('artificial variables in basis: ', rem_var)
    # print('non artificial, non basic varaibles: ', non_art_nb)

    art_b_idx = [np.where(rem_var[i] == b_idx)[0][0] for i in range(len(rem_var))]
    R2 = newA[art_b_idx, :][:, non_art_nb]

    if (R2 == 0).all():
      # print('Stop Pivot')
      break

    # print('b_idx: ', b_idx)
    # print('n_idx: ', n_idx)

    e_v = non_art_nb[R2[0] != 0][0]
    l_v = rem_var[0]

    # print('enter_v: ', e_v)
    # print('leave_v: ', l_v)

    i_e = np.where(n_idx == e_v)[0][0]
    i_l = np.where(b_idx == l_v)[0][0]

    b_idx[i_l] = e_v
    n_idx[i_e] = l_v

    # print('b_idx: ', b_idx)
    # print('n_idx: ', n_idx)

    newAb = newA[:, b_idx]
    newAn = newA[:, n_idx]

    newAb_inv = np.linalg.inv(newAb)
    newA = newAb_inv @ newA

  return newA, b_idx, n_idx

def check_infeasibility(n, u, v, A, b, b_idx, n_idx):

  idx = np.arange(n + u + 2*v)
  art_idx = idx[-v:]

  c_phase1 = np.zeros(n + u + 2*v)
  c_phase1[-v:] = 1
  # print(c_phase1)

  newA, rhs, b_idx, n_idx, xb, x = simplex(A, b, c_phase1, b_idx, n_idx)

  if np.in1d(art_idx, n_idx).sum() == v:

    n_idx = list(np.setdiff1d(n_idx, art_idx))
    newA = np.delete(newA, art_idx, axis=1)

    return newA, None, b_idx, n_idx, xb, None
    #_ = simplex(newA, xb, c, b_idx, n_idx)

  else:

    if (x[art_idx] > 0).any():

      print('Infeasible')
      return [None]*6
    else:
      # print('Infeasible2')
      # print('x:', x)
      # print('x_art:', x[art_idx])
      # print('xb:', xb)
      newA, b_idx, n_idx = rem_art_vars(n, u, v, newA, b, b_idx, n_idx, art_idx)
      n_idx = list(np.setdiff1d(n_idx, art_idx))
      newA = np.delete(newA, art_idx, axis=1)
      #return [None]*6
      return newA, rhs, b_idx, n_idx, xb, x     #, art_idx
    
def twoPhase(n, u, v, A, b, c, b_idx, n_idx):

  idx = np.arange(n + u + 2*v)
  
  # A, b, c, b_idx, n_idx = standardize(n, u, v, A, b, c)

  if v == 0:
    # jumping to phase 2 if no artificial variables
    #print('direct to phase 2')
    y, rhs, b_idx, n_idx, xb, x = simplex(A, b, c, b_idx, n_idx)
    return y, rhs, b_idx, n_idx, xb, x

  # phase 1
  '''
  art_idx = idx[-v:]
  c_phase1 = np.zeros(n + u + 2*v)
  c_phase1[-v:] = 1
  newA, rhs, b_idx, n_idx, xb = simplex(A, b, c_phase1, b_idx, n_idx)
  print(newA)

  if np.in1d(art_idx, n_idx).sum() == v:

    n_idx = list(np.setdiff1d(n_idx, art_idx))
    newA = np.delete(newA, art_idx, axis=1)

    # phase 2

    _ = simplex(newA, xb, c, b_idx, n_idx)

  else:

    if (xb > 0).all():
      print('Infeasible')
    else:
      pass
  '''
  newA, _, b_idx, n_idx, xb, _ = check_infeasibility(n, u, v, A, b, b_idx, n_idx)
  
  # phase 2

  if newA is not None:
    y, rhs, b_idx, n_idx, xb, x = simplex(newA, xb, c, b_idx, n_idx)
    return y, rhs, b_idx, n_idx, xb, x
  else:
    return [None]*6

def bBound(num, C):

  m = int((num-1)*num/2)            # num edges        
  num_subsets = np.power(2, num, dtype=np.int8) - 2

  c = np.hstack([np.asarray([C[i,j] for j in range(i)]) for i in range(num)])

  
  e = np.vstack([np.asarray([(i+1,j+1) for j in range(i)]).reshape(-1,2,) for i in range(num)]).reshape(-1,2)

  A1 = np.array([[np.array([[1 if z in e[i] else 0] for i in range(len(e))]).reshape(-1,)] for z in range(1, num+1)]).reshape(-1, int(m))
  b1 = np.ones(num, dtype=np.float64) + 1

  A2 = np.eye(m)
  b2 = np.ones(m)

  A = np.vstack([A1, A2, A1])
  b = np.hstack([b1, b2, b1])

  n = m
  u = num + m
  v = num

  A, b, c, b_idx, n_idx = standardize(n, u, v, A, b, c)
  out = twoPhase(n, u, v, A, b, c, b_idx, n_idx)
  k = out[-1][:n]
  soln = np.zeros((num, num), dtype=np.int64)
  for z in range(m):
    i = int(e[z][0]) - 1
    j = int(e[z][1]) - 1

    soln[i, j] = k[z]
    soln[j, i] = k[z]

  print(out[1])
  # print(soln)
  for i in range(len(soln)):
    for j in range(len(soln[i])):
      print(soln[i,j], end=' ')
    print('')

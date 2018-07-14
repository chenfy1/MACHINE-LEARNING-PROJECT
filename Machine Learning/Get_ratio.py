
# coding: utf-8

# In[ ]:
import math
import time
import random
import scipy as spy
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

def Get_ratio_runtime(X_profit_more,y_profit_more,option,f1):
    n=X_profit_more.shape[0]
    rankA=np.linalg.matrix_rank(X_profit_more.values)
    start_total = time.clock()
    Guassian1_time=[]
    ratio_1=[]
    
    if option==0:
        Project_S=np.random.normal(0,1,(n,n))
    if option==1:
        Project_S=np.random.random(size=(n,n))-0.5
        for i in range(0,n):
            S_norm=np.linalg.norm(Project_S[i])
            Project_S[i]=Project_S[i]/S_norm*math.sqrt(n)
    if option==2:
        Project_S=np.random.randint(0,2,size=[n,n])  
        Project_S[np.where(Project_S==0)]=Project_S[np.where(Project_S==0)]-1
    if option==3:
        density=0.2
        matrixformat='coo'
        B=spy.sparse.rand(n,n,density=density,format=matrixformat,dtype=None)
        Project_S=B.todense()
        Project_S[np.where(Project_S>=0.5)]=1
        Project_S[np.where((Project_S>0)&(Project_S<1))]=-1
        Project_S=np.array(Project_S)
    if option==4:
        import HD
        Project_S=HD.HD(12)
        H_index=np.array(random.sample(range(7240), 4096)) 
        X_profit_more=X_profit_more.iloc[H_index]
        y_profit_more=y_profit_more[H_index]
    
    a_range=np.arange(0,3.3,0.11)
    for a in a_range:
        start = time.clock()
        m=int(4*a*rankA)+1
        
        S=np.array(random.sample(Project_S, m))
    
        SX=np.dot(S,X_profit_more)
        Sy=np.dot(S,y_profit_more)
        #clf_s = linear_model.Lasso(alpha=0.1,tol=10**(-200))
        #clf_s.fit(SX,Sy)
    
        #n = SX.shape[0]
        #m = SX.shape[1]
        A = SX
        b = Sy
        # gamma must be positive due to DCP rules.
        gamma = 0.2

        # Construct the problem.
        x = Variable(65)
        error = sum_squares(A*x - b)
        obj = Minimize(error + gamma*norm(x, 1))
        prob = Problem(obj)
        prob.solve(solver=OSQP)
    
        end = time.clock()
    
        Guassian1_time.append(end-start)
        f2=np.linalg.norm(np.dot(X_profit_more,x.value)-y_profit_more)**2
        #f2=np.linalg.norm(clf_s.predict(X_profit_more)-y_profit_more)**2
        ratio=f2/f1
        ratio_1.append(ratio)
    end_total = time.clock()
    print(end_total-start_total)
    print np.mean(Project_S)
    print(np.var(Project_S))  
    
    plt.figure(figsize=(8,5),dpi=1200)
    plt.plot(a_range,ratio_1,'ro-')  
    plt.title('LASSO d=65')  
    plt.xlabel('Control parameter' r'$\quad\alpha$')  
    plt.ylabel('Approx. ratio' r'$\quad\frac{f(x)}{f(x^*)}$')    


    #foo_fig = plt.gcf() 
    #foo_fig.savefig('H_ratio.eps', format='eps', dpi=1200)

    plt.show() 

    plt.figure(figsize=(8,5),dpi=1200)
    plt.plot(a_range,Guassian1_time,'ro-')  
    plt.title('LASSO d=65')  
    plt.xlabel('Control parameter' r'$\quad\alpha$')  
    plt.ylabel('CPU runtime' r'$(s)$')    


    #foo_fig = plt.gcf() 
    #foo_fig.savefig('H_time.eps', format='eps', dpi=1200)

    plt.show() 
    return ratio_1,Guassian1_time

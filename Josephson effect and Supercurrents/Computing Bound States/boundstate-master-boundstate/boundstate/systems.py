import numpy as np
import kwant 
import scipy 

sx = np.array([[0 , 1] , [1 , 0]])
sy = np.array([[0 , -1j] ,[1j , 0]])
sz = np.array([[1 , 0],[0 , -1]])
I = np.identity(2)

def normal_junction(params):
    syst = kwant.Builder(particle_hole= sx)
    lat = kwant.lattice.square(a = params.a , norbs =2 )

    # Peierl Phase:
    junctionarea = params.W *params.L
    peierl = 2*np.pi*(params.flux / junctionarea)*(params.a)

    for i in range(params.L // params.a):
        for j in range(params.W // params.a):
            syst[lat(i , j)] = (4*params.t/(params.a**2) - params.mu + params.V_n)*np.array([[1, 0] ,[0 , -1]]) 
            if i>0:
                hopphase = np.exp(1j*peierl*j*params.a)
                hopping_matrix = np.diag([hopphase , hopphase.conj()])
                syst[lat(i , j) , lat(i-1 , j)] = (-params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) )@hopping_matrix
            if j > 0:
                syst[lat(i , j) , lat(i , j-1)] = -params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) 
            
    lead_L = kwant.Builder(kwant.TranslationalSymmetry((-params.a , 0)) , particle_hole= sx)
    for i in range(params.W // params.a):
        lead_L[lat(0 , i)] = ( (4*params.t/(params.a**2) - params.mu)*np.array([[1, 0] ,[0 , -1]]) 
                               + params.Delta*np.exp(1j*params.phi)*np.array([[0 , 1],[0 , 0]])
                               + params.Delta*np.exp(-1j*params.phi)*np.array([[0 , 0],[1 , 0]]))
        if i > 0:
            lead_L[lat(0 , i) , lat(0 , i-1)] = -params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) 
        lead_L[lat(1 ,i) , lat(0 , i)] = -params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) 

    lead_R = kwant.Builder(kwant.TranslationalSymmetry((params.a , 0)) , particle_hole= sx)
    for i in range(params.W // params.a):
        lead_R[lat(0 , i)] = ( (4*params.t/(params.a**2) - params.mu)*np.array([[1, 0] ,[0 , -1]]) 
                               + params.Delta*np.array([[0 , 1],[1 , 0]]))
        if i > 0:
            lead_R[lat(0 , i) , lat(0 , i-1)] = -params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) 
        lead_R[lat(1 ,i) , lat(0 , i)] = -params.t/(params.a**2)*np.array([[1, 0] ,[0 , -1]]) 

    syst.attach_lead(lead_L)
    syst.attach_lead(lead_R)

    syst = syst.finalized()
    return syst  

def topological_junction_rashba(params):

    tau_plus = np.kron(np.array([[0 , 1],[0 , 0]]) , I)    
    tau_minus = tau_plus.T
    syst = kwant.Builder()
    lat = kwant.lattice.square(a = params.a , norbs = 4)
    # The normal region is a single lattice site with no pair potential
    syst[lat(0 , 0)] = (-params.mu + 2*params.t + params.V_n)*np.kron(sz , I) + params.Zee*np.kron(I , sz)
    # syst[lat(1 , 0)] = (-params.mu + 2*params.t + params.V_n)*np.kron(sz , I) + params.B*np.kron(I , sz)
    # syst[lat(1 , 0) , lat( 0 , 0)] = -params.t*np.kron(sz , I) + 1j*params.alpha*np.kron(sz , sx)
    leadL = kwant.Builder(kwant.TranslationalSymmetry((-params.a , 0)))
    leadL[lat(0 , 0)] = ((-params.mu + 2*params.t)*np.kron(sz , I) + params.Zee*np.kron(I , sz) 
                        + params.Delta*np.exp(1j*params.phi)*tau_plus + params.Delta*np.exp(-1j*params.phi)*tau_minus)
    leadL[lat(1 , 0) , lat(0 , 0)] = -params.t*np.kron(sz , I) + 1j*params.alpha*np.kron(sz , sx)

    leadR = kwant.Builder(kwant.TranslationalSymmetry((params.a , 0)))
    leadR[lat(0 , 0)] = ((-params.mu + 2*params.t)*np.kron(sz , I) + params.Zee*np.kron(I , sz) 
                        + params.Delta*np.kron(sx , I))
    leadR[lat(1 , 0) , lat(0 , 0)] = -params.t*np.kron(sz , I) + 1j*params.alpha*np.kron(sz , sx)

    syst.attach_lead(leadL)
    syst.attach_lead(leadR)

    syst = syst.finalized()

    return syst


def qhz_junction(params):

    syst = kwant.Builder()
    lat = kwant.lattice.square(a = params.a, norbs = 4)

    # Peierl Phase:
    junctionarea = params.W *params.L
    peierl = 2*np.pi*(params.flux / junctionarea)*(params.a)
    for i in range(params.L // params.a):
        for j in range(params.W // params.a):
            syst[lat(i , j)] = params.C*np.kron(I , sz) + 4*(params.B/params.a**2)*np.kron(I , sz) + params.mu*np.kron(sz , I)    
            if i > 0:
                hopphase = np.exp(1j*peierl*j*params.a)
                hoppingmatrix = np.diag([hopphase , hopphase , hopphase.conj() , hopphase.conj()])
                syst[lat(i , j) , lat(i-1 , j)] = hoppingmatrix@(-1j*(params.A/params.a)*np.kron(sz , sx) - (params.B/params.a**2)*np.kron(I , sz) )
            if j > 0:
                syst[lat(i , j) , lat(i, j-1)] = (-1j*(params.A/params.a)*np.kron(sz , sy) - (params.B/params.a**2)*np.kron(I , sz) )
            
    leadL = kwant.Builder(kwant.TranslationalSymmetry((-params.a , 0)))
    
    for j in range(params.W // params.a):
        leadL[lat(0 , j)] = params.C*np.kron(I , sz) + 4*(params.B/params.a**2)*np.kron(I , sz) + params.mu*np.kron(sz , I)  + params.Delta*np.kron(sx , I)
        if j > 0:
            leadL[lat(0 , j) , lat(0 , j-1)] = (-1j*(params.A/params.a)*np.kron(sz , sy) - (params.B/params.a**2)*np.kron(I , sz) )
        leadL[lat(1 , j) , lat(0 , j)] = (-1j*(params.A/params.a)*np.kron(sz , sx) - (params.B/params.a**2)*np.kron(I , sz) )
    
    leadR = kwant.Builder(kwant.TranslationalSymmetry((params.a , 0)))
    
    for j in range(params.W // params.a):
        leadR[lat(0 , j)] = (params.C*np.kron(I , sz) + 4*(params.B/params.a**2)*np.kron(I , sz) + params.mu*np.kron(sz , I)  
                             + params.Delta*np.exp(1j*params.phi)*np.kron(0.5*(sx + 1j*sy) , I) + params.Delta*np.exp(-1j*params.phi)*np.kron(0.5*(sx - 1j*sy) , I))
        if j > 0:
            leadR[lat(0 , j) , lat(0 , j-1)] = (-1j*(params.A/params.a)*np.kron(sz , sy) - (params.B/params.a**2)*np.kron(I , sz) )
        leadR[lat(1 , j) , lat(0 , j)] = (-1j*(params.A/params.a)*np.kron(sz , sx) - (params.B/params.a**2)*np.kron(I , sz) )
    
    syst.attach_lead(leadL)
    syst.attach_lead(leadR)

    syst = syst.finalized()
    return syst


    

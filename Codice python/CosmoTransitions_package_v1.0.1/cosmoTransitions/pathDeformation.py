
__version__ = "1.0.1"

import numpy as np
from scipy import optimize, interpolate, special
import tunneling1D

pylab = None
	
# ------------------------------------------------------------------------------    <-- 80 characters long
class Deformation:
	"""
	This class deforms a set of points phi so that their path matches the
	equation
		d^2phi/dr^2 +f(phi,r) dphi/dr = grad_{phi}(V)
	where phi is an N-dimensional vector and f(phi, r) is any arbitrary scalar
	function. Note that the deformation is zero at either end of the path.
	"""
	def __init__(self, phi, dphidr, V, dV, nb, kb, v2min = 0.0, fixEnd = True):
		"""
		Inputs:
		phi - The n initial points of dimension N that we want to deform.
			Should be shape (n,N).
		dphidr - The 'speed' of the initial points (absolute value of dphi/dr).
			Should be shape (N,).
		V - The potential as a function of phi. Should be able to handle input
			arrays of shape (n,N) and output arrays of shape (n,).
		dV - The gradient of the potential as a function of phi. Should be able
			to handle input arrays of shape (n,N) and output arrays of shape
			(n,N).
		nb - The number of independent basis functions used to approximate the
			spline (default 15).
		kb - The degree of the polynomials used as basis functions (default 5).
		v2min - The smallest the square of dphidr is allowed to be, relative
			to the characteristic force exterted by dV. Note that the
			self-correcting nature of the deformation goes away when dphidr=0.
		"""		
		# First step: convert phi to a set of path lengths.
		phi = np.array(phi)
		dphi = phi[1:]-phi[:-1]
		dL = np.sqrt(np.sum(dphi*dphi,axis=-1))
		y = np.cumsum(dL)
		self.L = y[-1]
		self.t = np.append(0,y)/self.L # this is now how we parametrize the path
		self.t[0] = 1e-100 # Without this, the first data point isn't in any bin (this matters for dX).
		self.t = self.t[:,np.newaxis] # Shape (n, 1)
		self.fixEnd = fixEnd
		
		# Create the starting spline:
		# make the knots and then the spline matrices at each point t
		self.t0 = np.append(np.append([0.]*(kb-1), np.linspace(0,1,nb+3-kb)), [1.]*(kb-1)) # I don't actually use t0 again
		self.X,self.dX,self.d2X = Nbspld2(self.t0, self.t[:,0], kb)
		# subtract off the linear component.
		phi0, phi1 = phi[:1], phi[-1:]  # These are shape (1,N)
		phi_lin = phi0 + (phi1-phi0)*self.t
		self.beta, residues, rank, s = np.linalg.lstsq(self.X, phi-phi_lin)
		
		# save the points for future use.
		self.phi, self.v2 = phi, dphidr[:,np.newaxis]**2 # shape (n,N) and (n,1)
		self.phi_last = self.F_last = self.lastStep = None
		self.V, self.dV = V, dV
		self.i = 0
		self.sameCount = 0
		self.kb = kb
		self.lastStepReveresed = False
		dVp = self.dV(self.phi)
		self.forceNormalization = self.L/np.max(np.sqrt(np.sum(dVp*dVp, axis=-1)))
		
		# ensure that v2 isn't too small:
		v2 = dphidr**2
		v2min *= np.max( np.sum(dV(self.phi)**2, -1)**.5*self.L/nb )
		v2[v2<v2min] = v2min
		self.v2 = v2[:,np.newaxis]
		
		# We can set self.pylab to be the pylab library if we want to do plotting at each step.
		self.pylab = None
		
	def _deltaDeform(self, indexSlice = None):
		# This function should only be used internally. It gives the expected change in beta per change in stepsize.
		sl = slice(len(self.phi)) if indexSlice == None else indexSlice
		X, dX, d2X = self.X[sl], self.dX[sl], self.d2X[sl]
		beta = self.beta
		t = self.t[sl]
		# First, find phi, dphi, and d2phi.
		phi = self.phi[sl]
		dphi = np.sum(beta[np.newaxis,:,:]*dX[:,:,np.newaxis], axis=1) + (self.phi[-1]-self.phi[1])[np.newaxis,:]
		d2phi = np.sum(beta[np.newaxis,:,:]*d2X[:,:,np.newaxis], axis=1) # no linear component to add.
		# Now compute dphi/ds, where s is the path length instead of the path parameter t.
		# This is going to just be the direction along the path.
		dphi_sq = np.sum(dphi*dphi, axis=-1)[:,np.newaxis]
		dphids = dphi/np.sqrt(dphi_sq)
		# Then find the acceleration along the path, i.e. d2phi/ds2:
		d2phids2 = (d2phi - dphi * np.sum(dphi*d2phi, axis=-1)[:,np.newaxis]/dphi_sq)/dphi_sq
	
		# Now we have the path at the points t, as well its derivatives with respect to it's path length.
		# We still need to get the normal force acting on the path.
		dV = self.dV(phi) # get the gradient
		dV_perp = dV - np.sum(dV*dphids, axis=-1)[:,np.newaxis]*dphids
		F = d2phids2 * self.v2[sl] - dV_perp # normal force, direction we want to push the path
		
		# Normalize the normal force:
		if indexSlice == None:
			self.forceNormalization = self.L/np.max(np.sqrt(np.sum(dV*dV, axis=-1)))
		F *= self.forceNormalization
		return F, dV
						
	def step(self, maxstep = .1, stepIncrease = 1.5, reverseCheck = .15, stepDecrease = 5., minstep=1e-10, verbose = 0):
		"""
		This method deforms the path one step. Each point is pushed in the
		direction of the normal force - the force that the path exerts on a
		classical particle moving the inverted potential in order to keep the
		particle along the path. A stepsize of 1 corresponds to moving the path
		an amount L*N/(dV_max), where L is the length of the (original) path,
		N is the normal force, and dV_max is the maximum force exerted by
		the potential along the path.
		Inputs:
			maxstep - The largest stepsize that we'll allow.
			stepIncrease - The relative increase in stepsize each iteration.
			reverseCheck - The number of points in which the normal force need
				be reversed before the stepsize is decreased. If reverseCheck
				is one then all stepsizes will be constant.
			verbose - Set to at least 2 to get output at every step.
		Output:
			Returns (fRatio1, fRatio2). These numbers describe the ratio of the
			maximum normal force to the maximum force dV parallel to the spline.
			When these numbers are small, the routine converges. Note that
			fRatio2 is fitted to the spline, while fRatio1 is not. If the spline
			is a poor fit to the true path (i.e., there aren't enough basis
			functions), then only fRatio2 will converge.
		"""
		self.i += 1

		phi, lastStep = self.phi, self.lastStep
		
		if self.fixEnd == False:
			# Before we do anything else,
			# find the force acting on the first little section of the path.
			# It's important that this isn't just at the last point, since the force
			# there should always be zero (if v2 = 0 and it's aligned with the gradient)
			npoints, nb = self.X.shape
			i = (npoints/nb) * .5
			if i < 2: i = 2
			sl = slice(i)
			F0, dV = self._deltaDeform(sl)
			F0 = np.average(F0, axis=0)
			# Now move the path by moving the linear component
			phi0, phi1 = phi[:1], phi[-1:] # These are shape (1,N)
			Lold = np.sum((phi0-phi1)**2)**.5
			phi0 = phi0 + F0*lastStep
			Lnew = np.sum((phi0-phi1)**2)**.5
			phi_lin = phi1 + (phi0-phi1)*(1-self.t)*Lold/Lnew
			
			# Now, make sure that the gradient matches the path at the start
			dV0 = self.dV(phi0[0])
			dphi0 = self.L*dV0/np.sum(dV0*dV0)**.5
			beta0 = (dphi0 - (phi1-phi0)[0])/self.dX[0,0]
			self.beta[0] = beta0
			phi = np.sum(self.beta[np.newaxis,:,:]*self.X[:,:,np.newaxis], axis=1) + phi_lin
			self.phi = phi
						
		# Find out the direction of the deformation.
		F,dV = self._deltaDeform()
		fRatio1 = np.max(np.sqrt(np.sum(F*F,-1)))/self.L
				
		# Now, see how big the stepsize should be
		stepsize = lastStep
		if reverseCheck < 1 and self.F_last != None:
			FdotFlast = np.sum(F*self.F_last, axis=1)
			if np.sum(FdotFlast < 0) > len(FdotFlast)*reverseCheck:
				# we want to reverse the last step
				if stepsize > minstep:
					phi, F = self.phi_last, self.F_last
					self.lastStepReveresed = True
					if verbose >= 2: print "step reversed"
				elif minstep == maxstep:
					raise Exception, "Error in Deformation.step. Step failed reverse check with constant stepsize."
				stepsize = lastStep/stepDecrease
			else:	# No (large number of) indices reversed, just do a regular stepsize
				stepsize = lastStep * stepIncrease # Increase the stepsize a bit over the last one.
				self.lastStepReveresed = False
		if stepsize > maxstep: stepsize = maxstep
		if stepsize < minstep: stepsize = minstep
				
		# Save the state before the step
		self.lastStep, self.phi_last, self.F_last = stepsize, phi, F
				
		# now make the step
		phi_lin = phi[:1] + (phi[-1:]-phi[:1])*self.t
		phi = phi+F*stepsize # important to not use += so that we don't change phi_last too
		temp_phi = phi.copy()
		
		# fit to the spline
		phi -= phi_lin
		self.beta, residues, rank, s = np.linalg.lstsq(self.X, phi)
		phi = np.sum(self.beta[np.newaxis,:,:]*self.X[:,:,np.newaxis], axis=1)
		phi += phi_lin
		self.phi = phi
		
		Ffit = (phi-self.phi_last)/stepsize
		fRatio2 = np.max(np.sqrt(np.sum(Ffit*Ffit,-1)))/self.L
		
		pylab = self.pylab
		if (pylab):
			p, dp = phi, F
			pylab.figure(1); pylab.plot(p[:,0], p[:,1],'k')
			pylab.figure(1); pylab.plot(temp_phi[:,0], temp_phi[:,1],'g')
			pylab.figure(1); pylab.plot((p+dp)[:,0], (p+dp)[:,1],'m')
			pylab.figure(2); pylab.plot(self.t[:,0], dp[:,0],'r')
			pylab.figure(2); pylab.plot(self.t[:,0], dp[:,1],'b')
			
		if verbose >= 2: print self.i, stepsize, fRatio1, fRatio2
		
		return fRatio1, fRatio2		
		
	def deformPath(self, fRatioConv = .02, DeltaV_max = np.inf, maxiter = 500, maxstep = .1, minstep=1e-4, startstep = 2e-3, 
		stepIncrease = 1.5, reverseCheck = .1, fRatioIncrease = 5., stepDecrease = 5., verbose = 1, **etc):
		"""
		This method deforms the path many steps until either the convergence
		criterium is reached, the potential has changed substantial, or the
		maximum number of iterations is reached.
		Inputs:
			fRatioConv - The routine stops when fRatio2 < fRatioConv.
				See the helpfile for Deformation.step for more info.
			DeltaV_max - The maximum acceptable change in the value of the
				potential, relative to the height of the potenital
				initially.
			maxiter - Maximum number of allowed iterations.
			maxstep, stepIncrease, reverseCheck - Values used for step routine.
			fRatioIncrease - The maximum amount the fRatio can increase before
				we throw out an error.
		Outputs rcode:
		  0 for convervegence after just 1 iteration
		  +1 for convergence after multiple iterations
		  +2 for stop by change in potential
		  -1 for maximum number of iterations
		  -2 for non-convergence of deformation
		"""
		if len(etc) > 0 and verbose >= 2:
			print "Warning, extra parameters sent to pathDeformation.Deformation:", etc
			
		V0 = self.V(self.phi)
		DVmax = (np.max(V0)-np.min(V0))*DeltaV_max
		i = 0
		minfRatio = np.inf
		minfRatio_index = 0
		self.lastStep = startstep
		while True:
			i += 1
			self.fRatio1,self.fRatio2 = self.step(maxstep, stepIncrease, reverseCheck, stepDecrease, minstep=minstep, verbose=verbose)
			fRatio = min(self.fRatio1, self.fRatio2)
			minfRatio = min(minfRatio, fRatio)
			if (fRatio < fRatioConv or i == 1 and fRatio < 2*fRatioConv):
				if verbose >= 1: print "Path deformation converged."
				return 0 if i == 1 else +1
			if minfRatio == fRatio:
				minfRatio_state = (self.beta, self.phi)
				minfRatio_index = i
			if fRatio > fRatioIncrease*minfRatio and not self.lastStepReveresed:
				self.beta, self.phi = minfRatio_state
				if verbose >= 1: print "Deformation doesn't appear to be converging. Stopping at the point of best convergence."
				return -2 if minfRatio_index > 5 else 0
			V = self.V(self.phi)
			if np.max(np.abs(V0-V)) > DVmax:
				if verbose >= 1: print "Maximum allowed change in potential reached."
				print DVmax, np.max(np.abs(V0-V))
				return +2
			if self.i >= maxiter:
				if verbose >= 1: print "Maximum number of iterations reached."
				return -1
		return 0 # shouldn't ever get here
	
	def extrapolatePhi_old(self, npoints = 100., tails = .2):
		"""
		Creates a list of points phi along with the path length to those points.
		Inputs:
		  npoints - The number of points from that we want to span the length
		    of self.phi.
		  tails - The length of the tails. These extend beyond self.phi.
		Returns x, phi, L
		"""
		t = np.linspace(1e-10, 1, npoints)[:,np.newaxis] # The first point needs to be > 0, but just barely.
		dt = t[1,0]
		X,dX,d2X = Nbspld2(self.t0, t[:,0], self.kb)
		if self.fixEnd == False and False:
			X = np.append(X, 1-t, axis=1)
			dX = np.append(dX, 0*t-1, axis=1)
			d2X = np.append(d2X, 0*t, axis=1)
		dphi = np.sum(self.beta[np.newaxis,:,:]*dX[:,:,np.newaxis], axis=1) # dphi/dt
		if self.fixEnd == True:
			dphi += (self.phi[-1]-self.phi[0])[np.newaxis,:] # Need to add in linear component.
		dphi_mag = np.sqrt(np.sum(dphi*dphi,axis=-1))
		dL = .5*(dphi_mag[1:]+dphi_mag[:-1])*dt
		x1 = np.append(0, np.cumsum(dL)) # path length along the main section
		L = x1[-1]
		phi1 = self.phi

		p0,dp0 = self.phi[0], dphi[0]/dphi_mag[0]
		V0 = lambda x: self.V(p0+dp0*x)
		x0min = optimize.fmin(V0, 0, disp=0)
		if x0min > 0: x0min = 0.0
		p2,dp2 = self.phi[-1], dphi[-1]/dphi_mag[-1]
		V2 = lambda x: self.V(p0+dp0*x)
		x2min = optimize.fmin(V0, L, disp=0)
		if x2min < L: x2min = L
		x0 = np.linspace(x0min-L*tails,0,npoints*tails)[:-1] # path length along start tail
		x2 = np.linspace(L,x2min+L*tails,npoints*tails)[1:] # path length along end tail
		phi0 = (dphi[0]/dphi_mag[0])[np.newaxis,:]*x0[:,np.newaxis] + self.phi[0][np.newaxis,:]
		phi2 = (dphi[-1]/dphi_mag[-1])[np.newaxis,:]*(x2[:,np.newaxis]-L) + self.phi[-1][np.newaxis,:]
		x = np.append(x0, np.append(x1, x2))
		phi = np.append(phi0, np.append(phi1, phi2, 0), 0)

		return x, phi, L
		
def extrapolatePhi(phi, V=None, tails = .2):
	"""
	Returns a list of points along the path, going linearly
	beyond the path to include the nearest minima.
	Inputs:
	  phi - the path to extend
	  V - the potential to minimize. Can be None if you just want to extend the tails.
	  tails - fractional amount to go beyond the end of the path/minima.
	Outputs:
	  phi - list of points
	  s - path length to the points
	  L - length of path excluding tails.
	"""
	phi1 = phi
	dphi = np.append(0,np.sum((phi1[1:]-phi1[:-1])**2,1)**.5)
	s1 = np.cumsum(dphi)
	L = s1[-1]
	npoints = phi1.shape[0]
	
	phi_hat0 = (phi[1]-phi[0])/np.sum((phi[1]-phi[0])**2)**.5
	if V == None:
		s0min = 0.0
	else:
		V0 = lambda x: V( phi[0] + phi_hat0*x)
		s0min = optimize.fmin(V0, 0.0, disp=0)[0]
	if s0min > 0: s0min = 0.0
	s0 = np.linspace(s0min - L*tails, 0.0, npoints*tails)[:-1]
	phi0 = phi[0] + phi_hat0*s0[:,np.newaxis]

	phi_hat2 = (phi[-1]-phi[-2])/np.sum((phi[-1]-phi[-2])**2)**.5
	if V == None:
		s2min = 0.0
	else:
		V2 = lambda x: V( phi[-1] + phi_hat2*(x-L))
		s2min = optimize.fmin(V2, L, disp=0)[0]
	if s2min < L: s2min = L
	s2 = np.linspace(L, s2min + L*tails, npoints*tails)[1:]
	phi2 = phi[-1] + phi_hat2*(s2[:,np.newaxis]-L)
	
	phi = np.append(phi0, np.append(phi1, phi2, 0), 0)
	s = np.append(s0, np.append(s1, s2))
	
	return phi, s, L

		
# -----------------
# The following functions return the matrices that, when multiplied by spline coefficients, give the path.
# These are directly copied out of the file funcs/interpExtras.py.
def Nbspl(t, x, k = 3):
	"""This function returns the B-spline basis functions for the knots t evaluated at the points x.
	The return shape is (len(x), len(t)-1-k).
	For len(x) = 500, len(t) = 20, and k = 3, this operates in a few milliseconds.
	"""
	kmax = k
	if kmax > len(t)-2:
		raise Exception, "Input error in Nbspl: require that k < len(t)-2"
	t = np.array(t)#[np.newaxis, :]
	t2 = t.copy()
	x = np.array(x)[:, np.newaxis]
	N = 1.0*((x > t[:-1]) & (x <= t[1:]))
	for k in xrange(1, kmax+1):
		dt = t[k:] - t[:-k]
		_dt = dt.copy()
		_dt[dt != 0] = 1./dt[dt != 0]
		N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
	return N
	
def Nbspld1(t, x, k = 3):
	"Same as Nbspl, but returns the first derivative too."
	kmax = k
	if kmax > len(t)-2:
		raise Exception, "Input error in Nbspl: require that k < len(t)-2"
	t = np.array(t)#[np.newaxis, :]
	x = np.array(x)[:, np.newaxis]
	N = 1.0*((x > t[:-1]) & (x <= t[1:]))
	dN = np.zeros_like(N)
	for k in xrange(1, kmax+1):
		dt = t[k:] - t[:-k]
		_dt = dt.copy()
		_dt[dt != 0] = 1./dt[dt != 0]
		dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] 
		dN += N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:] 
		N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
	return N, dN
	
def Nbspld2(t, x, k = 3):
	"Same as Nbspl, but returns first and second derivatives too."
	kmax = k
	if kmax > len(t)-2:
		raise Exception, "Input error in Nbspl: require that k < len(t)-2"
	t = np.array(t)#[np.newaxis, :]
	x = np.array(x)[:, np.newaxis]
	N = 1.0*((x > t[:-1]) & (x <= t[1:]))
	dN = np.zeros_like(N)
	d2N = np.zeros_like(N)
	for k in xrange(1, kmax+1):
		dt = t[k:] - t[:-k]
		_dt = dt.copy()
		_dt[dt != 0] = 1./dt[dt != 0]
		d2N = d2N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - d2N[:,1:]*(x-t[k+1:])*_dt[1:] \
			+ 2*dN[:,:-1]*_dt[:-1] - 2*dN[:,1:]*_dt[1:]
		dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] \
			+ N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:]
		N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
	return N, dN, d2N

# ------------------------------------------------------------------------------    <-- 80 characters long

class fullTunneling:
	def __init__(self, phi, V, dV, alpha = 2, npoints = 100, quickTunneling = False, fixEndCutoff = .05, \
					pylab = None, deformPlot = False, **etc):
		"""
		Inputs:
		  phi - Either an array of starting points, or a tuple containing the
		    start and end points. If an array, should be of shape (n,N) where
			N is the dimension of the space.
		  V(phi) - The potential. Should accepts arrays of shape (n,N)
		  dV(phi) - The gradient of the potential. Should accept arrays of
		    shape (n,N) and output arrays of shape (n,N).
		  alpha - The number of space-time dimensions minus one. Used in
		    the equations of motion.
		  npoints - The number of points to sample along the bubble profile.
		  quickTunneling - If True, the velocity along the bubble profile is
		    simply found by conversation of energy. The over/undershooting
			method is only used on the last step. If False, the full bubble
			profile is found before each deformation.
		  fixEndCutoff - If |phi(r=0) - phi_absmin| > fixEndCutoff*|phi_absmin-phimetamin|,
		    then we treat it as a thick wall and do not fix the end during
			deformation. Otherwise, do fix the end.
		  pylab - Set this to the pylab module to have plotting at each step.
		  deformPlot - Set this to true along with pylab to have it plot each
		    deformation.
		  **etc - Unused, but prevents an error for passing in extra parameters.
		"""
	#	if len(etc) > 0:
	#		print "Warning, extra parameters sent to pathDeformation.fullTunneling:", etc
		self.V, self.dV, self.alpha, self.npoints = V, dV, alpha, npoints
		self.quickTunneling = quickTunneling
		self.pylab = pylab
		self.deformPlot = deformPlot
		self.rcode = None
		self.fixEnd = True # default to True. Can only go from True to False, not other way around
		self.fixEndCutoff = fixEndCutoff
		self.lastProfile = None
		self.tranType = 1
		
		# First step, make the points phi if they weren't input.
		self.phi = phi = np.array(phi)
		if len(phi) == 2:
			Dphi = (phi[1]-phi[0])
			self.phi = phi = np.linspace(0,1,npoints)[:,np.newaxis]*Dphi[np.newaxis,:] + phi[0][np.newaxis,:]

		# The absolute minimum should always be at the start of the array. If not, return early.
		if V(self.phi[0]) > V(self.phi[-1]):
			self.L = None
			return
			
		# Now that we have phi, we need to find V(x), phi(x)
		if False:
			dphi = phi[1:]-phi[:-1]
			dx = np.sum(dphi*dphi, -1)**.5
			x1 = np.append(0, np.cumsum(dx))
			phi1 = phi
			self.L = x1[-1]
			tails = .2
			x0 = x1[1:npoints*tails]
			phi0 = -x0[::-1,np.newaxis]*(dphi[0]/dx[0])[np.newaxis,:] + phi[0][np.newaxis,:]
			phi2 = x0[:,np.newaxis]*(dphi[-1]/dx[-1])[np.newaxis,:] + phi[-1][np.newaxis,:]
			x0, x2 = -x0[::-1], self.L+x0
			x = np.append(x0, np.append(x1,x2))
			phi = np.append(phi0, np.append(phi, phi2, 0), 0)
			V_ = V(phi)
			self.x, self.phi, self.V_ = x,phi,V_		
		self.phi, self.x, self.L = extrapolatePhi(phi, self.V)

		phi,x = self.phi, self.x
		phi_tck = interpolate.splprep(phi.T, u=x, k=3, s=0)[0]
		self.phi_interp = lambda x, tck=phi_tck: np.array(interpolate.splev(x, tck)).T
		V_tck = interpolate.splrep(x, self.V(phi), k=3, s=0)
		self.V_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=0)
		self.dV_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=1)
		self.d2V_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=2)
				
	def tunnel1D(self, xtol = 1e-4, phitol = 1e-6):
		pmin = optimize.fmin(self.V_interp, 0.0, disp=0)[0]
		pmax = optimize.fmin(self.V_interp, self.L, disp=0)[0]
		# We want to make sure that pmax hasn't changed too much. If it has, it signals that that phase doesn't exist.
		if abs(pmax-self.L) > .2*self.L:
			self.action = -np.inf
			self.phi = None
			return
		A = tunneling1D.bubbleProfile(pmin, pmax, self.V_interp, self.dV_interp, self.d2V_interp(pmin), alpha = self.alpha)
		r, phi, dphi = A.findProfile(xtol = xtol, phitol = phitol, npoints = self.npoints*2, thinCutoff = 1e-3)
		self.lastProfile = {"r":r,"phi1d":phi,"dphi":dphi}
		self.lastProfile["phi"] = self.phi_interp(phi)
		if abs(phi[0]-pmin) > self.fixEndCutoff*abs(pmax-pmin):
			self.fixEnd = False
		phi_even, dphi_even = A.evenlySpacedPhi(phi, dphi, npoints = self.npoints, fixAbs = self.fixEnd)
		self.phi, self.dphi = self.phi_interp(phi_even), dphi_even
	#	self.lastProfile["phi"] = self.phi.copy()
		
		if self.pylab != None and self.deformPlot == False:
			self.pylab.figure(2)
			self.pylab.plot(self.lastProfile["r"], self.lastProfile["phi1d"])
			
	def doQuickTunnel(self):
		x = np.linspace(0.0, self.L, self.npoints)
		self.phi = self.phi_interp(x)
		V = self.V(self.phi)
		dphi2 = 2*( V-V[-1] )
		j = dphi2 < 0
		dphi2[j] = 0.0
		self.dphi = dphi2**.5
		if sum(j) > self.fixEndCutoff*len(j):
			self.fixEnd = False
		if self.fixEnd == False:
			self.dphi = self.dphi[~j]
			self.phi = self.phi[~j]
		
	def deform(self, nb=10, kb=3, **deformationParams):
		# First deform the path
		initParams = {}
		if "v2min" in deformationParams:
			initParams["v2min"] = deformationParams["v2min"]
		A = Deformation(self.phi, self.dphi, self.V, self.dV, nb,kb, fixEnd = self.fixEnd, **initParams)
		if self.deformPlot:
			A.pylab = self.pylab
		rcode = A.deformPath(**deformationParams)
		self.phi = A.phi
		self.deformation = A
		# Then make the interpolation functions
		self.phi2,self.x2,self.L = extrapolatePhi(A.phi, self.V)
		V_ = self.V(self.phi2)
		
		phi_tck = interpolate.splprep(self.phi2.T, u=self.x2, k=3, s=0)[0]
		self.phi_interp = lambda x, tck=phi_tck: np.array(interpolate.splev(x, tck)).T
		V_tck = interpolate.splrep(self.x2, V_, k=3, s=0)
		self.V_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=0)
		self.dV_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=1)
		self.d2V_interp = lambda x, tck=V_tck: interpolate.splev(x, tck, der=2)
		
		if self.pylab != None and self.deformPlot == False:
			self.pylab.figure(1)
			self.pylab.plot(self.phi[:,0], self.phi[:,1], 'k')
		if self.pylab:
			self.pylab.figure(3)
			self.pylab.plot(np.cumsum(np.append(0, np.sum((A.phi[1:]-A.phi[:-1])**2,axis=-1)**.5)), A.v2.ravel())
			
		return rcode
				
	def run(self, xtol = 1e-4, phitol = 1e-6, nb=10, kb=3, maxiter2 = 20, fRatioConv = .02, **deformationParams):
		deformationParams["fRatioConv"] = fRatioConv
	#	if self.V(self.phi[0]) > self.V(self.phi[-1]):
		if self.L == None:
			# don't do anything if the minima are reversed
			return 0
			
		self.rcode = +1
		i = 0
		while i < maxiter2:
			i += 1
			print i, "Updating the bubble profile."
			if self.quickTunneling or self.alpha==0:
				self.doQuickTunnel()
			else:
				self.tunnel1D(xtol,phitol)
			if self.phi == None:
				print "fullTunneling failed in 1D tunneling."
				self.rcode = -11
				break
			print "Deforming the path."
			self.rcode = self.deform(nb, kb, **deformationParams)
			if self.rcode == 0:
				break # success!
			elif min(self.deformation.fRatio2, self.deformation.fRatio1) > fRatioConv * 50:
				print "fullTunneling.run: Deformation failed far away from convergence."
				self.rcode = -10
				break
		if self.quickTunneling:
			self.tunnel1D(xtol, phitol)
		if i >= maxiter2 and self.rcode > 0:
			self.rcode -3
		return self.rcode
		
	def findAction(self):
		if self.V(self.phi[0]) > self.V(self.phi[-1]):
			self.action = np.inf
			return self.action
			
		d = self.alpha+1 # Number of dimensions in the integration
		omega =  2*np.pi**(d*.5)/special.gamma(d*.5) # area of unit sphere in d dimensions
		r,phi,dphi = self.lastProfile["r"], self.lastProfile["phi"], self.lastProfile["dphi"]
		
		if r[0] < 0:
			self.action = self.actionerr_V = self.actionerr_dphi = np.inf
			return np.inf
		# DV is the difference in potential energy of V(phi) - V(phi_metastable)
		DV = self.V(phi) - self.V(self.phi[-1]) # Note that DV[-1] != 0 unless the shooting converged perfectly.
		dr = r[1:] - r[:-1]
		r_mid = .5*(r[1:] + r[:-1])
		dphi_mid = .5*(dphi[1:] + dphi[:-1]) # find the values at the midpoints (not that there should be much difference)
		DV_mid= .5*(DV[1:] + DV[:-1])
		bulk = (omega/d)*r[0]**d*DV[0]
		wall = omega * np.sum( r_mid**self.alpha * ( .5*dphi_mid**2 + DV_mid ) * dr )
		self.action = bulk+wall
		self.actionerr_V = np.abs( (omega/d)*r[-1]**d*DV[-1] / self.action )
		self.actionerr_dphi = np.abs( (omega/d)*(r[-1]**d-r[0]**d) * .5*dphi[-1]**2 / self.action )
		return self.action
			
def criticalTunneling(*args, **params):
	A = criticalTunneling_class(*args, **params)
	return A.run()
			
class criticalTunneling_class:
	"""
	This class will find the transition temperature of a potential
	V(phi, T) such that S3 = nuclCriterium(T) (generally 140*T).
	"""
	def __init__(self, V, dV, Tmin, Tmax, tck_phiLow, tck_phiHigh, nuclCriterium = lambda T: T*140.0, \
						npoints = 100, phieps = 1e-3, phimintol = 1e-5, Ttol = 1e-3, quickTunneling = False, **runParams):
		"""
		Inputs:
		  V, dV - The potential and its gradient. V(phi, T). Set None for dV
		    for the gradient to be calculated automatically.
		  Tmin, Tmax - The minimum and maximum temperature between which we'll
			try to tunnel.
		  tck_phiLow, tck_phiHigh - Knots and spline coefficients for the minima
			of the potential as a function of T.
		  nuclCriterium - The function such that S/T = nuclCriterium(T)
			at the actual nucleation temperature.
		  npoints - The number of points to sample along the bubble profile.
		  phieps - The difference in phi to use for calculation of the gradient.
		  phimintol - The tolerance to use for minimization.
		  Ttol - The tolerance for finding the nucleation temperature.
		  quickTunneling - If True, the velocity along the bubble profile is
			simply found by conversation of energy. The over/undershooting
			method is only used on the last step. If False, the full bubble
			profile is found before each deformation.
		  runParams - Parameters to input into fullTunneling.run()
		Outputs:
		  fullTunneling object
		  Tcrit
		"""
		self.V, self.dV, self.Tmin, self.Tmax, self.tck_low, self.tck_high = V, dV, Tmin, Tmax, tck_phiLow, tck_phiHigh
		self.nuclCriterium, self.npoints = nuclCriterium, npoints
		self.quickTunneling, self.runParams = quickTunneling, runParams
		
		s = np.array(tck_phiHigh[1]).shape
		self.N = s[0] if len(s) > 1 else 1
		if len(s) == 1:
			# Need to get the potential to accept things of shape (..., N)
			self.V = lambda phi, T: V(phi[...,0],T)
			self.dV = lambda phi, T: dV(phi[...,0],T)[...,np.newaxis]
			self.tck_low = [self.tck_low[0], [self.tck_low[1]], self.tck_low[2]]
			self.tck_high = [self.tck_high[0], [self.tck_high[1]], self.tck_high[2]]
		
		if dV == None:
			self.phieps = phieps
			self.phiepsM = np.diag(np.ones(self.N))*phieps
			self.dV = lambda phi, T: .5*(self.V(phi[...,np.newaxis]+self.phiepsM)-self.V(phi[...,np.newaxis]-self.phiepsM))/self.phieps
			
		self.phitol = phimintol
		self.Ttol = Ttol
		self.phiLow = lambda T: np.array(interpolate.splev(T, self.tck_low)).T
		self.phiHigh = lambda T: np.array(interpolate.splev(T, self.tck_high)).T
		self.tunnelObj = self.tunnelObj_old = None
		
	def _calcAtTemp(self, T):
		print "Calculating bubble at T =",T
		phiLow = optimize.fmin(self.V, self.phiLow(T), args=(T,), xtol=self.phitol, ftol=np.inf, disp=0)
		phiHigh = optimize.fmin(self.V, self.phiHigh(T), args=(T,), xtol=self.phitol, ftol=np.inf, disp=0)
		print "phiLow = ",phiLow
		print "phiHigh = ",phiHigh
		if np.sum((phiHigh-phiLow)**2)**.5 < 20*self.phitol:
			# the two minima are really close together. Assume that we're below the transition.
			return -np.inf
			
		path = (phiLow, phiHigh)
		# get the approximate path by averaging the last two paths (this needs to be completed)
		to1, to2 = self.tunnelObj, self.tunnelObj_old
		if to1 != None and to2 != None and to1.lastProfile != None and to2.lastProfile != None and False:
			# Everything that we need has been calculated
			if (to1.T < T < to2.T or to2.T < T < to1.T) and abs(to1.T-to2.T) < abs(self.Tmax-self.Tmin)*.9:
				# and we're in about the right temperature range to average
				# if they're both thick-walled, just average the two paths directly
				# if they're both thin-walled, subtract off the linear component and average that, adding back
				# in the linear component for the current temp
				# if one's thick and one's thin, just use the thin-walled one.
				w1 = abs(T-to1.T)/abs(to2.T-to1.T)
				w2 = abs(T-to2.T)/abs(to2.T-to1.T)
				if to1.fixEnd == False and to2.fixEnd == False:
					print "averaging two thick-walled solutions for initial condition"
					path = w1*to1.lastProfile["phi"]+w2*to2.lastProfile["phi"]
					path += phiHigh - path[-1]
				elif to1.fixEnd == True and to2.fixEnd == True:
					pass
				else:
					to0 = to1 if to1.fixEnd else to2
			
		
		self.tunnelObj_old = self.tunnelObj
		self.tunnelObj = fullTunneling(path, lambda x: self.V(x,T), lambda x: self.dV(x,T), alpha=2, \
						npoints=self.npoints, quickTunneling = self.quickTunneling, pylab=pylab)
		self.tunnelObj.T = T
		self.tunnelObj.tranType = 1
		rcode = self.tunnelObj.run(**self.runParams)
		if (rcode <= -10):
			err = "Error in deformation (rcode = "+str(rcode)+")."
			if T == self.Tmin:
				S = self.tunnelObj.action = -np.inf
				print err
			elif T == self.Tmax:
				S = self.tunnelObj.action = +np.inf
				print err
			else:
				raise Exception, err
		else:
			S = self.tunnelObj.findAction()
		if abs(T) > 0:
			print "S/T =",S/T,"\n"
		return S - self.nuclCriterium(T)
	
	def run(self):
	#	print "Tunneling at T =", (self.Tmin+self.Tmax)/2
	#	print self._calcAtTemp((self.Tmin+self.Tmax)/2)
	#	return self.tunnelObj
		
		try:
			print self.Ttol
			Tcrit = optimize.brentq(self._calcAtTemp, self.Tmin, self.Tmax, xtol=self.Ttol)
		except ValueError, err:
			# Make sure thate to1.T < to2.T
			to1, to2 = (self.tunnelObj, self.tunnelObj_old) if self.tunnelObj.T < self.tunnelObj_old.T \
					else (self.tunnelObj_old, self.tunnelObj)
			if to2.action > self.nuclCriterium(to2.T) and to1.action > self.nuclCriterium(to1.T):
				return to1 # The start and end points are above the critical temperature. Return the lower temperature profile.
			if to2.action < self.nuclCriterium(to2.T) and to1.action < self.nuclCriterium(to1.T):
				return to2 #  The start and end points are below the critical temperature. Return the higher temperature profile.
			raise ValueError, err
	#	except Exception, err:
	#		print err
	#		return None
		# If we got here, we found the critical temperature without an error
		if self.tunnelObj.T == Tcrit:
			return self.tunnelObj
		if self.tunnelObj_old.T == Tcrit:
			return self.tunnelObj_old
		self._calcAtTemp(Tcrit)
		return self.tunnelObj
		
	
class secondOrderTransition:
	"""
	This class doesn't really do anything, but it defines an object that
	can be handled in the same sort of way as fullTunneling objects.
	"""
	def __init__(self, phi, T):
		self.phi = phi
		self.T = T
		self.tranType = 2
		self.action = -np.inf
		
	def findAction(self):
		return -np.inf
		
		
		
				

__version__ = "1.0.1"

import numpy as np
from scipy import optimize, interpolate
import transitionFinder, pathDeformation
from finiteT import Jb_spline as Jb
from finiteT import Jf_spline as Jf

# ------------------------------------------------------------------------------    <-- 80 characters long

class generic_potential:
	"""
	This class will operate as the abstract super-class from which one can
	easily create finite temperature effective potentials.
	
	The following absolutely must be overrided in subclasses:
	  init() - initialization including self.Ndim
	  V0() - the tree-level potential
	  boson_massSq() and fermion_massSq() - the particle content of the theory
	  approxZeroTMin() - a list of the zero temp minima (you could get away 
	    without implementing this, but only if there is only one minimum)
	"""
	def __init__(self, *args, **dargs):
		self.Ndim = 0  # subclasses should override this to specify the number of dimensions in the potential
		self.x_eps = .001 # The epsilon to use in brute-force evalutations of the gradient.
		self.T_eps = .001 # The epsilon to use in brute-force evalutations of the temperature derivative.
		self.renormScaleSq = 1000.**2 # The square of the renormalization scale to use in the MS-bar one-loop zero-temp potential.
		self.Tmax = 1e4 # This is the highest temperature we're willing to try. Only the high-T phase should exist above this.
		self.forbidPhaseCrit = None # When tracing the minimum, we forbid phases that have starts satisfying forbidPhaseCrit(X)=True.
		
		self.T0 = None # this gets set by findT0()
		self.phases = self.transitions = None # These get set by getPhases
		self.TcTrans = None # Set by calcTcTrans()
		self.TnTrans = None # Set by calcFullTrans()
		
		if 'pylab' in dargs:
			self.pylab = dargs['pylab']
			del dargs['pylab']
		else:
			self.pylab = None
		
		self.init(*args, **dargs)
		self._makeDerivMatrices()
		
		if self.Ndim <= 0:
			raise Exception, "The number of dimensions in the potential must be at least 1."
		
	def init(self, *args, **dargs):
		# Subclasses should override this method (not __init__) to do all initialization. 
		# At a bare minimum, subclasses need to specify the number of dimensions in the potential with self.Ndim.
		pass
		
	# EFFECTIVE POTENTIAL CALCULATIONS -----------------------
		
	def _makeDerivMatrices(self):
		# This function makes the matrices for calculating the first and second derivatives.
		N = self.Ndim
		eps = self.x_eps
		
		d1 = np.zeros((N,N,N))
		d2 = np.zeros((N,N,N))
		for i in xrange(N):
			for j in xrange(N):
				if(i == j):
					d1[i,i,i] = 1
				else:
					d1[i,j,i] = .5; d1[i,j,j] = .5
					d2[i,j,i] = .5; d2[i,j,j] -= .5
		d1 *= eps
		d2 *= eps
		self._d1M, self._d2M = d1,d2
		self._gradM = np.diag(np.ones(N))*eps

	def V0(self, X):
		# Here, and everywhere else, arrays of points should be of the shape (..., Ndim)
		# This should be overridden.
		return X[...,0]*0
		
	def boson_massSq(self, X, T):
		# This should return three things:
		#	1) The masses of all of the bosons (including thermal corrections)
		#	2) The number of degrees of freedom of each boson
		#	3) The constant to feed in to the one-loop corrections. This should be
		#	   c = 1/2 for gauge boson traverse modes, and c = 3/2 for everything else.
		# The input should be such that X.shape[:-1] and T.shape are broadcastable.
		# The output should be (massSq, dof, c) where
		#	shape(massSq) = (..., Nboson)
		#	shape(dof) = shape(c) = (Nboson,)  (these could be constants, actually)
		Nboson = 1
		massSq = (X[...,0]*T)[..., np.newaxis]*np.zeros(Nboson) # placeholder, but has the right shape
		dof = np.zeros(Nboson)
		c = np.zeros(Nboson)
		return massSq, dof, c
		
	def fermion_massSq(self, X):
		# This is the same as boson_massSq, except for fermions and no thermal corrections.
		# Output should just be massSq and dof, since c = 3/2 for all fermions.
		Nfermions = 1
		massSq = X[...,0][..., np.newaxis]*np.zeros(Nfermions) # placeholder, but has the right shape
		dof = np.zeros(Nfermions)
		return massSq, dof
		
	def V1(self, bosons, fermions):
		"""
		The one-loop corrections to the zero-temperature potential
		using MS-bar renormalization.
		"""
		# This does not need to be overridden.
		m2, n, c = bosons
		y = np.sum( n*m2*m2*( np.log(np.abs(m2/self.renormScaleSq + 1e-100)) - c ), axis=-1 )
		m2, n = fermions
		c = 1.5
		y -= np.sum( n*m2*m2*( np.log(np.abs(m2/self.renormScaleSq + 1e-100)) - c ), axis=-1 )
		return y/(64*np.pi*np.pi)
		
	def V1T(self, bosons, fermions, T):
		"""
		The one-loop finite-temperature potential.
		"""
		# This does not need to be overridden.
		T2 = (T*T)[..., np.newaxis] + 1e-100 # the 1e-100 is to avoid divide by zero errors
		T4 = T*T*T*T
		m2, n, c = bosons
		y = np.sum( n*Jb(m2/T2), axis=-1)
		m2, n = fermions
		y += np.sum( n*Jf(m2/T2), axis=-1) 
		return y*T4/(2*np.pi)
		
	def Vtot(self, X, T):
		"""
		The total finite temperature effective potential.
		X should be of shape (xshape, Ndim), and T should be
		broadcastable to xshape.
		"""
		T = abs(np.array(T))*1.0
		X = np.array(X)
		bosons = self.boson_massSq(X,T)
		fermions = self.fermion_massSq(X)
		return self.V0(X) + self.V1(bosons,fermions) + self.V1T(bosons, fermions, T)
		
	def DVtot(self, X, T):
		"""
		The finite temperature effective potential, but rescaled
		so that V(0, T) = 0.
		"""
		X0 = np.zeros(self.Ndim)
		return self.Vtot(X,T) - self.Vtot(X0,T)
		
	def gradV(self, X, T):
		"""
		Calculates the gradient of the potential.
		Output has same shape as X.
		"""
		X = np.array(X)[..., np.newaxis, :]
		T = np.array(T)[..., np.newaxis]
		return (self.Vtot(X+self._gradM,T)-self.Vtot(X-self._gradM,T))/(2*self.x_eps)
		
	def massSqMatrix(self, X):
		"""
		Calculates the tree-level mass matrix.
		"""
		X = np.array(X)[..., np.newaxis, np.newaxis, :]
		return (self.V0(X+self._d1M)+self.V0(X-self._d1M)-self.V0(X+self._d2M)-self.V0(X-self._d2M))/self.x_eps**2
	
	def d2V(self, X, T):
		"""
		Calculates the second derivative matrix for the finite-temp effective potential.
		"""
		X = np.array(X)[..., np.newaxis, np.newaxis, :]
		T = np.array(T)[..., np.newaxis, np.newaxis]
		return (self.Vtot(X+self._d1M,T)+self.Vtot(X-self._d1M,T)-self.Vtot(X+self._d2M,T)-self.Vtot(X-self._d2M,T))/self.x_eps**2
		
	def entropyDensity(self, X, T):
		return -.5*(self.Vtot(X, T+self.T_eps)-self.Vtot(X, T-self.T_eps))/self.T_eps
		
	# MINIMIZATION AND TRANSITION ANALYSIS --------------------------------	
		
	def approxZeroTMin(self):
		# This should be overridden.
		return [np.ones(self.Ndim)*self.renormScaleSq**.5]
		
	def findMinimum(self, X=None, T=0.0):
		if X == None:
			X = self.approxZeroTMin()[0]
		return optimize.fmin(self.Vtot, X, args=(T,), disp=0)
		
	def findT0(self):
		"""
		This finds the temperature at which the high-T minimum disappears.
		"""
		X = self.findMinimum(np.zeros(self.Ndim), self.Tmax)
		f = lambda T: min(np.linalg.eigvalsh(self.d2V(X,T)))
		if f(0.0) > 0:
			# barrier at T = 0, or we Tmax isn't high enough.
			self.T0 = 0.0
		self.T0 = optimize.brentq(f, 0.0, self.Tmax)
		return self.T0
		
	def getPhases(self,startHigh=True,**tracingArgs):
		"""
		Uses the transitionFinder module to find the different phases as
		functions of temperature, and then finds where the phases overlap.
		Inputs:
		  startHigh - If True, self.transitions only shows transitions
		    from phases that can be reached from the high-temp phase.
		  **tracingArgs - extra parameters to pass to traceMultiMin
		"""
		T0 = self.T0 if self.T0 != None else self.findT0() # used as the characteristic T scale
		tstop = min(T0*5, self.Tmax)
		points = []
		for x0 in self.approxZeroTMin():
			points.append([x0,0.0])
		defaultArgs = {"tjump":T0*1e-2, "forbidCrit":self.forbidPhaseCrit, "dtabsMax":.005*T0, "dtmin":1e-5*T0, \
			"deltaX_target":100*self.x_eps, "deltaX_tol":1.2, "teps":self.T_eps, "xeps":self.x_eps}
		defaultArgs.update(tracingArgs)
		phases = transitionFinder.traceMultiMin(self.Vtot, points, 0.0, tstop, T0*1e-2, df=self.gradV, d2f=self.d2V, **defaultArgs)
		self.phases = phases
		transitionFinder.removeRedundantPhases(self.Vtot, phases, self.x_eps*1e-2, self.x_eps*10)
		self.transitions = transitionFinder.findTransitionRegions(self.Vtot, phases)
		
		if startHigh:
			startPhase = transitionFinder.getStartPhase(self.phases, self.Vtot)
			transitions = []+self.transitions # make a copy
			validPhases = [startPhase]
			validTrans = []
			while True:
				for i in xrange(len(transitions)):
					lowP, highP = transitions[i][2], transitions[i][3]
					if highP in validPhases:
						validPhases.append(lowP)
						validTrans.append(transitions[i])
						del transitions[i]
						break
				else: # went through the whole loop without hitting any valid phases
					break
			self.transitions = validTrans
		
	def calcTcTrans(self, startHigh = True):
		"""
		Calculates a dictionary of physical quantities at the critical temperature:
		- The critical temperature.
		- The total change in vev (abs val)
		- The high- and low-T vevs.
		- The latent heat density alpha (temperature times entropy density).
		"""
		if self.phases == None:
			self.getPhases()
		TcTrans = []
		for trans in self.transitions:
			lowP = self.phases[trans[2]]
			highP = self.phases[trans[3]]
			Tmin, Tmax = trans[0], trans[1]
			D = {}
			if Tmin == Tmax and Tmax >= lowP['T'][-1]:
				# second order transition
				D['dphi'] = 0.0
				D['low vev'] = D['high vev'] = highP['X'][0]
				D['alpha'] = 0.0
				D['Tcrit'] = Tmax
			elif Tmax == lowP['T'][-1] or Tmax == highP['T'][-1]:
				# The low-T phase is always lower. This isn't actually a phase transition.
				continue
			else:
				# first order transition
				Tcrit = Tmax
				xlow = np.array(interpolate.splev(Tcrit, lowP['tck']))
				xhigh = np.array(interpolate.splev(Tcrit, highP['tck']))
				D['dphi'] = np.sum((xlow-xhigh)**2)**.5
				D['low vev'], D['high vev'] = xlow, xhigh
				D['alpha'] = Tcrit * (self.entropyDensity(xhigh,Tcrit) - self.entropyDensity(xlow,Tcrit))
				D['Tcrit'] = Tcrit
			TcTrans.append(D)
		self.TcTrans = TcTrans
		return TcTrans
		
	def calcFullTrans(self, nuclCriterion = lambda S, T: S/(T+1e-100) - 140.0, overlapAngle = 45.0, dtBeta = 1e-2, \
			outTunnelObj = False, **tunnelParams):
		"""
		Calculates all of the phase transitions in the theory,
		starting from the hottest phase and working down.
		Inputs:
		  nuclCriterion, overlapAngle - see transitionFinder.fullTransitions.
		  dtBeta - The change in temperature (relative to the length of the phase)
		    to use in calculation of beta, or None if beta shouldn't be
			calculated.
		  outTunnelObj - Set to True if you want the full tunnel object
		    to be output.
		  tunnelParams - arguments to input into pathDeformation.criticalTunneling.
		Outputs a list of dictionaries containing
		  'tranType' - 1 or 2 for first or second order
		  'Tnuc' - The bubble nucleation temperature
		  'low vev', and 'high vev' - The vevs of the low and 
		    high-temp phases
		  'dphi' - The absolute difference between the two phases
		  'S3' - The Euclidean action
		  'alpha' - The energy density difference between the phases
		  'betaH' - T * d(S3/T)/dT
		  'tunnelObj' - The full tunneling object output from
		    pathDeformation.fullTunneling
		"""
		if self.phases == None:
			self.getPhases()
		
		import time
		t1 = time.time()
		fullTrans = transitionFinder.fullTransitions(self.Vtot, self.gradV, self.phases, self.transitions, \
								nuclCriterion=nuclCriterion, overlap=overlapAngle, **tunnelParams).out
		t2 = time.time()
		print "**** Time for finding fullTransitions:",t2-t1,"\n"
		
		self.TnTrans = []
		for tunnelObj, ilow, ihigh in fullTrans:
			D = {}
			self.TnTrans.append(D)
			D['tranType'] = tunnelObj.tranType
			D['Tnuc'] = Tnuc = tunnelObj.T
			if outTunnelObj:
				D['tunnelObj'] = tunnelObj
			if tunnelObj.tranType == 1:
				lowP,highP = self.phases[ilow],self.phases[ihigh]
				xlow = np.array(interpolate.splev(Tnuc, lowP['tck']))
				xhigh = np.array(interpolate.splev(Tnuc, highP['tck']))
				D['dphi'] = np.sum((xlow-xhigh)**2)**.5
				D['low vev'], D['high vev'] = xlow, xhigh
				D['S3'] = tunnelObj.findAction()
				D['alpha'] = Tnuc * (self.entropyDensity(xhigh,Tnuc) - self.entropyDensity(xlow,Tnuc)) \
						+ self.Vtot(xhigh,Tnuc) - self.Vtot(xlow,Tnuc)
				if dtBeta != None:
					print "Calculating beta/H"
					tmin = max(self.phases[ilow]['T'][ 0], self.phases[ihigh]['T'][ 0])
					tmax = min(self.phases[ilow]['T'][-1], self.phases[ihigh]['T'][-1])
					dt = (tmax-tmin)*dtBeta*.5
					if Tnuc + dt > tmax or Tnuc - dt < tmin:
						D['betaH'] = np.inf
					else:
						T1 = Tnuc+dt
						print "Finding rate just above Tnuc (T = "+str(T1)+")..."
						phiLow  = np.array(interpolate.splev(T1, self.phases[ilow ]['tck']))
						phiHigh = np.array(interpolate.splev(T1, self.phases[ihigh]['tck']))
						a1 = pathDeformation.fullTunneling((phiLow, phiHigh), lambda x: self.Vtot(x,T1), lambda x: self.gradV(x,T1), \
								alpha=2, **tunnelParams)
						a1.run(**tunnelParams)
						S1 = a1.findAction()
						T2 = Tnuc-dt
						print "\nFinding rate just below Tnuc (T = "+str(T2)+")..."
						phiLow  = np.array(interpolate.splev(T2, self.phases[ilow ]['tck']))
						phiHigh = np.array(interpolate.splev(T2, self.phases[ihigh]['tck']))
						a2 = pathDeformation.fullTunneling((phiLow, phiHigh), lambda x: self.Vtot(x,T2), lambda x: self.gradV(x,T2), \
								alpha=2, **tunnelParams)
						a2.run(**tunnelParams)
						S2 = a2.findAction()
						D['betaH'] = Tnuc * (S1/T1 - S2/T2)/(2*dt)
			if tunnelObj.tranType == 2:
				lowP, highP = self.phases[ilow], self.phases[ihigh]
				if ( Tnuc >= lowP['T'][-1] ): 
					# True second-order (as far as we can tell)
					D['low vev'] = D['high vev'] = np.array( highP['X'][0] )
					D['dphi'] = 0.0
					D['S3'] = np.nan
					D['alpha'] = 0.0
				else: 
					# unresolved transition
					xlow = np.array(interpolate.splev(Tnuc, lowP['tck']))
					xhigh = np.array(interpolate.splev(Tnuc, highP['tck']))
					D['low vev'], D['high vev'] = xlow, xhigh
					D['dphi'] = np.sum((xlow-xhigh)**2)**.5
					D['S3'] = np.nan
					D['alpha'] = Tnuc * (self.entropyDensity(xhigh,Tnuc) - self.entropyDensity(xlow,Tnuc)) \
							+ self.Vtot(xhigh,Tnuc) - self.Vtot(xlow,Tnuc)
				if dtBeta != None: 
					D['betaH'] = np.inf
			
		return self.TnTrans
		
		
	# PLOTTING ---------------------------------
		
	def plot2d(self, box, T=0, treelevel = False, offset = 0, xaxis = 0, yaxis = 1, n = 50, clevs = 200, cfrac=.3, **contourParams):
		"""
		Makes a countour plot of the potential.
		Inputs:
		  box - The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
		  T - The temperature
		  offset - A constant to add to all coordinates. Especially
		    helpful if Ndim > 2.
		  x,yaxis - the integers of the axes that we want to plot.
		  n - number of points evaluated in each direction.
		  clevs - number of contour levels to draw
		  cfrac - The cutoff used to avoid plotting really closely spaced contours
		    near max(V).
		  contourParams - Any extra parameters to be passed to pylab.contour.
		"""
		if self.pylab == None:
			print "You have to set generic_potential.pylab first"
			return
		xmin,xmax,ymin,ymax = box
		X = np.linspace(xmin, xmax, n).reshape(n,1)*np.ones((1,n))
		Y = np.linspace(ymin, ymax, n).reshape(1,n)*np.ones((n,1))
		XY = np.zeros((n,n,self.Ndim))
		XY[...,xaxis], XY[...,yaxis] = X,Y
		Z = self.V0(XY) if treelevel else self.Vtot(XY,T)
		minZ, maxZ = min(Z.ravel()), max(Z.ravel())
		N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
		self.pylab.contour(X,Y,Z, N, **contourParams)
		self.pylab.axis(box)
		
	def plot1d(self, x1, x2, T=0, treelevel = False, subtract = True, n = 500, **plotParams):
		if self.pylab == None:
			print "You have to set generic_potential.pylab first"
			return
		if self.Ndim == 1:
			x = np.linspace(x1,x2,n)
			X = x[:,np.newaxis]
		else:
			dX = np.array(x2)-np.array(x1)
			X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
			x = np.linspace(0,1,n)*np.sum(dX**2)**.5
		if treelevel:
			y = self.V0(X) - self.V0(X*0) if subtract else self.V0(X)
		else:
			y = self.DVtot(X,T) if subtract else self.Vtot(X, T)
		self.pylab.plot(x,y, **plotParams)
		
	def plotPhasesV(self, useDV = True, npoints = 100, **plotArgs):
		if self.pylab == None:
			print "You have to set generic_potential.pylab first"
			return
		if self.phases == None:
			self.getPhases()
		for p in self.phases:
			t = np.linspace(p["T"][0], p["T"][-1], npoints)
			phi = np.array(interpolate.splev(t, p['tck'])).T
			V = self.DVtot(phi,t) if useDV else self.Vtot(phi,t)
			self.pylab.plot(t,V,**plotArgs)

	def plotPhasesPhi(self, npoints = 100, **plotArgs):
		if self.pylab == None:
			print "You have to set generic_potential.pylab first"
			return
		if self.phases == None:
			self.getPhases()
		for p in self.phases:
			t = np.linspace(p["T"][0], p["T"][-1], npoints)
			phi = np.array(interpolate.splev(t, p['tck'])).T
			phi_mag = np.sum( phi**2, -1 )**.5
			self.pylab.plot(t,phi_mag,**plotArgs)

# END GENERIC_POTENTIAL CLASS ------------------


# FUNCTIONS ON LISTS OF MODEL INSTANCES ---------------

def funcOnModels(f, models):
	"""
	If you have a big array of models, this function allows you
	to extract big arrays of model outputs. For example, suppose
	that you have a 2x5x20 nested list of models and you want to
	find the last critical temperature of each model. Then use
		Tcrit = funcOnModels(lambda A: A.TcTrans[-1]['Tcrit'], models).
	Tcrit will be a numpy array with shape (2,5,20).
	"""
	M = []
	for a in models:
		if isinstance(a,list) or isinstance(a,tuple):
			M.append(funcOnModels(f, a))
		else:
			try:
				M.append(f(a))
			except:
				M.append(np.nan)
	return np.array(M)
	
def linkTransitions(models, critTrans = True):
	"""
	This function will take a list of models that represent the
	variation of some continuous model parameter, and output several
	lists of phase transitions such that all of the transitions
	in a single list roughly correspond to each other.
	"""
	allTrans = []
	for model in models:
		allTrans.append(model.TcTrans if critTrans else model.TnTrans)
	# allTrans is now a list of lists of transitions. We want to rearrange each sublist so that it matches the previous sublist.
	for j in xrange(len(allTrans)-1):
		trans1, trans2 = allTrans[j], allTrans[j+1]
		if trans1 == None: trans1 = []
		if trans2 == None: trans2 = []
		# First, clear the transiction dictionaries of link information
		for t in trans1+trans2:
			if t != None:
				t['link'] = None
				t['diff'] = np.inf
		for i1 in xrange(len(trans1)):
			t1 = trans1[i1] #t1 and t2 are individual transition dictionaries
			if t1 == None: continue
			for i2 in xrange(len(trans2)):
				t2 = trans2[i2] #t1 and t2 are individual transition dictionaries
				if t2 == None: continue
				# See if t1 and t2 are each other's closest match
				diff = np.sum((t1['low vev']-t2['low vev'])**2)**.5 + np.sum((t1['high vev']-t2['high vev'])**2)**.5
				if diff < t1['diff'] and diff < t2['diff']:
					t1['diff'] = t2['diff'] = diff
					t1['link'], t2['link'] = i2, i1
		for i2 in xrange(len(trans2)):
			t2 = trans2[i2]
			if t2 != None and t2['link'] != None and trans1[t2['link']]['link'] != i2:
				t2['link'] = None # doesn't link back.
		# Now each transition in tran2 is linked to its closest match in tran1, or None if it has no match
		newTrans = [None]*len(trans1)
		for t2 in trans2:
			if t2 == None:
				continue
			elif t2['link'] == None:
				newTrans.append(t2) # This transition doesn't match up with anything.
			else:
				newTrans[t2['link']] = t2
		allTrans[j+1] = newTrans
	# Almost done. Just need to clean up the transitions and make sure that the allTrans list is rectangular.
	for trans in allTrans:
		for t in trans:
			if t != None:
				del t['link']
				del t['diff']
	n = len(allTrans[-1])
	for trans in allTrans:
		while len(trans) < n:
			trans.append(None)
	# Finally, transpose allTrans:
	allTrans2 = []
	for i in xrange(len(allTrans[0])):
		allTrans2.append([])
		for j in xrange(len(allTrans)):
			allTrans2[-1].append(allTrans[j][i])
	return allTrans2

				
		

		

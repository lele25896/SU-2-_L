#
#  FieldTunneling1D.py
#  

import numpy as np
from scipy import optimize, integrate, special, interpolate

__version__ = "1.0.1"

class bubbleProfile:
	"""
	This class contains all of the code for finding the bubble 
	profile via the overshoot/undershoot method.
	"""
#--------- --------- --------- --------- --------- --------- --------- ---------  <-- 80 chars long
	def __init__(self, phi_absMin, phi_metaMin, V, dV = None, d2V = None, dV_eps = 1e-3, alpha=2, x_inf=1e8):
		"""
		Inputs:
		  phi_absMin - The absolute minimum of the potential. The vev at the
		    center of the bubblewill be very close to this value.
		  phi_metaMin - The metastable minimum. This is the vev outside of the 
		    bubble.
		  V, dV - The potential V(phi) and dV/dphi, both as functions of phi.
		  d2V - The second derivative of V evaluated at the metastable minimum.
		  dV_eps - The small change in phi (relative to phi_absMin-phi_metaMin)
		    to use for numerical calculations of derivatives.
		  alpha - The number of spacetime dimensions minus 1. Should be 2 for
		    tunneling at finite temperature, 3 at zero temperature.
		  x_inf - The largest x we're willing to use before we just say that
		    the tunneling happens at r = infinity.
		"""
		self.phi_absMin, self.phi_metaMin = phi_absMin, phi_metaMin
		self.V = V
		self.dV_eps = dV_eps * abs(phi_absMin - phi_metaMin)
		if (dV == None):
			self.dV = lambda phi: .5*(V(phi+self.dV_eps)-V(phi-self.dV_eps))/self.dV_eps
		else:
			self.dV = dV
		if (d2V == None):
			self.d2V = lambda phi: .5*(self.dV(phi+self.dV_eps)-self.dV(phi-self.dV_eps))/self.dV_eps
			self.d2V = self.d2V(phi_absMin)
		else:
			self.d2V = d2V
		self.alpha = alpha
		self.x_inf = x_inf
		
		# Need to find the approximate radial scale of the problem. If the potential
		# are degenerate and it's a pure quartic potential, then the solution is 
		# phi(r) = (1/2) (phi_min1-phi_min2)*tanh(r/rscale),
		# where rscale is given by
		phiavg = (phi_absMin+phi_metaMin)*.5
		Vavg = ( V(phi_metaMin)+V(phi_absMin) )*.5
		Vmid = V(phiavg)
		self.rscale = np.abs(phi_absMin-phi_metaMin)/np.sqrt(8*np.abs(Vavg-Vmid))
		
#--------- --------- --------- --------- --------- --------- --------- ---------  <-- 80 chars long
	def initialConditions(self, x, rmin, thinCutoff = 1e-2):
		"""
		This method returns the initial conditions (r0, y0, dy0/dphi) for some
		value x, where phi(r=0) = phi_absMin + exp(-x)*(phi_metaMin-phi_absMin).
		The value r0 is chosen such that 
		     |phi(r=r0)-phi_absMin| >= thinCutoff*|phi_metaMin-phi_absMin|.
		For thick walls, r0 = 0 and dphi0 = 0.
		rmin is the smallest initial r that we'll return.
		"""
		# If r -> infty, phi(r) = phi_absMin - (phi_absMin - phi0)*exp(d2V^.5 * r), where phi0 = phi(r=0) and then we
		# add an infinite constant to r.
		if x > self.x_inf: #x == np.inf:
			r0 = np.inf
			phi0 = self.phi_absMin + (self.phi_metaMin - self.phi_absMin)*thinCutoff
			dphi0 = (phi0 - self.phi_absMin) * self.d2V**.5
			return r0, phi0, dphi0

	#	if abs(phi0-phi_absMin) >= thinCutoff*abs(phi_absMin-phi_metaMin):
		if np.exp(-x) >= thinCutoff: # Thick wall.
			r0 = rmin
			
		#	phi0 = self.phi_absMin + np.exp(-x)*(self.phi_metaMin-self.phi_absMin)
		#	return r0, phi0, 0.0
			
		else: # thin wall
		#	Dy = lambda r: self.exactSolution(r)-thinCutoff*np.exp(x-r) # where thinCutoff*exp(x) = (phi(r=r0)-phi_absMin)/(phi(r=0)-phi_absMin)
			Dy = lambda r: np.log(self.exactSolution(r)/thinCutoff)-(x-r*self.d2V**.5)
			rmax = (x+np.log(thinCutoff))/self.d2V**.5
			ymax = Dy(rmax)
			while ymax < 0:
				rmax *= 1.5
				ymax = Dy(rmax)
			ymin = ymax
			rmin = rmax
			while ymin > 0:
				rmin *= .5
				ymin = Dy(rmin)
			if not np.isfinite(ymax) or not np.isfinite(ymin): # or rmin*self.d2V**.5 > 1e10:
				r0 = np.inf
				phi0 = self.phi_absMin + (self.phi_metaMin - self.phi_absMin)*thinCutoff
				dphi0 = (phi0 - self.phi_absMin) * self.d2V**.5
				return r0, phi0, dphi0
		#	print "rmax", rmax
			r0 = optimize.brentq(Dy, rmin, rmax)
	#	phi0 = self.exactSolution(r0) * np.exp(-x+r0)*(self.phi_metaMin-self.phi_absMin) + self.phi_absMin
	#	dphi0 = self.exactSolution(r0, nderiv=1) * np.exp(-x+r0)*(self.phi_metaMin-self.phi_absMin)
		phi0 = self.exactSolution(r0)*np.exp(r0*self.d2V**.5-x)*(self.phi_metaMin-self.phi_absMin) + self.phi_absMin
		dphi0 = self.exactSolution(r0, nderiv=1)*np.exp(r0*self.d2V**.5-x)*(self.phi_metaMin-self.phi_absMin)
		return r0, phi0, dphi0
		
	def exactSolution(self, r, nderiv = 0):
		"""
		This method returns 
		  (phi(r)-phi_absMin)/(phi(0)-phi_absMin) * exp(-r*sqrt(d2V)),
		assuming that phi is small enough such that the potential can
		be approximated as dV(phi) = d2V * (phi-phi_absMin).
		The exp(-r) part is to keep the value finite for large r.
		"""
		# For phi close to phi_absMin, we can estimate the potential to just be V(phi) = (1/2)d2V/dphi2 * (phi-phi_absMin)^2,
		# which we can solve with Bessel functions. See http://eqworld.ipmnet.ru/en/solutions/ode/ode0207.pdf
		# Note that the absolute value sign at that page is wrong. It should be just be I_{-nu}.
		# For alpha=0 the solution is just phi(r) = phi_absMin - (phi_absMin - phi0)*cosh(d2V^.5 * r)
		# For alpha=2 the solution is y(r) = sinh(r*d2V**.5)/(r*d2V**.5)
		# We could also write the solution in an infinite series, which is much slower:
		#	y(r) = sum( a_2j * x**(2j) ), a_{j+2} = a_j * d2V/((j+2)(j+1+alpha))
		if nderiv != 0 and nderiv != 1:
			raise Exception, "nderiv must be either 0 or 1 in exactSolution"
		b_ = np.sqrt(self.d2V)
				
		if self.alpha == 0:
			if nderiv == 0:
				y = .5*(1+np.exp(-2*r*b_))
			if nderiv == 1:
				y = .5*(1-2*b_*np.exp(-2*r*b_))
		elif self.alpha == 2:
			if nderiv == 0:
				y = .5*(1-np.exp(-2*r*b_))/(r*b_)
			if nderiv == 1:
				if r*b_ > 5:
					y = .5*(1+np.exp(-2*r*b_))/(r) - 0.5*(1-np.exp(-2*r*b_))/(r*r*b_)
				else: # The above doesn't do a good job of cancelling to zero when r*b is small
					y = ( np.cosh(r*b_)/r - np.sinh(r*b_)/(r*r*b_) ) * np.exp(-r*b_)
		else:
			nu_ = .5*(self.alpha-1)
			norm_ = 2**nu_*special.gamma(nu_+1)*b_**(-nu_) # multiply by this so that phi(0) = 1.
			if nderiv == 0:
				y = r**(-nu_) * special.ive(nu_, r*b_) * norm_
			elif nderiv == 1:
				y = r**(-nu_) * .5*(-2*nu_*special.ive(nu_,r*b_)/r \
												+ b_*special.ive(nu_-1,r*b_) + b_*special.ive(nu_+1,r*b_) ) * norm_
		return y
		
		
#--------- --------- --------- --------- --------- --------- --------- ---------  <-- 80 chars long
	def integrateProfile(self, r0, phi0, dphi0, dr, epsfrac, epsabs, rmax,drmin):
		"""
		This method integrates forward from some initial conditions r0, phi0 and
		dphi0 until either phi < phi_metaMin, dphi > 0 (the signs of the
		inequalities are reversed if phi_metaMin > phi_absMin), or until the
		integral converges towards phi = dphi = 0. The function returns r_final,
		phi_final, dphi_final, and a dictionary of return codes, which contains
		0 for convergence, -1 for error, +1 for undershoot, and +2 for
		overshoot.
		Inputs:
		  r0, phi0, dphi0 - the starting values for integration. If r0 is an
		    array, then it outputs an array of points at each value of r0.
		  dr - the starting stepsize.
		  epsfrac, epsabs - the error tolerances.
		  rmax, drmin - the largest change in r and smallest stepsize allowed 
		    before we return an error.
		"""
		try:
			N = len(r0)
			R, r0 = r0, r0[0]
			Phi, dPhi = np.zeros_like(R), np.zeros_like(R)
			Phi[0], dPhi[0] = phi0, dphi0
		except:
			R = None
		y0 = np.array([phi0, dphi0])
		dY = lambda y, r: np.array([y[1], self.dV(y[0])-self.alpha*y[1]/r])
		dydr0 = dY(y0, r0)
		ysign = np.sign(phi0-self.phi_metaMin) # positive means we're heading down, negative means heading up.
		err = {}
		rmax += r0
		
		i = 1
		while True:
			dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, epsfrac, epsabs)		
			r1 = r0 + dr
			y1 = y0 + dy
			dydr1 = dY(y1,r1)
			# First, check to see if we need to fill the array
			if (R != None and (r0 < R[i] <= r1)):
				# y = a0+a1*x+a2*x^2+a3*x^3+a4*x^4+a5*x^5, x = (r-r0)/dr so that x=0 at r0 and x=1 at r1 = r0+dr
				a0,a1,a2, z,dz,d2z = y0[0],dr*y0[1],.5*dr*dr*dydr0[1], y1[0],dr*y1[1],dr*dr*dydr1[1]
				b1,b2,b3 = z-a0-a1-a2, dz-a1-2*a2, d2z-2*a2
				a5 = .5*b3 - 3*b2 + 6*b1
				a4 = b2 - 3*b1 - 2*a5
				a3 = b1 - a4 - a5
				df = lambda x: a1+x*(2*a2+x*(3*a3+x*(4*a4+x*5*a5))) # = dphi/dx
				f = lambda x: a0+x*(a1+x*(a2+x*(a3+x*(a4+x*a5))))
				while (r0 < R[i] <= r1):
					x = (R[i]-r0)/dr
					Phi[i] = f(x)
					dPhi[i] = df(x)/dr
					i += 1
					if i >= len(R):
						return Phi, dPhi
				continue
				
			# Otherwise, check for completion
			if (r1 > rmax):
				r,y = r1,y1
				err[-1] = "Integration error: r > rmax."
				break
			elif (dr < drmin):
				r,y = r1,y1
				err[-1] = "Integration error: dr < drmin."
				break
			elif( (abs(y1 - np.array([self.phi_metaMin,0])) < 3*epsabs).all() ):
				r,y = r1,y1
				err[0] = "No error. Integration converged."
				break
				
			elif( y1[1]*ysign > 0 or (y1[0]-self.phi_metaMin)*ysign < 0 ):
				# y = a0+a1*x+a2*x^2+a3*x^3+a4*x^4+a5*x^5, x = (r-r0)/dr so that x=0 at r0 and x=1 at r1 = r0+dr
				a0,a1,a2, z,dz,d2z = y0[0],dr*y0[1],.5*dr*dr*dydr0[1], y1[0],dr*y1[1],dr*dr*dydr1[1]
				b1,b2,b3 = z-a0-a1-a2, dz-a1-2*a2, d2z-2*a2
				a5 = .5*b3 - 3*b2 + 6*b1
				a4 = b2 - 3*b1 - 2*a5
				a3 = b1 - a4 - a5
				df = lambda x: a1+x*(2*a2+x*(3*a3+x*(4*a4+x*5*a5))) # = dphi/dx
				f = lambda x: a0+x*(a1+x*(a2+x*(a3+x*(a4+x*a5)))) - self.phi_metaMin
			
				if(y1[1]*ysign > 0): # undershoot
					# Extrapolate to where dphi(r) = 0
					x = optimize.brentq(df, 0, 1 )
					err[+1] = "Undershoot."
				else: # overshoot
					# Extrapolate to where phi(r) = phi_metaMin
					x = optimize.brentq(f, 0, 1 )
					err[+2] = "Overshoot."
				r = r0 + dr*x
				phi = f(x) + self.phi_metaMin
				dphi = df(x)/dr
				y = np.array([phi, dphi])
				break
			# move up the initial conditions
			r0,y0,dydr0 = r1,y1,dydr1
			dr = drnext
		# Check convergence for a second time. The extrapolation in overshoot/undershoot might have gotten us within acceptable error.
		if (abs(y - np.array([self.phi_metaMin,0])) < 3*epsabs).all():
			err[0] = "No error. Integration converged."
		return r,y[0],y[1],err
		
#--------- --------- --------- --------- --------- --------- --------- ---------  <-- 80 chars long
	def findProfile(self, xguess = 1.0, xtol = 1e-4, phitol = 1e-4, thinCutoff=.01, npoints=500, verbose = False):
		"""
		This method finds the bubble profile using the overshoot/undershoot
		method. It calls integrateProfile() many times, adjusting the initial
		conditions until they're just right.
		Inputs:
		  xguess - initial guess for x, where 
		    phi(r=0) = phi_absMin + exp(-x)*(phi_metaMin-phi_absMin)
		  xtol - the target accuracy of x
		  phitol - the target accuracy for convergence in phi(r)
		  thinCutoff - The smallest fractional difference from phi_absMin that
		    the initial condition is allowed to be.
		  npoints - The length of the return arrays.
		Returns r, phi(r), dphi(r)
		If V(phi_absMin) >= V(phi_metaMin), then the wall should be very thin
		and we should have r = inf. Instead, this function outputs a negative
		r in these cases so that one can still plot it.
		"""
		xmin = 0.0
		xmax = np.inf
		x = xguess
		xincrease = 5.0 # The relative amount to increase x by if there is no upper bound.
		dr = max(1e-2, phitol*10)*self.rscale # The starting stepsize.
		drmin = phitol*self.rscale # the smallest stepsize we allow before spitting out an error
		rmax = 1e3*self.rscale # The farthest out we'll integrate before spitting out an error
		rmin = 1e-4*self.rscale # The smallest starting value of r0
		deltaPhi = np.abs(self.phi_absMin-self.phi_metaMin)
		epsabs = np.array([deltaPhi, deltaPhi/self.rscale])*phitol
		epsfrac = np.array([1,1]) * phitol
		# First see if the metastable minimum is in fact metastable.
		if self.V(self.phi_metaMin) <= self.V(self.phi_absMin):
			r0, phi0, dphi0 = self.initialConditions(np.inf, rmin, thinCutoff)
			alpha = self.alpha
			self.alpha = 0
			r0 = 1.0
			rf, phif, dphif, interr = self.integrateProfile(r0, phi0, dphi0, dr, epsfrac, epsabs, rmax, drmin)
			r0, rf = r0 - .5*(r0+rf), rf - .5*(r0+rf)
		else:
			while True:
				# First, make sure that x is at least big enough so V(phi(r=0)) < V(phi_metaMin)
				phi_r0 = np.exp(-x)*(self.phi_metaMin-self.phi_absMin) + self.phi_absMin
				if self.V(self.phi_metaMin) <= self.V(phi_r0): # x is too low.
					xmin = x
					x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)
					continue
				r0, phi0, dphi0 = self.initialConditions(x, rmin, thinCutoff)
				alpha = self.alpha
				if r0 == np.inf: # We want to effectively take out the friction term.
					r0 = 1
					self.alpha=0
					x = np.inf
				rf, phif, dphif, interr = self.integrateProfile(r0, phi0, dphi0, dr, epsfrac, epsabs, rmax, drmin)
				if x == np.inf:
					break
				# Check for overshoot, undershoot
				if 0 in interr: # Converged
					break
				elif 1 in interr: # undershoot. x is too low
					xmin = x
					x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)
				elif 2 in interr: # overshoot. x is too high
					xmax = x
					x = .5*(xmin+xmax)
				elif -1 in interr:
					raise Exception, interr[-1]
				# Check if we've reached xtol
				if (xmax-xmin) < xtol:
					break
					
		# Integrate a second time, this time getting the points along the way
		r = np.linspace(r0, rf, npoints)
		phi, dphi = self.integrateProfile(r, phi0, dphi0, dr, epsfrac, epsabs, rmax, drmin)

		self.alpha = alpha
		
		return r, phi, dphi
	
	def evenlySpacedPhi(self, phi, dphi, npoints = 100, k=1,fixAbs = True):
		"""
		This method takes phi and dphi as input, which will probably
		come from the output of findProfile(), and returns a different
		set of arrays phi and dphi such that phi is linearly spaced
		(instead of r).
		Other inputs:
		  npoints - number of points on output
		  k - degree of spline fitting. k=1 means linear interpolation.
		  fixAbs - If true, make phi go all the way to phi_absMin.
		"""
		if fixAbs == True:
			phi = np.append(self.phi_absMin, np.append(phi, self.phi_metaMin))
			dphi = np.append(0.0, np.append(dphi, 0.0))
		else:
			phi = np.append(phi, self.phi_metaMin)
			dphi = np.append(dphi, 0.0)
		# Make sure that phi is increasing everywhere (this is uglier than it ought to be)
		i = monotonicIndices(phi)
		# Now do the interpolation
		tck = interpolate.splrep(phi[i], dphi[i], k=k)
		if fixAbs:
			p = np.linspace(self.phi_absMin, self.phi_metaMin, npoints)
		else:
			p = np.linspace(phi[i][0], self.phi_metaMin, npoints)
		dp = interpolate.splev(p, tck)
		return p, dp
		
def monotonicIndices(x):
	"""
	This is a helper function that returns the indices of x
	such that x[i] is purely increasing.
	"""
	x = np.array(x)
	if x[0] > x[-1]:
		x = x[::-1]
		reversed = True
	else:
		reversed = False
	I = [0]
	for i in xrange(1, len(x)-1):
		if x[i] > x[I[-1]] and x[i] < x[-1]:
			I.append(i)
	I.append(len(x)-1)
	if reversed:
		return len(x)-1-np.array(I)
	else:
		return np.array(I)
	


# ---------------
				
# Now to write a runge kutta integrator...		
def rkck(y,dydt,t,f,dt):
	"""This function takes one fifth-order Cash-Karp Rnge-Kutta step to advance the solution
	y(t) to y(t+dt) given a function dy/dt = f(y,t). It returns the new dy and an estimation of the error.
	Note that the variable dydt is just f(y,t). This should be input so that we don't have to calculate
	it more than necessary.
	This one is readable, but slower (barely)."""
	a2=0.2;a3=0.3;a4=0.6;a5=1.0;a6=0.875;b21=0.2
	b31=3.0/40.0;b32=9.0/40.0;b41=0.3;b42 = -0.9;b43=1.2;
	b51 = -11.0/54.0; b52=2.5;b53 = -70.0/27.0;b54=35.0/27.0;
	b61=1631.0/55296.0;b62=175.0/512.0;b63=575.0/13824.0;
	b64=44275.0/110592.0;b65=253.0/4096.0;c1=37.0/378.0;
	c3=250.0/621.0;c4=125.0/594.0;c6=512.0/1771.0;
	dc5 = -277.00/14336.0;
	dc1=c1-2825.0/27648.0;dc3=c3-18575.0/48384.0;
	dc4=c4-13525.0/55296.0;dc6=c6-0.25
	ytemp = y+b21*dt*dydt
	ak2 = f(ytemp, t+a2*dt)
	ytemp = y+dt*(b31*dydt+b32*ak2)
	ak3 = f(ytemp, t+a3*dt)
	ytemp = y+dt*(b41*dydt+b42*ak2+b43*ak3)
	ak4 = f(ytemp, t+a4*dt)
	ytemp = y + dt*(b51*dydt+b52*ak2+b53*ak3+b54*ak4)
	ak5 = f(ytemp, t+a5*dt)
	ytemp = y + dt*(b61*dydt+b62*ak2+b63*ak3+b64*ak4+b65*ak5)
	ak6 = f(ytemp, t+a6*dt)
	dyout = dt*(c1*dydt+c3*ak3+c4*ak4+c6*ak6)
	yerr = dt*(dc1*dydt+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)
	return dyout, yerr

def rkqs(y,dydt,t,f, dt_try, epsfrac, epsabs):
	"""This function takes the a step using rkck, while making sure that it's within the error.
	epsfrac is the acceptable fractional error, and epsabs is the acceptable absolute error. We 
	want the error to be at least as small as the LARGER of these two. These can be either numbers
	or numpy arrays."""
	dt = dt_try
	err = ''
	while True:
		dy,yerr = rkck(y,dydt,t,f,dt)
		#errmax = min( np.max(abs(yerr/epsabs)), np.max(abs((yerr/y)/epsfrac)) )
		errmax = np.nan_to_num(np.max( np.min([abs(yerr/epsabs), abs((yerr/y)/epsfrac)],axis=0) ))
		if(errmax < 1.0):
			break # Step succeeded
		dttemp = 0.9*dt*errmax**-.25
		dt = max(dttemp,dt*.1) if dt > 0 else min(dttemp,dt*.1)
		if(t+dt==t):
			dt = 0; break	# step failed, the stepsize got too small
	if errmax > 1.89e-4:
		dtnext = 0.9 * dt * errmax**-.2
	else:
		dtnext = 5*dt
	return dy, dt, dtnext

			
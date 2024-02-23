
import numpy as np
import pylab
from scipy import optimize, interpolate

from cosmoTransitions.generic_potential import generic_potential
from cosmoTransitions import tunneling1D, pathDeformation

__version__ = "1.0.1"
		
v2 = 246.**2
	
class model1(generic_potential):
	# The init method is called by the generic_potential class, after it already does some of
	# its own initialization in the default __init__() method. This is necessary for all subclasses
	# to implement.
	def init(self,m1=120.,m2=50.,mu=25.,Y1=.1,Y2=.15,n=30,forbidNegX=True):
		"""
		  m1 - tree-level mass of first singlet when mu = 0.
		  m2 - tree-level mass of second singlet when mu = 0.
		  mu - mass coefficient for the mixing term.
		  Y1 - Yukawa coupling to the two scalars individually
		  Y2 - Coupling to the two scalars together: m^2 = Y2*s1*s2
		  n - degrees of freedom of the boson that is coupling.
		  forbidNegX - Keeps the phases from going to (very) negative x.
		"""
		# This first line is absolutely essential in all subclasses. It specifies the number of
		# field-dimensions in the theory.
		self.Ndim = 2
		
		# Setting self.pylab is only necessary for plotting purposes. It can also be set from
		# outside of the class (i.e., anInstance = model1(); anInstance.pylab = pylab)
		self.pylab = pylab
		
		# self.renormScaleSq is the renormalization scale used in the Coleman-Weinberg potential.
		self.renormScaleSq = v2
		
		# This next block sets all of the parameters that go into the potential and the masses.
		# This will obviously need to be changed for different models.
		self.l1 = .5*m1**2/v2
		self.l2 = .5*m2**2/v2
		self.mu2 = mu**2		
		self.Y1, self.Y2 = Y1, Y2
		self.n = n
		
		# forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry in the theory
		# and you don't want to double-count all of the phases. In this case, we're throwing away
		# all phases whose zeroth (since python starts arrays at 0) field component of the vev
		# goes below -5. Note that we don't want to set this to just going below zero, since we are
		# interested in phases with vevs exactly at 0, and floating point numbers will never be 
		# accurate enough to ensure that these aren't slightly negative.
		if forbidNegX:
			self.forbidPhaseCrit = lambda X: (np.array([X])[...,0] < -5.0).any()
		
						
	def V0(self, X):
		# This method defines the tree-level potential. It should generally be subclassed.
		# (You could also subclass Vtot() directly, and put in all of quantum corrections yourself).
		
		# X is the input field array. It is helpful to ensure that it is a numpy array before splitting
		# it into its components.
		X = np.array(X)
		# x and y are the two fields that make up the input. The array should always be defined such
		# that the very last axis contains the different fields, hence the ellipses.
		# (For example, X can be an array of N two dimensional points and have shape (N,2), but it
		# should NOT be a series of two arrays of length N and have shape (2,N).)
		x,y = X[...,0], X[...,1]
		r = .25*self.l1*(x*x-v2)**2 + .25*self.l2*(y*y-v2)**2 - self.mu2*x*y
		return r
		
	def boson_massSq(self, X, T):
		X = np.array(X)
		x,y = X[...,0], X[...,1]
		
		# We need to define the field-dependnet boson masses. This is obviously model-dependent.
		# Note that these can also include temperature-dependent corrections.
		a = self.l1*(3*x*x - v2)
		b = self.l2*(3*y*y - v2)
		A = .5*(a+b)
		B = np.sqrt(.25*(a-b)**2 + self.mu2**2)
		mb = self.Y1*(x*x+y*y) + self.Y2*x*y
		M = np.array([A+B, A-B, mb])
		
		# At this point, we have an array of boson masses, but each entry might be an array itself.
		# This happens if the input X is an array of points. The generic_potential class requires
		# that the output of this function have the different masses lie along the last axis,
		# just like the different fields lie along the last axis of X, so we need to reorder the
		# axes. The next line does this, and should probably be included in all subclasses.
		M = np.rollaxis(M, 0, len(M.shape))
		
		# The number of degrees of freedom for the masses. This should be a one-dimensional array
		# with the same number of entries as there are masses.
		dof = np.array([1,   1,   self.n])
		
		# c is a constant for each particle used in the Coleman-Weinberg potential. It equals 1.5
		# for all scalars and the longitudinal polarizations of the gauge bosons, and 0.5 for 
		# transverse gauge bosons.
		c = np.array([1.5, 1.5, 1.5])
		
		return M, dof, c
		
	def approxZeroTMin(self):
		# There are generically two minima at zero temperature in this model, and we want to include both of them.
		v = v2**.5
		return [np.array([v,v]), np.array([v,-v])]
		
		
class model1_3d(model1):
	"""
	This model is the same as model1, except that it adds an extra field dimension.
	This dimension always has a minimum at zero. It adds a coupling to this dimension
	in boson_massSq just so that all of the values in d2V are comparable.
	All coordinates are rotated such that the minimum should lie along the line y = z.
	"""
	def init(self, *a, **b):
		model1.init(self, *a, **b)
		self.Ndim = 3
		
	def V0(self, X):
		X = np.array(X)
		y,z = X[...,1], X[...,2]
		X[...,1],z = (y+z)/2**.5, (y-z)/2**.5
		return model1.V0(self, X) + v2*z*z
		
	def boson_massSq(self, X, T):
		X = np.array(X)
		x,y,z = X[...,0],X[...,1], X[...,2]
		y,z = (y+z)/2**.5, (y-z)/2**.5
		a = self.l1*(3*x*x - v2)
		b = self.l2*(3*y*y - v2)
		A = .5*(a+b)
		B = np.sqrt(.25*(a-b)**2 + self.mu2**2)
		mb = self.Y1*(x*x+y*y+z*z) + self.Y2*x*y
		M = np.array([A+B, A-B, mb])
		M = np.rollaxis(M, 0, len(M.shape)) # the different masses now lie along the last axis
		return M, np.array([1,   1,   self.n]), 1.5

	def approxZeroTMin(self):
		v = v2**.5
		return [np.array([v,v/2**.5,v/2**.5]), np.array([v,-v/2**.5,-v/2**.5])]


class model1_4d(model1):
	"""
	Similar to model1_3d, but adds another field.
	"""
	def init(self, *a, **b):
		model1.init(self, *a, **b)
		self.Ndim = 4
		
	def V0(self, X):
		X = np.array(X)
		y,z,w = X[...,1], X[...,2], X[...,3]
		y,z = (y+z)/2**.5, (y-z)/2**.5
		y,w = (y+w)/2**.5, (y-w)/2**.5
		X[...,1] = y
		return model1.V0(self, X) + v2*(z*z + w*w)
		
	def boson_massSq(self, X, T):
		X = np.array(X)
		x,y,z,w = X[...,0],X[...,1], X[...,2], X[...,3]
		y,z = (y+z)/2**.5, (y-z)/2**.5
		y,w = (y+w)/2**.5, (y-w)/2**.5
		a = self.l1*(3*x*x - v2)
		b = self.l2*(3*y*y - v2)
		A = .5*(a+b)
		B = np.sqrt(.25*(a-b)**2 + self.mu2**2)
		mb = self.Y1*(x*x+y*y+z*z+w*w) + self.Y2*x*y
		M = np.array([A+B, A-B, mb])
		M = np.rollaxis(M, 0, len(M.shape)) # the different masses now lie along the last axis
		return M, np.array([1,   1,   self.n]), 1.5

	def approxZeroTMin(self):
		v = v2**.5
		return [np.array([v,v/2.,v/2**.5,v/2.]), np.array([v,-v/2.,-v/2**.5,-v/2.])]
		
class model2_1d(generic_potential):
	"""
	Very simple model to test transitions in 1 dimension.
	"""
	def init(self,mh=120.,Y=1.0,n=30,forbidNegX=True):
		"""
		  mh - tree-level mass of the scalar field.
		  Y - Yukawa coupling.
		  n - degrees of freedom of the boson that is coupling.
		  forbidNegX - Keeps the phases from going to (very) negative x.
		"""
		self.Ndim = 1
		self.pylab = pylab
		self.renormScaleSq = v2
		
		self.lmda = .5*mh**2/v2		
		self.Y = Y
		self.n = n
		
		if forbidNegX:
			self.forbidPhaseCrit = lambda X: (np.array([X])[...,0] < -5.0).any()
		
						
	def V0(self, X):
		X = np.array(X)
		x = X[...,0]
		r = .25*self.lmda*(x*x-v2)**2
		return r
		
	def boson_massSq(self, X, T):
		X = np.array(X)
		x = X[...,0]
		mh = self.lmda*(3*x*x - v2)
		mx = (self.Y*x)**2
		M = np.array([mh, mx])
		M = np.rollaxis(M, 0, len(M.shape)) # the different masses now lie along the last axis
		return M, np.array([1, self.n]), 1.5
		
	def approxZeroTMin(self):
		v = v2**.5
		return [np.array([v])]

		
class testPotential2D:
	def __init__(self, barrier=1.0, tilt1 = 0.8, tilt2 = 0.0, x0 = 1.0, y0 = 1.0):
		self.tilt1, self.tilt2 = tilt1, tilt2
		self.x0, self.y0 = x0, y0
		self.c = (1.0-barrier) * ((1+tilt1)*x0**2 + (1-tilt1)*y0**2)
		self.metaMin = np.array([0.0, 0.0])
		self.absMin = optimize.fmin(self.V, np.array([1., 1.]), disp=0)
		self.xmag = np.sum((self.absMin-self.metaMin)**2)**.5
		self.xhat = (self.absMin-self.metaMin)/self.xmag
		
	def V(self, X):
		X = np.array(X)
		x,y = X[...,0], X[...,1]
		Z = ((1+self.tilt2)*x*x+(1-self.tilt2)*y*y) * ((1+self.tilt1)*(x-self.x0)**2 + (1-self.tilt1)*(y-self.y0)**2 - self.c)
		return Z

	def dV(self, X):
		X = np.array(X)
		x,y = X[...,0], X[...,1]
		dVdx = 2*x*(1+self.tilt2)*((1+self.tilt1)*(x-self.x0)**2 + (1-self.tilt1)*(y-self.y0)**2 - self.c) \
				+ 2*((1+self.tilt2)*x*x+(1-self.tilt2)*y*y)*(1+self.tilt1)*(x-self.x0)
		dVdy = 2*y*(1-self.tilt2)*((1+self.tilt1)*(x-self.x0)**2 + (1-self.tilt1)*(y-self.y0)**2 - self.c) \
				+ 2*((1+self.tilt2)*x*x+(1-self.tilt2)*y*y)*(1-self.tilt1)*(y-self.y0)
		Y = np.empty_like(X)
		Y[...,0] = dVdx
		Y[...,1] = dVdy
		return Y
						
	def plot(self, box = None, n = 50, clevs = 200, cfrac=.3, deriv = None, **contourParams):
		if box == None:
			d = max(abs(self.metaMin-self.absMin))*.25
			xmin, xmax = (self.metaMin[0], self.absMin[0]) if (self.metaMin[0] < self.absMin[0]) else (self.absMin[0], self.metaMin[0])
			ymin, ymax = (self.metaMin[1], self.absMin[1]) if (self.metaMin[1] < self.absMin[1]) else (self.absMin[1], self.metaMin[1])
		else:
			d = 0
			xmin,xmax,ymin,ymax = box
		X = np.linspace(xmin-d, xmax+d, n).reshape(n,1)*np.ones((1,n))
		Y = np.linspace(ymin-d, ymax+d, n).reshape(1,n)*np.ones((n,1))
		XY = np.empty((n,n,2))
		XY[...,0],XY[...,1] = X,Y
		if deriv == None:
			Z = self.V(XY)
		elif deriv == 'x':
			Z = self.dV(XY)[...,0]
		elif deriv == 'y':
			Z = self.dV(XY)[...,1]
		N = np.linspace(min(Z.ravel()), max(Z.ravel())*cfrac, clevs)
		pylab.contour(X,Y,Z, N, **contourParams)
		


		
	
		

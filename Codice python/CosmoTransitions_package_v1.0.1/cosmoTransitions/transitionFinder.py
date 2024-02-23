
__version__ = "1.0.1"

import numpy as np
from scipy import linalg
from scipy import interpolate
from scipy import optimize
import pathDeformation

def traceMinimum(f, x0, t0, tstop, dtstart, df=None, d2f=None, dtabsMax=10.0, dtfracMax=.25, \
		dxdtMax=1e2, dtmin=1e-3, deltaX_target=.01, deltaX_tol=2., xeps=1e-4, teps=1e-3):
	"""
	This function traces the minimum x_min(t) of the function f(x,t).
	Inputs:
	  f - function f(x,t) that we're trying to trace.
	  x0, t0 - initial starting point
	  tstop - Tracing stops at this value.
	  dtstart - Initial stepsize.
	  df, d2f - The gradient and second derivative functions (optional).
	  dtabsMax, dtfracMax - The largest stepsize in t will be the LARGEST of
	    dtabsMax and t*dtfracMax.
	  dxdtMax - The largest (absolute) value we'll allow for dxdt before we
	    assume that the phase ends. (not used)
	  dtmin - The smallest stepsize we'll allow before assuming the transition ends.
	  deltaX_target - The target error in x at each step. Determines the
	    stepsize in t by extrapolation from last error.
	  deltaX_tol - deltaX_tol*deltaX_target gives the maximum error in x
	    before we want to shrink the stepsize and refind the minimum.
	  xeps - difference to use in derivatives AND accuracy in minimization.
	  teps - difference to use in time derivative.
	Outputs:
	  X, T, dXdT - arrays of the minimum at different values of t, and
	    its derivative with respect to t.
	  overX, overT - The point beyond which the phase seems to disappear.
	"""
	print "t0 =", t0
	N = x0.size
	# First, make matrix for calculation of first and second derivatives
	if df == None and d2f == None:
		d2x1 = np.zeros((N,N,N))
		d2x2 = np.zeros((N,N,N))
		for i in xrange(N):
			for j in xrange(N):
				if(i == j):
					d2x1[i,i,i] = 1
				else:
					d2x1[i,j,i] = .5; d2x1[i,j,j] = .5
					d2x2[i,j,i] = .5; d2x2[i,j,j] -= .5
		d2x1 *= xeps
		d2x2 *= xeps
		dx_ = np.diag(np.ones(N))*xeps
		M0 = (f(x0+d2x1,t)+f(x0-d2x1,t)-f(x0+d2x2,t)-f(x0-d2x2,t))/xeps**2
		minratio = 1e-2*min(abs(linalg.eigvalsh(M0)))/max(abs(linalg.eigvalsh(M0)))
		dxmindt = lambda x,t: _dxmin_dt(f,x,t,xeps,teps,d2x1,d2x2,dx_, minratio)
	else:
		M0 = d2f(x0,t0)
		minratio = 1e-2*min(abs(linalg.eigvalsh(M0)))/max(abs(linalg.eigvalsh(M0)))
		dxmindt = lambda x,t: _dxmin_dt_2(x, t, df, d2f, teps, minratio)
		# dxmindt = lambda x,t: linalg.solve( d2f(x,t), (df(x,t-teps)-df(x,t+teps))/(2*teps), overwrite_a = True, overwrite_b = True )
	# In the next line, it's important to add an offset to x so that we don't get stuck on a saddle point
	# Nevermind! This is no longer necessary since we're explicitly checking for saddle points by finding eigenvalues
	fmin = lambda x,t: optimize.fmin(f, x, args = (t,), xtol=xeps, ftol=np.inf, disp=False)
	
	deltaX_tol *= deltaX_target
	
	x,t,dt,xerr = x0,t0,dtstart,0.0
	dxdt, negeig = dxmindt(x,t)
	X,T,dXdT =[x],[t],[dxdt]
	overX = overT = overdXdT = None
	import sys
	while True and dxdt != None:
		sys.stdout.write('.'); sys.stdout.flush()
		# Get the values at the next step
		tnext = t+dt
		xnext = fmin(x+dxdt*dt, tnext)
		dxdt_next, negeig = dxmindt(xnext,tnext)
		if dxdt_next == None or negeig == True:
			# We got stuck on a saddle, so there must be a phase transition there.
			dt *= .5
			overX, overT, overdXdT = xnext, tnext, dxdt_next
			hasHitSaddle = True
		else:
			# The step might still be too big if it's outside of our error tolerance.
			xerr = max( np.sum((x+dxdt*dt - xnext)**2), np.sum((xnext-dxdt_next*dt - x)**2) )**.5
			if xerr < deltaX_tol: # Normal step, error is small
				T.append(tnext)
				X.append(xnext)
				dXdT.append(dxdt_next)
				if overT == None:
					dt *= deltaX_target/(xerr+1e-100) # change the stepsize only if the last step wasn't troublesome
				x,t,dxdt = xnext, tnext, dxdt_next
				overX = overT = overdXdT = None
			else: # Either stepsize was too big, or we hit a transition. Just cut the step in half.
				dt *= .5
				overX, overT, overdXdT = xnext, tnext, dxdt_next
		# Now do some checks on dt.
		if abs(dt) < abs(dtmin):
			# Found a transition! Or at least a point where the step is really small.
			break
		if dt > 0 and t >= tstop or dt < 0 and t <= tstop:
			# Reached tstop, but we want to make sure we stop right at tstop.
			dt = tstop-t
			x = fmin(x+dxdt*dt, tstop)
			dxdt,negeig = dxmindt(x,tstop)
			t = tstop
			X[-1], T[-1], dXdT[-1] = x,t,dxdt
			break
		dtmax = max(t*dtfracMax, dtabsMax)
		if abs(dt) > dtmax:
			dt = np.sign(dt)*dtmax
			
	if overT == None:
		overX, overT = X[-1], T[-1]
					
	sys.stdout.write('\n'); sys.stdout.flush()
	return np.array(X),np.array(T),np.array(dXdT),overX,overT
			

def _dxmin_dt(f,x,t,eps=.001,teps=.001,d1=None,d2=None,dx=None, minratio=0):
	if d1 == None:
		N = x.shape[-1]
		# First, make matrix for calculation of first and second derivatives
		# d2f/(dxi*dxj) = ( f(x+d2x1)+f(x-d2x1)-f(x+d2x2)-f(x-d2x2) )/eps**2
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
		# d2f/(dx*dt) = ( f(x+dx,t+eps)-f(x-dx,t+eps)-f(x+dx,t-eps)+f(x-dx,t-eps) )/(2*eps)**2
		dx = np.diag(np.ones(N))*eps
	
	M = (f(x+d1,t)+f(x-d1,t)-f(x+d2,t)-f(x-d2,t))/eps**2
	b = -(f(x+dx,t+teps)-f(x-dx,t+teps)-f(x+dx,t-teps)+f(x-dx,t-teps))/(4*eps*teps)
	try:
		eigs = linalg.eigvalsh(M)
		return linalg.solve( M, b, overwrite_a = True, overwrite_b = True )	, ((eigs<=0).any() or min(eigs)/max(eigs) < minratio)
	except:
		return None, False
	
def _dxmin_dt_2(x, t, df, d2f, teps, minratio=0):
	M = d2f(x,t)
	if abs(linalg.det(M)) < 1e-4*max(abs(M.ravel())):
		# Assume matrix is singular
		return None, False
	b = (df(x,t-teps)-df(x,t+teps))/(2*teps)
	try:
		eigs = linalg.eigvalsh(M)
		return linalg.solve( M, b, overwrite_a = True, overwrite_b = True )	, ((eigs<=0).any() or min(eigs)/max(eigs) < minratio)
	except:
		return None, False
	
def traceMultiMin(f, points, tLow, tHigh, dtstart, tjump = 1.0, forbidCrit = None, xeps = 1e-4, deltaX_target=.01, **args):
	"""
	This function traces multiple phases, going all the way from t0 to
	tstop even if there are transitions in the way. At each transition,
	t jumps an amount tjump to get to the next phase. It traces both
	down and up from the jump point. args provides all of the optional
	arguments for traceMinimum().
	Inputs:
	  f - the function f(x,t) that we want to trace
	  points - a list of points [(x1,t1), (x2,t2),...] that we want to trace.
	  tLow,tHigh - The lowest and highest temperatures between which to trace.
	  dtstart - initial stepsize (should be > 0)
	  tjump - the jump in t from the end of one phase to the start of the trace of another.
	  forbidCrit - A function that determines whether we want to forbid a phase with a given starting point.
	  xeps, deltaX_target - These are passed to traceMinimum(), and they are used in the minimization function.
	Outputs a dictionary of phases, each containing:
	  X, T, dXdT - the minima and its derivative at different temperatures.
	  tck - a tuple of spline coefficients and knots.
	  linkedFrom - The index of the phase that lead to this one.
	  highLink, lowLink - The indices of the phases that this one links to.
	"""
	args['xeps'], args['deltaX_target'] = xeps, deltaX_target
	# We want the minimization here to be very accurate so that we don't get stuck on a saddle or something.
	# This isn't much of a bottle neck.
	fmin = lambda x,t: optimize.fmin(f, x+deltaX_target, args = (t,), xtol=xeps*1e-3, ftol=np.inf, disp=False)

	phases = []
	nextPoint = []
	for p in points:
		x,t = p
		nextPoint.append([t,dtstart,fmin(x,t),None])
	
	while len(nextPoint) != 0:
		t1,dt1,x1,linkedFrom = nextPoint.pop()
		x1 = fmin(x1, t1) # make sure we start as accurately as possible. Added 2011/11/28.
		# Check to see if this point is outside the bounds
		if t1 < tLow or (t1 == tLow and dt1 < 0):
			continue
		if t1 > tHigh or (t1 == tHigh and dt1 > 0):
			continue
		if forbidCrit != None and forbidCrit(x1) == True:
			continue
		# Check to see if it's redudant with another phase
		for i in xrange(len(phases)):
			phase = phases[i]
			if t1 < min(phase['T'][0], phase['T'][-1]) or t1 > max(phase['T'][0], phase['T'][-1]):
				continue
			x = fmin( np.array(interpolate.splev(t1, phase['tck'])), t1)
			if np.sum((x-x1)**2)**.5 < xeps*100:
				# The point is already covered
				if linkedFrom != i and linkedFrom >= 0:
					lastPhase = phases[linkedFrom]
					lastPhase['highLink' if t1>np.average(lastPhase['T']) else 'lowLink'] = i
				break
		else:
			# Otherwise, trace the phase
			print "Tracing phase starting at x =",x1,"; t =",t1
			oldNumPoints = len(nextPoint)
			if (t1 > tLow):
				print "Tracing minimum down"
				X_down, T_down, dXdT_down, nX, nT = traceMinimum(f,x1,t1,tLow,-dt1,**args)
				t2,dt2 = nT-tjump, .1*tjump
				x2 = fmin(nX,t2)
				nextPoint.append([t2,dt2,x2,len(phases)])
				for point in findApproxLocalMin(f,X_down[-1],x2,(nT,),mindeltax=deltaX_target*10):
					nextPoint.append([nT,dt2,fmin(point,nT),len(phases)])
				X_down, T_down, dXdT_down = X_down[::-1], T_down[::-1], dXdT_down[::-1]
			if (t1 < tHigh):
				print "Tracing minimum up"
				X_up, T_up, dXdT_up, nX, nT = traceMinimum(f,x1,t1,tHigh,dt1,**args)
				t2,dt2 = nT+tjump, .1*tjump
				x2 = fmin(nX,t2)
				nextPoint.append([t2,dt2,x2,len(phases)])
				for point in findApproxLocalMin(f,X_up[-1],x2,(nT,),mindeltax=deltaX_target*10):
					nextPoint.append([nT,dt2,fmin(point,nT),len(phases)])
			# Then join the two together
			if (t1 <= tLow):
				X,T,dXdT = X_up, T_up, dXdT_up
			elif (t1 >= tHigh):
				X,T,dXdT = X_down, T_down, dXdT_down
			else:
				X,T,dXdT = np.append(X_down, X_up[1:], 0), np.append(T_down, T_up[1:]), np.append(dXdT_down, dXdT_up[1:], 0)
			if len(X) > 1:
				phases.append({'X':X, 'T':T, 'dXdT':dXdT, 'linkedFrom':linkedFrom, 'highLink':None, 'lowLink':None})
				_makePhaseSpline(phases[-1])
				if (linkedFrom >= 0):
					lastPhase = phases[linkedFrom]
					lastPhase['highLink' if t1>np.average(lastPhase['T']) else 'lowLink'] = len(phases)-1
			else:
				# The phase is just a single point. Don't add it, and make it a dead-end.
				nextPoint = nextPoint[:oldNumPoints]
		
	return phases
	
def findApproxLocalMin(f,x1,x2,args=(),n=100,edge=.05,mindeltax=.01):
	"""
	When jumping between phases, we want to make sure that we
	don't jump over an intermediate phase. This function will
	find local minima between points x1 and x2.
	Outputs a list of points.
	"""
	x1,x2 = np.array(x1), np.array(x2)
	dx = np.sum((x1-x2)**2)**.5
	if dx < mindeltax:
		return np.array([]).reshape(0,len(x1))
	x = x1 + (x2-x1)*np.linspace(edge,1-edge,n).reshape(n,1)
	y = f(x,*args)
	i = (y[2:]>y[1:-1]) & (y[:-2]>y[1:-1])
	return x[1:-1][i]
	
def _makePhaseSpline(phase):
	# We shouldn't ever really need to sort the array, but there must be some bug in the above code that makes it
	# so that occasionally the last step goes backwards. This should fix that.
	i = np.argsort(phase['T'])
	phase['T'] = phase['T'][i]
	phase['X'] = phase['X'][i]
	phase['dXdT'] = phase['dXdT'][i]
	tck, u = interpolate.splprep(phase['X'].T, u=phase['T'], s=0, k = (3 if len(phase['T']) > 3 else 1))
	phase['tck'] = tck
		
def removeRedundantPhases(f, phases, xeps=1e-5, diftol=1e-2):
	delIndices = []
	redundantWith = []
	for i in xrange(len(phases)-1):
		if i in delIndices:
			continue
		for j in xrange(i+1, len(phases)):
			if j in delIndices:
				continue
			tmax = min(phases[i]['T'][-1], phases[j]['T'][-1])
			tmin = max(phases[i]['T'][ 0], phases[j]['T'][ 0])
			if tmin > tmax:
				# no overlap in the phases
				continue
			if abs(phases[i]['T'][0] - phases[j]['T'][0]) > tmax-tmin or abs(phases[i]['T'][-1] - phases[j]['T'][-1]) > tmax-tmin:
				# They overlap, but their endpoints are significantly different
				continue
			tcenter = .5*(tmin+tmax)
			x1 = np.array(interpolate.splev(tcenter, phases[i]['tck']))
			x2 = np.array(interpolate.splev(tcenter, phases[j]['tck']))
			x1 = np.array(optimize.fmin(f, x1, args=(tcenter,), xtol=xeps, ftol=np.inf, disp=False))
			x2 = np.array(optimize.fmin(f, x2, args=(tcenter,), xtol=xeps, ftol=np.inf, disp=False))
			dif = np.sum((x1-x2)**2)**.5
			if dif < diftol:
				# same phase, delete one
				delIndices.append(j)
				redundantWith.append(i)
				print "Phases", i, j, "are redundant"
	# It's possible that we're trying to delete a phase more than once
	for i in xrange(len(delIndices)-1,0,-1):
		if delIndices[i] in delIndices[:i]:
			del delIndices[i]
			del redundantWith[i]
	# New logic starts here --------
	# Make groupings of redundant phases.
	groupings = []
	for delIndex, rWithIndex in zip(delIndices,redundantWith):
		appendedGroups = []
		for i in xrange(len(groupings)):
			if delIndex in groupings[i] or rWithIndex in groupings[i]: appendedGroups.append(i)
		# Everything in appendedGroups will need to be merged together, since they now share delIndex and rWithIndex
		if len(appendedGroups) > 1:
			i0 = appendedGroups[0]
			for i in appendedGroups[:0:-1]: # slice is reversed indices, excluding the first one
				groupings[i0] += groupings[i]
				del groupings[i]
		# Add delIndex and rWithIndex to the remaining group
		if len(appendedGroups) >= 1: 
			appendedGroup = groupings[appendedGroups[0]]
			if rWithIndex not in appendedGroup: appendedGroup.append(rWithIndex)
			if delIndex   not in appendedGroup: appendedGroup.append(delIndex)
		# Or make a new group
		else:
			groupings.append([rWithIndex, delIndex])
	# Now for each group, check the links. We want the first phase in the group to link to something OUTSIDE the group.
	delIndices2 = []
	for group in groupings:
		i0 = group[0] # This is the index of the phase that we won't delete
		for link in ('linkedFrom', 'highLink', 'lowLink'):
			for index in group:
				if phases[index][link] not in group: # We link to outside the group. Great!
					phases[i0][link] = phases[index][link]
					break 
			else: # All links are within the group
				phases[i0][link] = None
			# Now we want to make sure that everything linking to the group links to i0
			for p in phases:
				if p[link] in group: p[link] = i0
		# just remake the delIndices, in case my old logic screw up here too		
		for index in group[1:]:
			if index not in delIndices2: delIndices2.append(index)
	# Finally, delete the indices and rename the links
	delIndices2.sort()
	for delIndex in delIndices2[::-1]:
		for i in xrange(len(phases)): # rename the links
			for link in ('linkedFrom', 'highLink', 'lowLink'):
				if phases[i][link] > delIndex: phases[i][link] -= 1
		del phases[delIndex] # delete the index

	# Old logic -------
#	for delIndex, rWithIndex in zip(delIndices[::-1],redundantWith[::-1]):
#		for i in xrange(len(phases)):
#			phase = phases[i]
#			for link in ('linkedFrom', 'highLink', 'lowLink'):
#				if phase[link] == delIndex and i != rWithIndex:
#					phase[link] = rWithIndex
#				elif phase[link] == delIndex:
#					phase[link] = phases[delIndex][link]
#				if phase[link] > delIndex:
#					phase[link] = phase[link]-1
#		del phases[delIndex]
				
def findTransitionRegions(f, phases):
	"""
	Finds all regions where a transition can occur, assuming that
	transitions always proceed in one direction (i.e., there are
	no transitions that go back and forth between phases multiple
	times).
	Inputs:
	  f - The free energy function f(x,T)
	  phases - a list containing dictionaries, each containing 
	    the 'T', 'tck', and 'linkedFrom' entries from traceMultiMin().
	Returns list of [(Tmin, Tmax, phase1, phase2)].
	  Tmin, Tmax - The minimum and maximum temperatures at which
	    the transition can occur. Tmin is generally the lowest
		temperature at which both phases exist, and Tmax is 
		generally the temperature at which the two phases are
		degenerate. For a second-order transition, Tmin = Tmax.
		If one phase is below the other for it's entire existance,
		then Tmax is the top of that phase.
	  phase1, phase2 - The low- and high-T phases indices, respectively.
	"""
	transitions = []
	X = lambda T, i: np.array(interpolate.splev(T, phases[i]['tck'])).T
	DV = lambda T, i, j: f(X(T,i),T) - f(X(T,j),T)
	for i in xrange(len(phases)-1):
		for j in xrange(i+1, len(phases)):
			tmax = min(phases[i]['T'][-1], phases[j]['T'][-1])
			tmin = max(phases[i]['T'][ 0], phases[j]['T'][ 0])
			if tmin > tmax:
				# no overlap in the phases, check for second order
				if phases[i]['linkedFrom'] == j or phases[j]['linkedFrom'] == i or \
				   phases[i]['lowLink'] == j or phases[j]['lowLink'] == i:
					# The two are linked, so there's a second order transition
					tcrit = .5*(tmin+tmax)
					transitions.append((tcrit, tcrit, i, j) if phases[i]['T'][0] < phases[j]['T'][0] else (tcrit, tcrit, j, i))
				continue
			a,b = DV(tmin,i,j), DV(tmax,i,j)
			if a < 0 and b < 0: # phase i is always lower				
				transitions.append((tmin, tmax, i, j))
				continue
			if a > 0 and b > 0: # phase j is always lower				
				transitions.append((tmin, tmax, j, i))
				continue
			tcrit = optimize.brentq(DV,tmin,tmax,args=(i,j))
			if a < 0: # i is low-T phase
				transitions.append((tmin, tcrit, i, j))
			else: # j is low-T phase
				transitions.append((tmin, tcrit, j, i))
	
	# Do some checks to make sure that we aren't transitioning from a low-T phase to high-T phase
	# instead of the other way around. This can happen if there's only a small overlap that wasn't
	# well-resolved. In this case, just set it to a 2nd-order phase transition.
	for i in xrange(len(transitions)):
		tmin, tmax, lowp, highp = transitions[i]
		if phases[highp]['highLink'] == lowp and phases[lowp]['lowLink'] == highp:
			transitions[i] = (tmin, tmin, highp, lowp)
				
	return transitions
	
def plotPhases(f, phases, pylab, N=500):
	for phase in phases:
		t = np.linspace(phase['T'][0], phase['T'][-1], N)
		x = np.array(interpolate.splev(t, phase['tck'])).T
		pylab.plot(t, f(x,t))
	
def getStartPhase(phases, V=None):
	"""
	Returns the index of the high-T phase.
	Inputs:
	  phases - output from findMultiMin()
	  V - the potential V(x,T). Only necessary if there are
	    multiple phases with the same Tmax.
	"""
	startPhases = []
	startPhase = None
	Tmax = None
	for i in xrange(len(phases)):
		if phases[i]['T'][-1] == Tmax:
			# add this to the startPhases list.
			startPhases.append(i)
		elif Tmax == None or phases[i]['T'][-1] > Tmax:
			startPhases = [i]
			Tmax = phases[i]['T'][-1]
	if len(startPhases) == 1 or V == None:
		startPhase = startPhases[0]
	else:
		# more than one phase have the same maximum temperature
		# Pick the stable one at high temp.
		Vmin = None
		for i in startPhases:
			V_ = V(phases[i]['X'][-1], phases[i]['T'][-1])
			if Vmin == None or V_ < Vmin:
				Vmin = V_
				startPhase = i
	if startPhase == None:
		raise Exception, "Error in transitionFinder.getStartPhase."
	return startPhase
	
							
# Next step, find the amount of supercooling and figure out how the system moves amongst the various phases.
# This can get really complicated when you have three phases at once, with one between the other two in field space.
# One option is to always tunnel to the closer of two phases if the angle of tunneling is less than some small amount
# (like 10 degrees or so).


def findFullTransitions(f, df, phases, transitions=None, startPhase=None, nuclCriterion = lambda T: 140.0*T, overlap=45.0, **tunnelParams):
	"""
	DEPRECIATED. Use the class fullTransitions instead.
	
	This function will find the actual transition temperature
	by solving the bubble nucleation rate.
	Inputs:
	  f - the potential energy f(x,T)
	  df - The gradient of the f.
	  phases - output from traceMultiMin()
	  transitions - output from findTransitionRegions()
	  startPhase - The index of the hot phase.
	  nuclCriterion - The function such that S = nuclCriterion(T)
	    at the actual nucleation temperature.
	  overlap - Suppose three phases A, B and C, where A is the hot phase.
	    If B and C are within 'overlap' degrees of each other with
		respect to A, and A can transition to both B and C, then this
		will restrict A to transition to the closer of B and C. This
		prevents us from attempting to tunnel through multiple phases
		at once.
	Outputs a list of tuples: [(tunnelObj, lowPhase, highPhase)]
	  lowPhase, highPhase - The low- and high-T phase indicies, respectively.
	  tunnelObj - Output from pathDeformation.fullTunneling.
	"""
	if transitions == None:
		transitions = findTransitionRegions(f, phases)
	fullTransitions = []
	highPhase_index = startPhase if startPhase != None else getStartPhase(phases, f)
	Tmax = phases[highPhase_index]['T'][-1]
	while True:
		# First, go through all the phases in transitions and write down which ones we can tunnel to. 
		tempTransitions = []
		for tmin,tmax,lowPhase,highPhase in transitions:
			if highPhase != highPhase_index or tmin > Tmax:
				continue # not a possible transition
			else:
				tempTransitions.append((tmin, min(tmax, Tmax), lowPhase))
		
		# Before we try tunneling to them, we need to check for overlap.
		excludedTemps = []
		for i in xrange(len(tempTransitions)):
			excludedTemps.append([])
		for i in xrange(len(tempTransitions)-1):
			for j in xrange(i+1, len(tempTransitions)):
				# See where i and j overlap, and then exclude whichever is closer in the overlap region.
				tmini,tmaxi,I = tempTransitions[i]
				tminj,tmaxj,J = tempTransitions[j]
				if max(tmini, tminj) >= min(tmaxi, tmaxj):
					continue # no overlap
				t = .5*( max(tmini, tminj) + min(tmaxi, tmaxj) )
				xi = np.array(interpolate.splev(t, phases[I]['tck']))
				xj = np.array(interpolate.splev(t, phases[J]['tck']))
				x0 = np.array(interpolate.splev(t, phases[highPhase_index]['tck']))
				di, dj = xi-x0, xj-x0
				si, sj = np.sum(di**2)**.5, np.sum(dj**2)**.5
				if np.sum(di*dj) > si*sj*np.cos(overlap * np.pi/180.):
					# The phases are close to each other. Exclude the region in the further one.
					k = i if si > sj else j
					excludedTemps[k].append((max(tmini, tminj), min(tmaxi, tmaxj)))
		
		# Now we have a list of excluded temperatures that goes along with each transition region.
		# Combine all of these excluded temps to get the new transition regions.
		tempTran2 = []
		for i in xrange(len(tempTransitions)):
			exTemps = excludedTemps[i]
			tmin, tmax, lowPhase_index = tempTransitions[i]
			tranTemps = [(tmin,tmax)]
			for ex1, ex2 in exTemps:
				j = 0
				while j < len(tranTemps):
					t1,t2 = tranTemps[j]
					if t1 < ex1 < ex2 < t2: # the excluded region is in the center of this region
						tranTemps[j:j+1] = [(t1,ex1), (ex2,t2)]
					elif t1 < ex1 < t2 <= ex2: # excluded region covers the entire upper part of this region
						tranTemps[j] = (t1,ex1)
					elif ex1 <= t1 < ex2 < t2: # excluded region covers the entire lower part of this region
						tranTemps[j] = (ex2,t2)
					elif ex1 <= t1 < t2 <= ex2: # The entire region is excluded
						tranTemps[j:j+1] = []
					else: # None of the region is excluded
						pass
					j += 1
			for tmin,tmax in tranTemps:
				tempTran2.append( (tmin, tmax, lowPhase_index) )
		
		# OK! We now have a complete list of transitions from the current phase (highPhase_index), with all the proper regions excluded.
		# Now, start trying to tunnel at each of these regions.
		Tnuc = None # This is going to be the lowest we're allowed to go in the hot phase
		finalTran = None
		for tmin, tmax, lowPhase_index in tempTran2:
			if Tnuc != None and Tnuc > tmax:
				# This entire transition region happens below the nucleation temperature for a transition to a different phase,
				# so the system never reaches this transition region.
				continue
			if Tnuc != None and Tnuc > tmin:
				tmin = Tnuc	# Make sure that we don't try to find a transition below where we've already found one
			print "\n\nFinding the transtion from phases",highPhase_index,"to",lowPhase_index,\
				  "and at temperatures between",tmin,"and",tmax
			tunnelObj = critTranForPhases(f, df, phases[lowPhase_index], phases[highPhase_index], tmax, tmin, nuclCriterion, **tunnelParams)
			if tunnelObj == None:
				# some error happened
				continue
			if (Tnuc == None or tunnelObj.T > Tnuc) and tunnelObj.T > tmin: 
				# The highPhase transitions to lowPhase before any other transition
				# If tunnelObj.T == tmin, then we assume that the transition never actually happens
				Tnuc = tunnelObj.T
				finalTran = (tunnelObj, lowPhase_index, highPhase_index)
				
		if finalTran == None:
			break # There were no transitions. This was the last phase.
		
		# At this point we should have found the transition for highPhase.
		# Now set the highPhase to the thing we transitioned to, and make sure that it's maximum temperature is the temp
		# that we just nucleated at.
		highPhase_index = finalTran[0]
		Tmax = finalTran[2]
		fullTransitions.append(finalTran)
		
	return fullTransitions
			
	
def critTranForPhases(f, df, lowPhase, highPhase, Tcrit, Tmin = None, nuclCriterion = lambda T: 140.0*T, **tunnelParams):
	"""
	This function will find the actual transition temperature
	by solving the bubble nucleation rate between just two phases.
	Inputs:
	  f - the potential energy f(x,T)
	  df - The gradient of the f.
	  lowPhase, highPhase - The low and high-temperature phases.
	    These should be dictionaries containing 'T' and 'tck'.
	  nuclCriterion - The function such that S/T = nuclCriterion(T)
	    at the actual nucleation temperature.
	  **tunnelParams - All extra parameters to pass to 
	    pathDeformation.criticalTunneling().
	Outputs:
	  Instance of pathDeformation.fullTunneling. This contains all
	  of information on the bubble shape, action, and temperature. 
	"""
	if len(lowPhase['T']) == 0 or len(highPhase['T']) == 0:
		raise Exception, "Error in transitionFinder.transitionTempForPhases(): input empty phase."
	if len(lowPhase['T']) == 1 or len(highPhase['T']) == 1:
		# One of the phases is just a single point.
		raise Exception, "The two phases must be more than just a single point"
		return (Tcrit, None, None) if fulloutput else (Tcrit, None)
	Tmax = min(lowPhase['T'][-1], highPhase['T'][-1])
	if Tmin != None:
		Tmin = max(lowPhase['T'][ 0], highPhase['T'][ 0])
	if Tmax <= Tcrit:
		# Second-order transition
		return pathDeformation.secondOrderTransition(highPhase['X'][0], highPhase['T'][0])
	
	# Gotten all of the errors out of the way, so now we can just solve for the temperature using the method
	# in pathDeformation.py.
	return pathDeformation.criticalTunneling(f, df, Tmin, Tcrit, lowPhase['tck'], highPhase['tck'], nuclCriterion, **tunnelParams)
	

class fullTransitions:
	"""
	This class will find the actual transition temperature
	by solving the bubble nucleation rate.
	Outputs a list of tuples to self.out: [(lowPhase, highPhase, tunnelObj)]
	  lowPhase, highPhase - The low- and high-T phase indicies, respectively.
	  tunnelObj - Output from pathDeformation.fullTunneling.
	"""
	def __init__(self, V, dV, phases, transitions=None, startPhase=None, nuclCriterion = lambda S, T: S/(T+1e-100) - 140.0, \
				Ttol=1e-3, phiminTol=1e-5, overlapAngle=45.0, verbose = 1, **tunnelParams):
		"""
		Inputs:
		  V - the potential energy V(phi,T)
		  dV - The gradient of the V.
		  phases - output from traceMultiMin()
		  transitions - output from findTransitionRegions()
		  startPhase - The index of the hot phase.
		  nuclCriterion - The function such that nuclCriterion(S,T) = 0
			at the actual nucleation temperature, where S is the action.
		  Ttol - The tolerance in temperature when finding the nucleation temp.
		  phiminTol - The tolerance in minimizing the V(phi,T)
		  overlapAngle - Suppose three phases A, B and C, where A is the hot phase.
			If B and C are within 'overlap' degrees of each other with
			respect to A, and A can transition to both B and C, then this
			will restrict A to transition to the closer of B and C. This
			prevents us from attempting to tunnel through multiple phases
			at once.
		"""
		self.V, self.dV = V, dV
		if transitions == None:
			transitions = findTransitionRegions(f, phases)
		self.phases = phases
		self.transitions = transitions
		self.nuclCriterion = nuclCriterion
		self.Ttol = Ttol
		self.phitol = phiminTol
		self.overlapAngle = overlapAngle
		self.tunnelParams = tunnelParams
		self.verbose = verbose
		self.tunnelParams.update(verbose=verbose)
		self.out = []
		self.lastHighTunnel = (np.inf, np.inf)
		self.highTunnels = None
		
		next_phase = startPhase if startPhase != None else getStartPhase(phases, V)
		
		Tmax = np.inf
		while next_phase != None:
			next_phase, Tmax = self.findTransitionForIndex(next_phase, Tmax)


	def findTransitionForIndex(self, highIndex, Tmax):
		transTemps = []
		allLowPhases = {}
		# First, find all of the transition that we can go to from here, and the min/max temp
		for tmin, tmax, lowi, highi in self.transitions:
			if highi != highIndex or tmin==tmax or tmin >= Tmax:
				continue
			tmax = min(Tmax, tmax)
			allLowPhases[lowi] = {"lowT":tmin, "highT":tmax, "lowFT":None, "highFT":None, "lastT":None, "lastFT":None}
			transTemps += [tmin, tmax]
			
		# Create a different tunneling region for each set of transitions.
		transTemps.sort()
		regions = []
		for tmin, tmax in zip(transTemps[:-1], transTemps[1:]):
			if tmin < tmax:
				lowPhases = {}
				for i in allLowPhases:
					if allLowPhases[i]["lowT"] <= tmin and allLowPhases[i]["highT"] >= tmax:
						lowPhases[i] = allLowPhases[i]
				regions.append((tmin,tmax,lowPhases))
			
		rlow = None
		# Now for each distinct region, try and do the tunneling
		for Tmin, Tmax, lowPhases in regions[::-1]:
			rhigh = self.tunnelingAtT(Tmax, highIndex, lowPhases, False)
			rlow = self.tunnelingAtT(Tmin, highIndex, lowPhases, False)
			if rlow > 0 and rhigh > 0:
				if self.verbose >= 1: print "Tunneling cannot occur at either Tmin or Tmax. Checking for tunneling in between."
				rlow2 = self.tunnelingAtT(Tmin*.99+Tmax*.01, highIndex, lowPhases, False)
				if rlow2 < rlow:
					rhigh2 = self.tunnelingAtT(Tmin*.01+Tmax*.99, highIndex, lowPhases, False)
					if rhigh2 < rhigh: # There's definitely a minimum somewhere between. Find it.
						self.lastLowTunnel = (np.inf, Tmax)
						self.highTunnels = []
						try:
							def cb(x, a=self):
								if a.lastLowTunnel[0] < 0:
									raise Exception
							optimize.fmin(self.tunnelingAtT, .5*(Tmin+Tmax), args=(highIndex, lowPhases, False), \
													xtol=10*self.Ttol*(Tmax-Tmin), ftol=np.inf, disp=0, callback=cb)
						except: pass
						if self.lastLowTunnel[0] < 0: # Tunneling is possible, just not at Tmin or Tmax
							rlow, Tmin = self.lastLowTunnel
							for r,t in self.highTunnels:
								if Tmin < t < Tmax: rhigh, Tmax = r,t
						self.highTunnels = None
			elif rlow < 0 and rhigh < 0:
				# Tunneling happens at both Tmin and Tmax
				besti, bestft = None, None
				x = np.inf
				for i in lowPhases:
					lowP = lowPhases[i]
					if lowP["highFT"] != None and lowP["highFT"].action != None:
						ft = lowP["highFT"]
						y = abs(self.nuclCriterion(ft.action, ft.T))
						if y < x:
							x, besti, bestft = y, i, ft
				if besti != None:
					self.out.append( (bestft, besti, highIndex) )
				if besti == None:
					besti = self.phases[highIndex]['lowLink']
				return besti, Tmax
			if rlow < 0 < rhigh:
				# find the nucleation temperature
				Tnuc = optimize.brentq(self.tunnelingAtT, Tmin, Tmax, args=(highIndex, lowPhases), \
											xtol = self.Ttol*(Tmax-Tmin), disp = False)
				besti = None
				bestft = None
				x = np.inf
				for i in lowPhases:
					y = y1 = y2 = y3 = np.inf
					lowP = lowPhases[i]
					if lowP["lowT"] != None and lowP["lowFT"] != None and lowP["lowFT"].action != None:
						ft = lowP["lowFT"]
						y1 = abs(self.nuclCriterion(ft.action, ft.T))
					if lowP["highT"] != None and lowP["highFT"] != None and lowP["highFT"].action != None:
						ft = lowP["highFT"]
						y2 = abs(self.nuclCriterion(ft.action, ft.T))
					if lowP["lastT"] != None and lowP["lastFT"] != None and lowP["lastFT"].action != None:
						ft = lowP["lastFT"]
						y3 = abs(self.nuclCriterion(ft.action, ft.T))
					if y1 == min((y1,y2,y3)): 
						y, ft = y1, lowP["lowFT"]
					elif y2 == min((y1,y2,y3)): 
						y, ft = y2, lowP["highFT"]
					elif y3 == min((y1,y2,y3)): 
						y, ft = y3, lowP["lastFT"]
					if y < x:
						x, besti, bestft = y, i, ft
				
				if besti == None and self.phases[highIndex]['T'][0] == Tmin: # This block added 2011/11/22.
					# We couldn't do tunneling anywhere, but we know that tunneling must happen by Tmin.
					# Assume a second-order phase transition.
					print "Couldn't find any tunneling solution. Assuming second-order transition."
					phi = np.array(interpolate.splev(Tmin, self.phases[highIndex]['tck']))
					bestft = pathDeformation.secondOrderTransition(phi, Tmin)
					# Find the next phase (just by proximity)
					d = np.inf
					for i in lowPhases:
						phi2 = np.array(interpolate.splev(Tmin, self.phases[i]['tck']))
						d2 = np.sum((phi2-phi)**2)**.5
						if d2 < d:
							d2, besti = d, i
					Tnuc = Tmin
					
				if besti >= 0:
					self.out.append( (bestft, besti, highIndex) )
					if self.verbose >= 1: print "Found a transition from",highIndex,"to",besti,"at T =",Tnuc
					return besti, Tnuc
			
		else:
			# couldn't find a transition. Check for second-order
			for tmin, tmax, lowi, highi in self.transitions:
				if highi == highIndex and tmin==tmax:
					phi = np.array(interpolate.splev(tmin, self.phases[highIndex]['tck']))
					ft = pathDeformation.secondOrderTransition(phi, tmin)
					self.out.append((ft, lowi, highi))
					return lowi, tmin
			
		return None, 0.0		
			
			
	def findTransitionForIndex_old(self, highIndex, Tmax):
		# First, find all of the transition that we can go to from here, and the min/max temp
		# If the transition regions are disjoint, find the tunneling for each region separately.
		regions = []
		for tmin, tmax, lowi, highi in self.transitions:
			if highi != highIndex or tmin==tmax or tmin >= Tmax:
				continue
			tmax = min(Tmax, tmax)
			lowPhases = {lowi: {"lowT":tmin, "highT":tmax, "lowFT":None, "highFT":None, "lastT":None, "lastFT":None}}
			regions.append((tmin, tmax, lowPhases))
			
		# connect any regions that overlap
		didMerge = True
		while didMerge:
			didMerge = False
			for i in xrange(len(regions)-1,-1,-1):
				tmin1,tmax1,lowP1 = regions[i]
				for j in xrange(i-1,-1,-1):
					tmin2,tmax2,lowP2 = regions[j]
					if (tmin1 <= tmin2 <= tmax1 or tmin1 <= tmax2 <= tmax1 or tmin2 <= tmin1 <= tmax2 or tmin2 <= tmax1 <= tmax2):
						didMerge = True # one of the phases is between one of the others
						tmin = min(tmin1,tmin2)
						tmax = max(tmax1,tmax2)
						lowP2.update(lowP1)
						regions[j] = (tmin, tmax, lowP2)
						del regions[i]
						break
			
		# Sort the regions from highest to lowest.
		newRegs = []
		for reg in regions:
			i = 0
			for reg2 in newRegs:
				if reg[0] < reg2[0]: i += 1
			newRegs.insert(i, reg) # not exactly an efficient sort, but it sure is easy
		regions = newRegs
		
		next_phase = None
		# Now for each distinct region, try and do the tunneling
		for Tmin, Tmax, lowPhases in regions:
			rlow = self.tunnelingAtT(Tmin, highIndex, lowPhases, False)
			rhigh = self.tunnelingAtT(Tmax, highIndex, lowPhases, False)
			if rlow > 0 and rhigh > 0:
				if self.verbose >= 1: print "Tunneling cannot occur at either Tmin or Tmax. Checking for tunneling in between."
				# Try to see if there's a minimum inbetween
				T_maxnuc, r = optimize.fmin(self.tunnelingAtT, .5*(Tmin+Tmax), args=(highIndex, lowPhases, False), \
										xtol=10*self.Ttol*(Tmax-Tmin), ftol=np.inf, disp=0, full_output=True)[:2]
				if r < 0: # Tunneling is possible, just not at Tmin or Tmax
					rlow, Tmin = r, T_maxnuc[0]
			elif rlow < 0 and rhigh < 0:
				# Tunneling happens at both Tmin and Tmax
				besti, bestft = None, None
				x = np.inf
				for i in lowPhases:
					lowP = lowPhases[i]
					if lowP["highFT"] != None and lowP["highFT"].action != None:
						ft = lowP["highFT"]
						y = abs(self.nuclCriterion(ft.action, ft.T))
						if y < x:
							x, besti, bestft = y, i, ft
				if bestft != None:
					self.out.append( (bestft, besti, highIndex) )
				return besti, Tmax
			if rlow < 0 < rhigh:
				# find the nucleation temperature
				Tnuc = optimize.brentq(self.tunnelingAtT, Tmin, Tmax, args=(highIndex, lowPhases), \
											xtol = self.Ttol*(Tmax-Tmin), disp = False)
				besti = None
				bestft = None
				x = np.inf
				for i in lowPhases:
					y = y1 = y2 = y3 = np.inf
					lowP = lowPhases[i]
					if lowP["lowT"] != None and lowP["lowFT"] != None and lowP["lowFT"].action != None:
						ft = lowP["lowFT"]
						y1 = abs(self.nuclCriterion(ft.action, ft.T))
					if lowP["highT"] != None and lowP["highFT"] != None and lowP["highFT"].action != None:
						ft = lowP["highFT"]
						y2 = abs(self.nuclCriterion(ft.action, ft.T))
					if lowP["lastT"] != None and lowP["lastFT"] != None and lowP["lastFT"].action != None:
						ft = lowP["lastFT"]
						y3 = abs(self.nuclCriterion(ft.action, ft.T))
					if y1 == min((y1,y2,y3)): 
						y, ft = y1, lowP["lowFT"]
					elif y2 == min((y1,y2,y3)): 
						y, ft = y2, lowP["highFT"]
					elif y3 == min((y1,y2,y3)): 
						y, ft = y3, lowP["lastFT"]
					if y < x:
						x, besti, bestft = y, i, ft
				
				if besti == None: # This block added 2011/11/22.
					# We couldn't do tunneling anywhere, but we know that tunneling must happen by Tmin.
					# Assume a second-order phase transition.
					print "Couldn't find any tunneling solution. Assuming second-order transition."
					phi = np.array(interpolate.splev(Tmin, self.phases[highIndex]['tck']))
					bestft = pathDeformation.secondOrderTransition(phi, Tmin)
					# Find the next phase (just by proximity)
					d = np.inf
					for i in lowPhases:
						phi2 = np.array(interpolate.splev(Tmin, self.phases[i]['tck']))
						d2 = np.sum((phi2-phi)**2)**.5
						if d2 < d:
							d2, besti = d, i
					Tnuc = Tmin
					
				if besti >= 0:
					self.out.append( (bestft, besti, highIndex) )
					return besti, Tnuc
		else:
			# couldn't find a transition. Check for second-order
			for tmin, tmax, lowi, highi in self.transitions:
				if highi == highIndex and tmin==tmax:
					phi = np.array(interpolate.splev(tmin, self.phases[highIndex]['tck']))
					ft = pathDeformation.secondOrderTransition(phi, tmin)
					self.out.append((ft, lowi, highi))
					return lowi, tmin
			
		return None, 0.0		
				
		
	def tunnelingAtT(self, T, highIndex, lowPhases, updateBracketTemps = True):
		try:
			T = T[0] # need this when the function is run from optimize.fmin
		except:
			pass
						
		# Find all of the minima
		phiHigh_ = np.array(interpolate.splev(T, self.phases[highIndex]['tck']))
		phiHigh = optimize.fmin(self.V, phiHigh_, args=(T,), xtol = self.phitol, ftol = np.inf, disp=0)
		for i in lowPhases:
			lowP = lowPhases[i]
			lowP["phi"] = None
			if lowP["lowT"] <= T <= lowP["highT"]:
				phi_ = np.array(interpolate.splev(T, self.phases[i]['tck']))
				phi = optimize.fmin(self.V, phi_, args = (T,), xtol = self.phitol, ftol = np.inf, disp=0)
				if .95 < (np.sum((phi_-phiHigh_)**2)/np.sum((phi-phiHigh)**2))**.5 < 1.05:
					lowP["phi"] = phi # The two didn't shift too much. Use this.
					
		# If the high phase disappears here, assume tunneling happens instantly.
		# But ONLY if it doesn't undergo a second-order transition immediately afterwards.
		if T <= self.phases[highIndex]['T'][0] and self.phases[highIndex]['lowLink'] != None:
			for tmin, tmax, lowi, highi in self.transitions:
				if highi == highIndex and tmin==tmax:
					if self.highTunnels != None: self.highTunnels.append((np.inf, T))
					return np.inf # phase undergoes a second-order transition.
			else:
				# Set lowPhases['lowLink'] to have a second-order transition
				i = self.phases[highIndex]['lowLink']
				lowP = lowPhases[i]
				lowP.update(lowT=T, lowFT=pathDeformation.secondOrderTransition(phiHigh,T))
				self.lastLowTunnel = (-np.inf, T)
				return -np.inf # Phase disappears. Assume that it tunnels instantly at this point.

		# Check the overlap for each phase.
		excluded = []
		for i in lowPhases:
			if lowPhases[i]["phi"] == None:
				continue
			di = phiHigh-lowPhases[i]["phi"]
			di_mag = np.sum(di*di)**.5
			for j in lowPhases:
				if i == j or lowPhases[j]["phi"] == None:
					continue
				dj = phiHigh-lowPhases[j]["phi"]
				dj_mag = np.sum(dj*dj)**.5
				if np.sum(di*dj)/(di_mag*dj_mag) > np.cos(self.overlapAngle) and self.overlapAngle > 0:
					# need to exclude one of them
					excluded.append( i if di_mag > dj_mag else j )
		for i in excluded:
			lowPhases[i]["phi"] = None
			
		# Now try and tunnel to each phase.
		rval = None
		V = lambda x,T=T: self.V(x,T)
		dV = lambda x,T=T: self.dV(x,T)
		for i in lowPhases:
			lowP = lowPhases[i]
			if lowP["phi"] == None:
				if T > lowP["highT"] and rval == None:
					rval = np.inf # we're too high
				continue
			path = (lowP["phi"], phiHigh)
			if self.verbose >= 1:
				print "\nTunneling from phase %i to %i at T = %f." % (highIndex, i, T)
				print "phiLow =", lowP["phi"]
				print "phiHigh =", phiHigh
				
			if T == lowP["highT"] and lowP["highFT"] != None: tunnelObj = lowP["highFT"]
			elif T == lowP["lowT"] and lowP["lowFT"] != None: tunnelObj = lowP["lowFT"]
			else:	
				tunnelObj = pathDeformation.fullTunneling(path, V, dV, alpha=2, **self.tunnelParams)
				tunnelObj.T = T
				rcode = tunnelObj.run(**self.tunnelParams)
				if rcode <= -10:
					print "Deformation failed in fullTransations.tunnelingAtT. rcode =", rcode
					continue
			if self.verbose >= 1: print "Tunneling action:", tunnelObj.findAction(), " S3/T:",tunnelObj.findAction()/T
			newrval = self.nuclCriterion(tunnelObj.findAction(), T)
			rval = newrval if (rval == None or newrval < rval) else rval
			if newrval > 0 and T <= lowP["highT"] and updateBracketTemps: # new bound from above
				lowP["highT"] = T
			if newrval < 0 and T >= lowP["lowT"] and updateBracketTemps: # new bound from below
				lowP["lowT"] = T
			if T == lowP["highT"]:
				lowP["highFT"] = tunnelObj
			if T == lowP["lowT"]:
				lowP["lowFT"] = tunnelObj
			
			lowP.update(lastT=T, lastFT=tunnelObj)
			
			if rval < 0:
				break
				
		if rval == None:
			# if we couldn't complete tunneling, assume that there is an error due to thick-walled solution
			# i.e., need to increase temperature
			rval = -np.inf
			
		if rval < 0:
			self.lastLowTunnel = (rval, T)
		elif self.highTunnels != None:
			self.highTunnels.append((rval, T))
		return rval
			
				
			
		
		
		
		

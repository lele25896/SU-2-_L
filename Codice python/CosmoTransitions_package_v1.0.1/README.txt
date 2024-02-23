To install CosmoTransitions, first make sure that you have python, numpy, and scipy installed on your system. If you want to take advantage of the plotting functions, then you'll also need matplotlib. All of these are freely available. The easiest way to get them is to download and install the Enthought python distribution (http://www.enthought.com). See also documentation at http://www.scipy.org. Then you just need to drag the cosmoTransitions source folder to somewhere in your python search path.

For an overview of the program, see the "module_overview.pdf" file. You can also type "help(<function name>)" at the python prompt to get help on a specific function.

To run a simple program, load up a python interactive prompt (by typing "ipython --pylab" in a unix terminal) and try entering the following commands:

# --------------------------
import testModels
m = testModels.model1()

m.getPhases()
figure(1); m.plotPhasesPhi()
axis([0,200,-50,550]); xlabel("$T$"); ylabel("$|\\vec{\\phi}(T)|$")

tctrans = m.calcTcTrans()
tctrans

tntrans = m.calcFullTrans(outTunnelObj=True)
tntrans

figure(2)
m.plot2d((-450,450,-450,450), T=55, cfrac=.4,clevs=65,n=100,lw=.5)
ft = tntrans[-1]['tunnelObj']
plot(ft.phi[:,0], ft.phi[:,1],'k',lw=1.25)
# --------------------------

This will load a simple model from the testModels file (which will also need to be somewhere in your search path), find its phases, plot its phases, find all of its critical temperatures, find all of its phase transitions, and plot its last phase transition. If all goes well, the two figures should look like "testFig_phases.pdf" and "testFig_tran.pdf". The output should look like this:

# --------------------------
In [1]: import testModels

In [2]: m = testModels.model1()

In [3]: 

In [4]: m.getPhases()
Tracing phase starting at x = [ 295.56323185  406.39105679] ; t = 0.0
Tracing minimum up
t0 = 0.0
.....................................................................................................................
Tracing phase starting at x = [ 205.32102486 -146.21546786] ; t = 83.2976200312
Tracing minimum down
t0 = 83.2976200312
..................................................................................................
Tracing minimum up
t0 = 83.2976200312
.......................................................
Tracing phase starting at x = [ 135.1884929  -101.44802474] ; t = 112.190305455
Tracing minimum down
t0 = 112.190305455
........
Tracing minimum up
t0 = 112.190305455
...............................................................................................
Tracing phase starting at x = [  3.54557748e-06   1.75864794e-06] ; t = 129.672551193
Tracing minimum down
t0 = 129.672551193
.....................
Tracing minimum up
t0 = 129.672551193
...............

In [5]: figure(1); m.plotPhasesPhi()
Out[5]: <matplotlib.figure.Figure object at 0x5b67a70>

In [6]: axis([0,200,-50,550]); xlabel("$T$"); ylabel("$|\\vec{\\phi}(T)|$")
Out[6]: [0, 200, -50, 550]
Out[6]: <matplotlib.text.Text object at 0x5b787f0>
Out[6]: <matplotlib.text.Text object at 0x5b882f0>

In [7]: 

In [8]: tctrans = m.calcTcTrans()

In [9]: tctrans
Out[9]: 
[{'Tcrit': 128.18393432558665,
  'alpha': 11617509.243130691,
  'dphi': 50.548014054851961,
  'high vev': array([ 0.00051349, -0.0003934 ]),
  'low vev': array([ 40.25700573, -30.57025696])},
 {'Tcrit': 112.18775642474105,
  'alpha': 1813242.8407906268,
  'dphi': 1.6520467049387078,
  'high vev': array([ 135.19655621, -101.45392822]),
  'low vev': array([ 136.50463982, -102.46297508])},
 {'Tcrit': 75.19340413462272,
  'alpha': 591430266.16218328,
  'dphi': 498.5305156621742,
  'high vev': array([ 214.57581463, -149.36120917]),
  'low vev': array([ 273.47156235,  345.67814881])}]

In [10]: 

In [11]: tntrans = m.calcFullTrans(outTunnelObj=True)

Tunneling from phase 3 to 2 at T = 128.183934.
phiLow = [ 40.25702363 -30.57022685]
phiHigh = [ 0.00052232 -0.00039709]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 148492.05602  S3/T: 1158.42953956

Tunneling from phase 3 to 2 at T = 128.183934.
phiLow = [ 40.25702363 -30.57022685]
phiHigh = [ 0.00052232 -0.00039709]
Tunneling action: 148492.05602  S3/T: 1158.42953956

Tunneling from phase 3 to 2 at T = 128.183366.
phiLow = [ 40.27450308 -30.58347261]
phiHigh = [ 0.00053061 -0.00039786]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 448382823.022  S3/T: 3497979.78272

Tunneling from phase 3 to 2 at T = 127.899526.
phiLow = [ 48.22224839 -36.60260848]
phiHigh = [  1.13382632e-04  -8.41768079e-05]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 3402.28034522  S3/T: 26.6011958081

Tunneling from phase 3 to 2 at T = 127.900095.
phiLow = [ 48.20882549 -36.59245338]
phiHigh = [  4.99890060e-05  -3.48180757e-05]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
4 Updating the bubble profile.
Deforming the path.
Path deformation converged.
5 Updating the bubble profile.
Deforming the path.
Path deformation converged.
6 Updating the bubble profile.
Deforming the path.
Path deformation converged.
7 Updating the bubble profile.
Deforming the path.
Path deformation converged.
8 Updating the bubble profile.
Deforming the path.
Path deformation converged.
9 Updating the bubble profile.
Deforming the path.
Path deformation converged.
10 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 3414.38321593  S3/T: 26.6957051568

Tunneling from phase 3 to 2 at T = 128.041730.
phiLow = [ 44.45014279 -33.74689611]
phiHigh = [ 0.00157702 -0.00120378]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 11023.2144183  S3/T: 86.0907954584

Tunneling from phase 3 to 2 at T = 128.112548.
phiLow = [ 42.4020341  -32.19555537]
phiHigh = [ 0.00094098 -0.00071959]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 36421.9934031  S3/T: 284.296846093

Tunneling from phase 3 to 2 at T = 128.060992.
phiLow = [ 43.90028001 -33.33045533]
phiHigh = [ -9.67152166e-05   8.06959498e-05]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 14110.9596071  S3/T: 110.18936687

Tunneling from phase 3 to 2 at T = 128.080739.
phiLow = [ 43.33110412 -32.89935073]
phiHigh = [  5.81292905e-06  -7.45843603e-06]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 19061.6092681  S3/T: 148.824947446

Tunneling from phase 3 to 2 at T = 128.076229.
phiLow = [ 43.46156667 -32.99816767]
phiHigh = [  8.03740265e-05  -6.23762350e-05]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 17701.5028812  S3/T: 138.210681902

Tunneling from phase 3 to 2 at T = 128.076989.
phiLow = [ 43.43964767 -32.98155963]
phiHigh = [ 0.00123716 -0.00094527]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 17920.2386156  S3/T: 139.917707058

Tunneling from phase 3 to 2 at T = 128.077557.
phiLow = [ 43.42315534 -32.96906758]
phiHigh = [ 0.00120212 -0.00090747]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 18086.7989277  S3/T: 141.217550745
Found a transition from 3 to 2 at T = 128.076988914

Tunneling from phase 2 to 1 at T = 112.187756.
phiLow = [ 136.50484608 -102.46330795]
phiHigh = [ 135.1964229  -101.45379493]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 19174237167.7  S3/T: 170912029.786

Tunneling from phase 2 to 1 at T = 112.187756.
phiLow = [ 136.50484608 -102.46330795]
phiHigh = [ 135.1964229  -101.45379493]
Tunneling action: 19174237167.7  S3/T: 170912029.786

Tunneling from phase 2 to 1 at T = 112.187751.
phiLow = [ 136.50488986 -102.46334016]
phiHigh = [ 135.19643165 -101.45380269]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 21383697.5037  S3/T: 190606.34796

Tunneling from phase 2 to 1 at T = 112.185041.
phiLow = [ 136.52781174 -102.48041624]
phiHigh = [ 135.20510545 -101.46014277]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 85.6416177475  S3/T: 0.763396058356

Tunneling from phase 2 to 1 at T = 112.185046.
phiLow = [ 136.52776354 -102.48038594]
phiHigh = [ 135.20508615 -101.46012381]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 85.9884235796  S3/T: 0.76648739452

Tunneling from phase 2 to 1 at T = 112.186398.
phiLow = [ 136.51631536 -102.47185078]
phiHigh = [ 135.20073163 -101.45694801]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 340.47657768  S3/T: 3.03491851329

Tunneling from phase 2 to 1 at T = 112.187075.
phiLow = [ 136.51057587 -102.46757409]
phiHigh = [ 135.19858927 -101.45538468]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 1346.04696415  S3/T: 11.9982356906

Tunneling from phase 2 to 1 at T = 112.187413.
phiLow = [ 136.50772169 -102.46545235]
phiHigh = [ 135.19750777 -101.45458982]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 5289.17416023  S3/T: 47.1458787132

Tunneling from phase 2 to 1 at T = 112.187582.
phiLow = [ 136.50632309 -102.46440711]
phiHigh = [ 135.19697961 -101.4542074 ]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 20488.2050641  S3/T: 182.62453572

Tunneling from phase 2 to 1 at T = 112.187529.
phiLow = [ 136.50674528 -102.46471935]
phiHigh = [ 135.19713964 -101.45432058]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 12034.6407902  S3/T: 107.272536665

Tunneling from phase 2 to 1 at T = 112.187552.
phiLow = [ 136.50655988 -102.46458682]
phiHigh = [ 135.19707828 -101.45427706]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 14904.2810396  S3/T: 132.851468765

Tunneling from phase 2 to 1 at T = 112.187557.
phiLow = [ 136.50652628 -102.46456066]
phiHigh = [ 135.19705275 -101.45425137]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 15744.7626742  S3/T: 140.343216689
Found a transition from 2 to 1 at T = 112.187557373

Tunneling from phase 1 to 0 at T = 75.193404.
phiLow = [ 273.47135575  345.67791698]
phiHigh = [ 214.5756923  -149.36116038]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 1.38972563041e+15  S3/T: 1.8482015097e+13

Tunneling from phase 1 to 0 at T = 75.193404.
phiLow = [ 273.47135575  345.67791698]
phiHigh = [ 214.5756923  -149.36116038]
Tunneling action: 1.38972563041e+15  S3/T: 1.8482015097e+13

Tunneling from phase 1 to 0 at T = 75.169540.
phiLow = [ 273.51577046  345.8094771 ]
phiHigh = [ 214.60012971 -149.36571448]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 6219593966.8  S3/T: 82740880.8844

Tunneling from phase 1 to 0 at T = 63.249474.
phiLow = [ 287.37178366  384.81230522]
phiHigh = [ 225.18508483 -147.290781  ]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 43271.8014586  S3/T: 684.144838171

Tunneling from phase 1 to 0 at T = 57.289442.
phiLow = [ 290.81871209  393.99012389]
phiHigh = [ 229.55801276 -140.52484559]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 15364.4458306  S3/T: 268.189833974

Tunneling from phase 1 to 0 at T = 54.309425.
phiLow = [ 292.04575099  397.21556123]
phiHigh = [ 231.66410722 -134.15177188]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 7479.65428649  S3/T: 137.722950712

Tunneling from phase 1 to 0 at T = 54.361436.
phiLow = [ 292.02668262  397.16559417]
phiHigh = [ 231.62680183 -134.30326001]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 7602.42977485  S3/T: 139.849687271

Tunneling from phase 1 to 0 at T = 54.385300.
phiLow = [ 292.01790273  397.14259817]
phiHigh = [ 231.60970002 -134.37207049]
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Tunneling action: 7658.22509889  S3/T: 140.814248505
Found a transition from 1 to 0 at T = 54.361435647
**** Time for finding fullTransitions: 56.0183670521 

Calculating beta/H
Finding rate just above Tnuc (T = 128.080883346)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
3 Updating the bubble profile.
Deforming the path.
Path deformation converged.

Finding rate just below Tnuc (T = 128.073094481)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Calculating beta/H
Finding rate just above Tnuc (T = 112.187586779)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.

Finding rate just below Tnuc (T = 112.187527968)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.
Calculating beta/H
Finding rate just above Tnuc (T = 54.5148841322)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.

Finding rate just below Tnuc (T = 54.2079871618)...
1 Updating the bubble profile.
Deforming the path.
Path deformation converged.
2 Updating the bubble profile.
Deforming the path.
Path deformation converged.

In [12]: tntrans
Out[12]: 
[{'S3': 17920.238615610819,
  'Tnuc': 128.07698891352695,
  'alpha': 13495441.845742226,
  'betaH': 289688.6680106991,
  'dphi': 54.538827523614529,
  'high vev': array([ 0.00123716, -0.00094527]),
  'low vev': array([ 43.43864181, -32.98087227]),
  'tranType': 1,
  'tunnelObj': <cosmoTransitions.pathDeformation.fullTunneling instance at 0x52c1d00>},
 {'S3': 15744.762674243495,
  'Tnuc': 112.18755737346311,
  'alpha': 1814694.2879476547,
  'betaH': 165318241.93389976,
  'dphi': 1.6533548681170207,
  'high vev': array([ 135.19719128, -101.4543935 ]),
  'low vev': array([ 136.5063202 , -102.46422704]),
  'tranType': 1,
  'tunnelObj': <cosmoTransitions.pathDeformation.fullTunneling instance at 0x5bc56e8>},
 {'S3': 7602.4297748523832,
  'Tnuc': 54.36143564703841,
  'alpha': 361539399.89478821,
  'betaH': 2227.1030345079412,
  'dphi': 534.89031417928879,
  'high vev': array([ 231.62690997, -134.30371958]),
  'low vev': array([ 292.0268236 ,  397.16546922]),
  'tranType': 1,
  'tunnelObj': <cosmoTransitions.pathDeformation.fullTunneling instance at 0x52ddd00>}]

In [13]: figure(2)
Out[13]: <matplotlib.figure.Figure object at 0x5ab7030>

In [14]: m.plot2d((-450,450,-450,450), T=55, cfrac=.4,clevs=65,n=100,lw=.5)

In [15]: ft = tntrans[-1]['tunnelObj']

In [16]: plot(ft.phi[:,0], ft.phi[:,1],'k',lw=1.25)
Out[16]: [<matplotlib.lines.Line2D object at 0x5989cf0>]

# --------------------------


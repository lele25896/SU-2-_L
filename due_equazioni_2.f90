! Il codice attualmente risolve 4 equazioni
! L'equazione 1 è la W-H e funziona perfettamente
! L'equazione 2 è quella proper time parametrica e anche lei funziona solo devo sistemare il calcolo del termine di volume quindi attualmente è inserito a mano
! L'equazione 3 è la W-H in LPA' con z field independent. Funziona solo per casi simmetrici e con il numero di punti della griglia fissato a 1501 perché non ho inserito la ricerca del minimo
! L'equazione 4 deve risolvere W-H con Z e ci sto lavorando, attualmente non funziona



IMPLICIT REAL(8)(a-h,o-z)
REAL(8) :: lamb,mm,k,k2, volume, radice_volume, md
REAL(8) max_conv, current_conv, min_val, uderivata
INTEGER :: i, min_idx, dim, equation_selector
PARAMETER(ndim=3100)
DIMENSION u(ndim),a(ndim),b(ndim),f(ndim),z(ndim)
DIMENSION r(ndim),ut(ndim),zt(ndim), upredictor(ndim)
DIMENSION zpredictor(ndim), zbare(ndim), zrenormalized(ndim)
REAL (8) :: uo(-ndim:ndim),zo(-ndim:ndim), chem(-ndim:ndim), om(-ndim:ndim)
REAL (8) :: s2(-ndim:ndim),potenziale(-ndim:ndim)

COMMON/dati/h,dt
COMMON/dimens/np


pigreco=3.1415
equazione=4

dim=1.0
md=2.0
beta=md-dim/2.D0
if (equazione==1) THEN
volume=2.0/((4.0*pigreco)**(dim/2.0)*gamma(dim/2.0))
else if (equazione==2) then
volume=0.353553
else if (equazione==3)then
  volume=2.0/((4.0*pigreco)**(dim/2.0)*gamma(dim/2.0))
else if (equazione==4)then
  volume=2.0/((4.0*pigreco)**(dim/2.0)*gamma(dim/2.0))
!volume=md**(dim/2.0)*gamma(md-dim/2.0)/(gamma(md)*2**(dim)*pigreco**(dim/2.0))
end if
radice_volume= SQRT(volume)
WRITE(*,*) volume


g = 0.15
mm = 1.0/2.0
!lamb=0.1*volume
!asim1=0.1/radice_volume
!asim3=0
!cost=0
!lamb = 0.5*(g**2)*volume !Questo è il termine di volume (il K dell articolo nel caso d=1)
!asim3 = (-g)*radice_volume
!asim1 = g/radice_volume
!cost = (-0.5)/volume

nmax=10000            !massimo numero di iterazioni
itermin=1000        !minimo numero di terazioni
ikmax=12000           !scritture parziali su fort.20
k=1.d0/100.d0        !spaziatura t
np=1501            !numero punti di griglia
lo=7         !posizione iniziale
l1=7
h=2*lo/(np-1.d0)        !spaziatura x

t=0
n=0
icount=0


rap=h**2/k
k2=2.d0/k

!     inizializzazione Qui il codice inizializza il vettore da -7 a 7, aumentando con il passo spaziale da 1 a NP
DO  i=1,np
  x=-lo+h*(i-1)
  om(i)=cost + asim1*x + mm*x**2 + asim3*x**3 + lamb*x**4
  ctmp=2.d0*mm + 6.d0*asim3*x + 12.d0*lamb*x**2
  if (equazione==1) THEN
    uo(i)=DLOG(1.d0+ctmp*DEXP(2*t))
  else if (equazione==2) then 
    uo(i)=((1.0+DEXP(2*t)*ctmp/MD))**(-BETA)
  else if (equazione==3) then
    zbare(i) = 1
    uo(i) = DLOG(zbare(i)+ctmp*DEXP(2*t))
  else if (equazione==4) then
    zo(i) = 1
    uo(i) = DLOG(zo(i)+ctmp*DEXP(2*t))
  end if
END DO
10   n=n+1

!     PREDICTOR

t=t+k/2.d0

if (equazione==1) THEN
visc=DEXP((dim-2.d0)*t)
else if (equazione==2) then
VISC=DEXP((dim-2.D0)*T)*MD
else if (equazione==3) THEN
visc=DEXP((dim-2.d0)*t)
else if (equazione==4) THEN
visc=DEXP((dim-2.d0)*t)
end if


if (equazione==1) THEN
  DO  i=2,np-1
    fi=uo(i)
    espo=DEXP(fi)
    a(i-1)=1.d0
    f(i-1)=1.d0
    b(i-1)=-2.d0-visc*4*rap*espo
    r(i-1)=4.d0*h*h*visc*(1.d0-espo)-4.d0*visc*rap*uo(i)*espo
  END DO

else if (equazione==2) then
  DO  i=2,np-1
  ESPO=sign(abs(UO(I))**(-1/BETA),uo(i))
  ESPO1=sign(abs(UO(I))**(-1/BETA-1.D0)/BETA,uo(i))
  A(I-1)=1.D0
  F(I-1)=1.D0
  B(I-1)=-2.D0-VISC*2*RAP*ESPO1
  R(I-1)=-2.D0*H*H*VISC*(1.D0-ESPO)-2.D0*VISC*RAP*UO(I)*ESPO1
  END DO

else if (equazione==3) then 
  DO  i=2,np-1
    fi=uo(i)
    espo=DEXP(fi)
    a(i-1)=1.d0
    f(i-1)=1.d0
    b(i-1)=-2.d0-visc*4*rap*espo
    r(i-1)=4.d0*h*h*visc*(zbare(i)-espo)-4.d0*visc*rap*uo(i)*espo
  END DO
else if (equazione==4) then 
  DO  i=2,np-1
    fi=uo(i)
    espo=DEXP(fi)
    a(i-1)=1.d0
    f(i-1)=1.d0
    b(i-1)=-2.d0-visc*4*rap*espo
    r(i-1)=4.d0*h*h*visc*(zo(i)-espo)-4.d0*visc*rap*uo(i)*espo
  END DO
end if

!   BOUNDARY

u(np)=uo(np)

u(1) = uo(1)   ! Condizione al contorno a -L



!   CORREZIONI AL BOUNDARY

f(1)=2.d0
r(np-2)=r(np-2)-u(np)



CALL tridag(a, b, f, r, ut, np-2)

DO  i=1,np-2
  u(i+1)=ut(i)
  upredictor(i+1)=ut(i)
END DO
upredictor(1)=u(1)
upredictor(np)=u(np)

if (equazione==3) then
  uderivata=(u(752)-u(750))/(2.0*h)
  
  
  do i=2, np-1
    espo=DEXP(4*t)*DEXP(-2*u(751))
    espo1=DEXP(2*t)*DEXP(-2*u(751))
    zpredictor(i)=zbare(i)+k2*(0.106103*espo*zbare(i)*uderivata)-k2*(0.31831*espo1*zbare(i)*(uderivata**2))
  end do
  zpredictor(1)=1
  zpredictor(np)=1

end if

if (equazione==4) then
  DO  i=2,np-1
    fi=uo(i)
    espo=DEXP(fi)
    a(i-1)=1.d0
    f(i-1)=1.d0
    b(i-1)=-2.d0-4.0*pigreco*visc*espo*rap
    r(i-1)=-4.0*rap*visc*espo*pigreco*zo(i)
  END DO

z(np)= 1

z(1) = 1

f(1)=2.d0
r(np-2)=r(np-2)-z(np)

CALL tridag(a, b, f, r, zt, np-2)

DO  i=1,np-2
  z(i+1)=zt(i)
END DO

end if


!    CORRECTOR


if (equazione==1) THEN
DO  i=2,np-1
  fi=u(i)
  espo=DEXP(fi)
  d2=uo(i-1)-2.*uo(i)+uo(i+1)
 a(i-1)=1.d0/2.d0
 f(i-1)=1.d0/2.d0
 b(i-1)=-1.d0-visc*2*rap*espo
 r(i-1)=4.d0*h*h*visc*(1.d0-espo)- 2.d0*visc*rap*uo(i)*espo-d2/2.d0
END DO

else if (equazione==2) then
  DO  i=2,np-1
    FI=U(I)
    ESPO=sign(abs(fi)**(-1/BETA),fi)
    ESPO1=sign(abs(fi)**(-1/BETA-1.D0)/BETA,fi)
    D2=UO(I-1)-2.*UO(I)+UO(I+1)
    A(I-1)=1.D0/2.D0
    F(I-1)=1.D0/2.D0
    B(I-1)=-1.D0-VISC*RAP*ESPO1
    R(I-1)=-2.D0*H*H*VISC*(1.D0-ESPO)-VISC*RAP*UO(I)*ESPO1-D2/2.D0
  END DO


else if (equazione==3) THEN
  DO  i=2,np-1
    fi=u(i)
    espo=DEXP(fi)
    d2=uo(i-1)-2.*uo(i)+uo(i+1)
    a(i-1)=1.d0/2.d0
    f(i-1)=1.d0/2.d0
    b(i-1)=-1.d0-visc*2*rap*espo
    r(i-1)= 4.d0*h*h*visc*(zpredictor(i)-espo) - 2.d0*visc*rap*uo(i)*espo-d2/2.d0
  END DO


else if (equazione==4) THEN
  DO  i=2,np-1
    fi=u(i)
    espo=DEXP(fi)
    d2=uo(i-1)-2.*uo(i)+uo(i+1)
    a(i-1)=1.d0/2.d0
    f(i-1)=1.d0/2.d0
    b(i-1)=-1.d0-visc*2*rap*espo
    r(i-1)=4.d0*h*h*visc*(z(i)-espo)- 2.d0*visc*rap*uo(i)*espo-d2/2.d0
  END DO

end if

!   BOUNDARY


ctmp1 = 2.d0*mm - 6.d0*asim3*l1 + 12.d0*lamb*l1**2
ctmp2 = 2.d0*mm - 6.d0*asim3*l1 + 12.d0*lamb*l1**2

if (equazione==1) THEN
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))

else if (equazione==2) then
  u(np)=((1.0+DEXP(2*t)*ctmp2/MD))**(-BETA)
  u(1)=((1.0+DEXP(2*t)*ctmp1/MD))**(-BETA)

else if (equazione==3) then
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))

else if (equazione==4) then
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))
end if

!   CORREZIONI AL BOUNDARY

f(1)=1.d0
r(np-2)=r(np-2)-u(np)/2.d0

CALL tridag(a,b,f,r,ut,np-2)

DO  i=1,np-2
  u(i+1)=ut(i)
END DO





if (equazione==3) then
  uderivata=(u(752)-u(750))/(2.0*h)

  do i=2, np-1
    espo=DEXP(4*t)*DEXP(-2*u(751))
    espo1=DEXP(2*t)*DEXP(-2*u(751))
    zrenormalized(i) = zpredictor(i)+k2*(0.106103*espo*zpredictor(i)*uderivata)-k2*(0.31831 * espo1 * zpredictor(i)*uderivata**2)
  end do
  zrenormalized(1)=1
  zrenormalized(np)=1
end if


if (equazione==4) then
  DO  i=2,np-1
    fi=upredictor(i)
    espo=DEXP(fi)
    d2=zo(i-1)-2.0*zo(i)+zo(i+1)
    a(i-1)=1.d0/2.d0
    f(i-1)=1.d0/2.d0
    b(i-1)=-1.d0-(1.0)*rap*visc*espo*pigreco
    r(i-1)=-2.0*rap*visc*espo*pigreco*zo(i)-d2/2.d0
  END DO
z(np)= 1

z(1) = 1

f(1)=1.d0
r(np-2)=r(np-2)-z(np)/2.0

CALL tridag(a, b, f, r, zt, np-2)

DO  i=1,np-2
  z(i+1)=zt(i)
END DO

end if
t=t+k/2.0
if (equazione==1) THEN
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))

else if (equazione==2) then
  u(np)=((1.0+DEXP(2*t)*ctmp2/MD))**(-BETA)
  u(1)=((1.0+DEXP(2*t)*ctmp1/MD))**(-BETA)

else if (equazione==3) then
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))

else if (equazione==4) then
  u(np)=DLOG(1.d0+ctmp2*DEXP(2*t))
  u(1)=DLOG(1.d0+ctmp1*DEXP(2*t))
  z(1)=1
  z(np)=1
end if



  !    END OF INTEGRATION

WRITE(20,*) n, u(751), z(751), zo(751)
!    UPDATING OF CHEMICAL POTENTIAL AND FREE ENERGY




fatcho=DEXP(-dim*(t-k))
fatch=DEXP(-dim*t)
DO i=1,np
  if(equazione==1) then
    om(i)=om(i)+0.25*(fatcho*uo(i)+fatch*(u(i)))*k
  else if (equazione==2) then
    OM(I)=OM(I)+0.5D0*(fatcho*uo(i)+FATCH*(U(I)))*K
  else  if(equazione==3) then
      om(i)=om(i)+0.25*(fatcho*uo(i)+fatch*(u(i)))*k
  end if
END DO


max_conv = 0.0D0
DO  i = 215, 1287
  yyy = u(i)
  yyz = uo(i)
  if (equazione==1) THEN
    s0 = (EXP(-2.d0*t))*(DEXP(yyy) - 1.0)
    s00 = EXP(-2.d0*(t-k))*(DEXP(yyz) - 1.0)
  else if (equazione==2) then
    s0=(EXP(2.D0*T))/sign(abs(yyy)**(-1/BETA)-1.0,yyy)
    s00=EXP(2.D0*(T-K))/sign(abs(yyz)**(-1/BETA)-1.0,yyz)
  
  else if(equazione==3) then
    s0 = (EXP(-2.d0*t))*(DEXP(yyy) - z(i))
    s00 = EXP(-2.d0*(t-k))*(DEXP(yyz) - zo(i)) 
  
  else if(equazione==4) then
    s0 = (EXP(-2.d0*t))*(DEXP(yyy) - z(i))
    s00 = EXP(-2.d0*(t-k))*(DEXP(yyz) - zo(i)) 
  end if

  current_conv = ABS(s0/s00 - 1.d0) / k
  
  IF (current_conv > max_conv) THEN
    max_conv = current_conv
    max_conv_idx = i
  END IF
END DO

conv = max_conv



icount=icount+1
IF(icount >= ikmax) THEN
  WRITE(20,*)
  WRITE(20,*)'N=',n,'    T=',t
  WRITE(20,*)
  icount=0
  chint=0.d0
  fint=0.d0
  DO  i=2,np-1
    x=(i-2)*h
    fi=u(i)
    pm=(DEXP(fi)-1.d0)*DEXP(-2.d0*t)
    derch=(chem(i)-chem(i-1))/h
    chint=chint+pm*h*fint
    fint=0.5D0
    chint=chint+pm*h*fint
!             WRITE(20,602)X,PM,OM(I)
  END DO
END IF

IF(n >= nmax)GO TO 777

DO  i=1,np
  uo(i)=u(i)
 if(equazione==4) then
  zo(i)=z(i)
 end if
end do
zbare(1)=1
zbare(np)=1


IF(conv >= 1.e-7.OR.n < itermin) GO TO 10

777  CONTINUE

DO  i=1,np
  x=(-l1+h*(i-1))*radice_volume
  potenziale(i)=om(i)*volume
  WRITE(20,27)x,potenziale(i)
END DO



min_val = potenziale(1)
min_idx = 1

DO i = 2, np
    IF (potenziale(i) < min_val) THEN
        min_val = potenziale(i)
        min_idx = i
    END IF
END DO

if (equazione==1) THEN
  s1 = SQRT((EXP(-2.d0*t))*(DEXP(u(min_idx)) - 1.0))
else if (equazione==2) then
  yyy=u(min_idx)
  s0=(EXP(2.D0*T))/sign(abs(yyy)**(-1/BETA)-1.0,yyy)
  s1 = SQRT(1/s0)
else if (equazione==3) then
  s1 = SQRT((EXP(-2.d0*t))*(DEXP(u(min_idx)) - z(min_idx)))
else if (equazione==4) then
  s1 = SQRT((EXP(-2.d0*t))*(DEXP(u(min_idx)) - z(min_idx)))
  WRITE(*,*)z(min_idx)
end if

e1=min_val+s1

! Stampa il valore minimo e l'indice corrispondente

!PRINT *, "Indice del valore minimo:", min_idx
!PRINT *, "Posizione spaziale del minimo:", x
PRINT *, "Valore minimo di potenziale:", min_val, om(751)
PRINT *, "Energia primo livello:", e1
PRINT *, "Gap energetico:", s1, min_idx






!PRINT*,'TIME STEP',k,'    SPACE STEP=',h
!PRINT*,'CONV=',conv,'  ITER=',n
!PRINT*,'VALORE FINALE =',s1   ! , 'U(Min)' ,MIN_VAL, 'indice', MIN_IDX
WRITE(20,*)'#T=',t


601  FORMAT(1X,4(e14.8,3X))
602  FORMAT(1X,3(e14.8,3X))
27   FORMAT(1X,2(e14.8,3X))

END



SUBROUTINE tridag(a,b,f,r,u,n)

REAL(8), INTENT(IN) :: a(n), b(n), f(n), r(n)
REAL(8), INTENT(OUT) :: u(n)
INTEGER, INTENT(IN)                      :: n
INTEGER, PARAMETER :: nmax=1000000
REAL(8) :: gam(nmax)
REAL(8) :: bet
INTEGER :: j

IF(b(1) == 0.d0)STOP
bet=b(1)
u(1)=r(1)/bet
DO  j=2,n
  gam(j)=f(j-1)/bet
  bet=b(j)-a(j)*gam(j)
  IF(bet == 0.d0)STOP
  u(j)=(r(j)-a(j)*u(j-1))/bet
END DO
DO  j=n-1,1,-1
  u(j)=u(j)-gam(j+1)*u(j+1)
END DO
RETURN
END SUBROUTINE tridag



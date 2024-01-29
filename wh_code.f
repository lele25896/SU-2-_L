
c
c    integrazione delle equazioni per la phi4
c
c     senza rinormalizzazione della densita'
c
c     predictor/corrector
c
      
      

      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 LAMB,MM,K,K2
      REAL*8 MAX_CONV, CURRENT_CONV, MIN_VAL
      INTEGER MAX_CONV_IDX, MIN_IDX
      PARAMETER(NDIM=3100)
      DIMENSION U(NDIM),A(NDIM),B(NDIM),C(NDIM)
      DIMENSION R(NDIM),UT(NDIM)
      REAL  UO(-NDIM:NDIM), CHEM(-NDIM:NDIM), OM(-NDIM:NDIM)
      
      
      COMMON/DATI/H,DIM,DT
      COMMON/DIMENS/NP

      
       DIM=1
       MM=1.0/2
       LAMB=0.5/3.14 !Questo è il termine di volume (il K dell articolo nel caso d=1)
       ASIM=0.5/1.77245

      WRITE(6,*)'DIMENSION=',DIM,'    R0=',MM,'   U0=',LAMB
      
      NMAX=3000            !massimo numero di iterazioni
      ITERMIN=1000         !minimo numero di terazioni
      IKMAX=4000           !scritture parziali su fort.20
      K=1.D0/100.D0        !spaziatura t
      NP=1501             !numero punti di griglia
      LO=7.D0                !posizione iniziale
      H=2*LO/(NP-1.D0)        !spaziatura x
      
      T=-7
      N=0
      ICOUNT=0               

      NSAFE=5                                     
      RAP=H**2/K
      K2=K/2.D0
      
      WRITE(*,*)'#  R0=',MM,'   U0=',LAMB
      

      
c
c     inizializzazione Qui il codice inizializza il vettore da -7 a 7, aumentando con il passo spaziale da 1 a NP
c
      DO 1 I=1,NP
      X=-LO+H*(I-1)
      CTMP=2.D0*MM+12.D0*LAMB*X*X + 6.D0*ASIM*X
      UO(I)=DLOG(1.D0+CTMP*DEXP(2*T))
      Y=X+H/2.D0
      CHEM(I)=2.D0*MM*Y+4.D0*LAMB*Y*Y*Y + 3.D0*ASIM*Y*Y*Y
      OM(I)=MM*X**2+LAMB*X**4+ASIM*X**3
c      WRITE(20,602)X,OM(I),UO(I)
 1    CONTINUE
c      WRITE(20,*)
c      WRITE(20,*)
 10   N=N+1

c     PREDICTOR

      T=T+K/2.D0

      VISC=DEXP((DIM-2.D0)*T)
      ET=1.D0

      DO 2 I=2,NP-1
       FI=ET*UO(I)
       ESPO=DEXP(FI)
       A(I-1)=1.D0
       C(I-1)=1.D0
       B(I-1)=-2.D0-VISC*4*RAP*ESPO
       R(I-1)=4.D0*H*H*VISC*(1.D0-ESPO)-4.D0*VISC*RAP*UO(I)*ESPO
 2    CONTINUE

C   BOUNDARY

c      DT=K2

C     ESTENSIONE A LEGGE DI POTENZA
      
      
      U(NP)=UO(NP)
      
      U(1) = UO(1)   ! Condizione al contorno a -L
      


C   CORREZIONI AL BOUNDARY 

      C(1)=2.D0
      R(NP-2)=R(NP-2)-U(NP)

      CALL TRIDAG(A,B,C,R,UT,NP-2)

      DO 3 I=1,NP-2
      U(I+1)=UT(I)
 3    CONTINUE


C    CORRECTOR

      DO 4 I=2,NP-1
       FI=ET*U(I)
       ESPO=DEXP(FI)
       D2=UO(I-1)-2.*UO(I)+UO(I+1)
       A(I-1)=1.D0/2.D0
       C(I-1)=1.D0/2.D0
       B(I-1)=-1.D0-VISC*2*RAP*ESPO
       R(I-1)=4.D0*H*H*VISC*(1.D0-ESPO)- 
     &         2.D0*VISC*RAP*UO(I)*ESPO-D2/2.D0
 4    CONTINUE

C   BOUNDARY

      T2=T+K/2.D0
      DT=K

C     ESTENSIONE A LEGGE DI POTENZA

      
      
      U(NP)=UO(NP)
      U(1)=UO(1)
      
      
C   CORREZIONI AL BOUNDARY 

      C(1)=1.D0
      R(NP-2)=R(NP-2)-U(NP)/2.D0

      CALL TRIDAG(A,B,C,R,UT,NP-2)

      DO 5 I=1,NP-2
      U(I+1)=UT(I)
 5    CONTINUE


      T=T+K/2.D0
      

C    END OF INTEGRATION

      

C    UPDATING OF CHEMICAL POTENTIAL AND FREE ENERGY

         FATCHO=DEXP(-DIM*(T-K))
         FATCH=DEXP(-DIM*T)
         DO I=1,NP-1
          FO1=0.5D0*FATCHO*UO(I)
          FO2=0.5D0*FATCHO*UO(I+1)
          F1=0.5D0*FATCH*U(I)
          F2=0.5D0*FATCH*U(I+1)
          DO=(FO2-FO1)/H
          DD=(F2-F1)/H
          CHEM(I)=CHEM(I)+0.5D0*(DO+DD)*K

          OM(I)=OM(I)+0.25D0*(FATCHO*(UO(I)-2.D0*(T-K))+
     &          FATCH*(U(I)-2.D0*T))*K
         ENDDO
      MAX_CONV = 0.0D0
      DO 200 I = 215, 1287
         YYY = U(I)
         YYZ = UO(I)
         S0 = (EXP(2.D0*T)) / (DEXP(YYY) - 1.0)
         S00 = EXP(2.D0*(T-K)) / (DEXP(YYZ) - 1.0)
         CURRENT_CONV = ABS(S0/S00 - 1.D0) / K

         IF (CURRENT_CONV .GT. MAX_CONV) THEN
            MAX_CONV = CURRENT_CONV
            MAX_CONV_IDX = I
         END IF
 200   CONTINUE

       CONV = MAX_CONV
      

         LO=LO+1
       IF(LO.GE.200) THEN

         LO=-7
         PRINT*,N,'  T= ',T
         PRINT*,'CONV=',CONV    
         PRINT*,'U(0)=',U(751),'    S0=',S0
      

       ENDIF
      ICOUNT=ICOUNT+1
      IF(ICOUNT.GE.IKMAX) THEN
      WRITE(20,*)
      WRITE(20,*)'N=',N,'    T=',T
      WRITE(20,*)
      ICOUNT=0
      CHINT=0.D0
      FINT=0.D0
      DO 92 I=2,NP-1
             X=(I-2)*H
             FI=U(I)
             PM=(DEXP(FI)-1.D0)*DEXP(-2.D0*T)
             DERCH=(CHEM(I)-CHEM(I-1))/H
             CHINT=CHINT+PM*H*FINT
             FINT=0.5D0
             CHINT=CHINT+PM*H*FINT
             WRITE(20,602)X,PM,OM(I)
 92   CONTINUE
      ENDIF

      IF(N.GE.NMAX)GOTO 777

      DO 40 I=1,NP
      UO(I)=U(I)
 40   CONTINUE

      X = -LO
      CTMP1 = 2.D0*MM + 12.D0*LAMB*7*7 - 6.D0*ASIM*7
      UO(1) = DLOG(1.D0 + CTMP1*DEXP(2*T))

      X = LO
      CTMP2 = 2.D0*MM + 12.D0*LAMB*7*7 + 6.D0*ASIM*7
      UO(NP) = DLOG(1.D0 + CTMP2*DEXP(2*T))

      
      IF(CONV.GE.1.E-5.OR.N.LT.ITERMIN) GO TO 10

 777  CONTINUE
      DO  I=1,NP
             X=-7+H*(I-1)
             WRITE(20,27)x,U(I)
      END DO
      MIN_VAL = U(1)
      MIN_IDX = 1
       DO 100 I=2,NP
        IF (U(I) .LT. MIN_VAL) THEN
          MIN_VAL = U(I)
          MIN_IDX = I
        END IF
100   CONTINUE
      S1 = (EXP(2.D0*T)) / (DEXP(MIN_VAL) - 1.0)
      PRINT*,'TIME STEP',K,'    SPACE STEP=',H
      PRINT*,'CONV=',CONV,'  ITER=',N
      PRINT*,'VALORE FINALE =',S1, 'U(Min)' ,MIN_VAL, 'indice', MIN_IDX
      CHINT=0.D0
      FINT=0.D0
      WRITE(20,*)'#T=',T
c      DO 920 I=1,NP
c             X=-7+H*(I-1)
c             FI=U(I)
c             PM=(DEXP(FI)-1.D0)*DEXP(-2.D0*T)
c             DERCH=(CHEM(I)-CHEM(I-1))/H
c             CHINT=CHINT+PM*H*FINT
c             WRITE(20,601)x,PM,OM(I),U(I)
c             FINT=0.5D0
c             CHINT=CHINT+PM*H*FINT
c 920    CONTINUE
        
        STOP

 601  format(1x,4(e14.8,3x))
 602  format(1x,3(e14.8,3x))
 27   format(1x,2(e14.8,3x))
 28   format(1x,5(e12.6,3x))
 29   format(1x,i3,3x,3(d12.6,3x))
      END


      SUBROUTINE TRIDAG(A,B,C,R,U,N)
      IMPLICIT REAL*8(A-H,O-Z)
      PARAMETER (NMAX=3100)
      DIMENSION GAM(NMAX),A(N),B(N),C(N),R(N),U(N)
      IF(B(1).EQ.0.D0)STOP
      BET=B(1)
      U(1)=R(1)/BET
      DO 11 J=2,N
        GAM(J)=C(J-1)/BET
        BET=B(J)-A(J)*GAM(J)
        IF(BET.EQ.0.D0)STOP
        U(J)=(R(J)-A(J)*U(J-1))/BET
11    CONTINUE
      DO 12 J=N-1,1,-1
        U(J)=U(J)-GAM(J+1)*U(J+1)
12    CONTINUE
      RETURN
      END

      



C----------------------------------------------------------------------
C----------------------------------------------------------------------
C   Figura 16 do artigo da lista 4
C----------------------------------------------------------------------
C----------------------------------------------------------------------
C
      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
C     ---------- ----
C
C Evaluates the algebraic equations or ODE right hand side
C
C Input arguments :
C      NDIM   :   Dimension of the ODE system 
C      U      :   State variables
C      ICP    :   Array indicating the free parameter(s)
C      PAR    :   Equation parameters
C
C Values to be returned :
C      F      :   ODE right hand side values
C
C Normally unused Jacobian arguments : IJAC, DFDU, DFDP (see manual)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(NDIM), PAR(*), F(NDIM), ICP(*)
C
      Da=PAR(1)
      B=16.2
      Beta=3
      X2c=0

      X1=u(1)
      X2=u(2)

c      Qe=(ahA/V/Cp)*(T-Tc)
c      F(1)=(q/V)*(Cf-C)-ak0*dexp(-E/R/T)*C
c      F(2)=(q/V)*(Tf-T)+(aH/Cp)*ak0*dexp(-E/R/T))*C-Qe

      F(1)=-1*X1+Da*(1-X1)*dexp(X2)
      F(2)=-X2+B*Da*(1-X1)*dexp(X2)-Beta*(X2-X2c) 
      
c       write(*,*)'f = ',f
c       read(*,*)

C
      RETURN
      END
C----------------------------------------------------------------------
C
      SUBROUTINE STPNT(NDIM,U,PAR)
C     ---------- -----
C
C Input arguments :
C      NDIM   :   Dimension of the ODE system 
C
C Values to be returned :
C      U      :   A starting solution vector
C      PAR    :   The corresponding equation-parameter values
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(NDIM), PAR(*)
C
C Initialize the equation parameters
       PAR(1)=0.126
C
C Initialize the solution
       U(1)=0.2881
       U(2)=1.1666
C



      RETURN
      END
C----------------------------------------------------------------------
C----------------------------------------------------------------------
C The following subroutines are not used here,
C but they must be supplied as dummy routines
C
      SUBROUTINE BCND 
      RETURN 
      END 
C 
      SUBROUTINE ICND 
      RETURN 
      END 
C 
      SUBROUTINE FOPT 
      RETURN 
      END 
C 
      SUBROUTINE PVLS
      RETURN 
      END 
C----------------------------------------------------------------------
C----------------------------------------------------------------------

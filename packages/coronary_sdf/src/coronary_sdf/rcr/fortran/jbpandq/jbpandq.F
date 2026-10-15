C Adapted from SCAPIS 24f36a5, Akash Sivananthan / Kaihong Wang.
C See ../THIRD_PARTY.md. Fixed-form CFX user routine.
#include "cfx5ext.h"
dllexport(jbpandq)
      SUBROUTINE JBPANDQ(CZ,DZ,IZ,LZ,RZ)
      IMPLICIT NONE
#include "stack_point.h"
#include "cfd_constants.h"
      CHARACTER CZ(*)*1, CRESLT*4, KEY*80
      DOUBLE PRECISION DZ(*)
      INTEGER IZ(*), NOUT, I
      LOGICAL LZ(*), LFIL, FIRST
      EXTERNAL LFIL
      REAL RZ(*), PC, Q, RD, CAP, PV, DT, BETA, TRIAL
      __stack_point__ PTR
      SAVE FIRST
      DATA FIRST /.TRUE./
      CALL USER_PEEKI('NOUTLET',1,NOUT,'STOP',CRESLT,CZ)
      CALL USER_PEEKR('RCR_DT',1,DT,'STOP',CRESLT,RZ)
C Names have ample space; no two-digit outlet limit.
      IF (.NOT.LFIL('/USER_DATA/HPC')) THEN
        CALL PSHDIR('/USER_DATA','STOP',CRESLT)
        CALL MAKDAT('HPC','REAL','STOP',NOUT,PTR,CRESLT)
        CALL MAKDAT('HQ','REAL','STOP',NOUT,PTR,CRESLT)
        CALL MAKDAT('HTRIAL','REAL','STOP',NOUT,PTR,CRESLT)
        CALL MAKDAT('HREADY','REAL','STOP',1,PTR,CRESLT)
        CALL POPDIR('STOP',CRESLT)
      END IF
      DO I=1,NOUT
        IF (FIRST) THEN
          WRITE(KEY,'(A,I0)') 'RCR_PC_',I-1
          CALL USER_PEEKR(TRIM(KEY),1,PC,'STOP',CRESLT,RZ)
          WRITE(KEY,'(A,I0)') 'RCR_Q_',I-1
          CALL USER_PEEKR(TRIM(KEY),1,Q,'STOP',CRESLT,RZ)
          CALL POKER('/USER_DATA/HPC',I,PC,'STOP',CRESLT,RZ)
          CALL POKER('/USER_DATA/HQ',I,Q,'STOP',CRESLT,RZ)
        ELSE
          CALL PEEKR('/USER_DATA/HPC',I,PC,'STOP',CRESLT,RZ)
          CALL PEEKR('/USER_DATA/HQ',I,Q,'STOP',CRESLT,RZ)
        END IF
        WRITE(KEY,'(A,I0)') 'RCR_RD_',I-1
        CALL USER_PEEKR(TRIM(KEY),1,RD,'STOP',CRESLT,RZ)
        WRITE(KEY,'(A,I0)') 'RCR_C_',I-1
        CALL USER_PEEKR(TRIM(KEY),1,CAP,'STOP',CRESLT,RZ)
        WRITE(KEY,'(A,I0)') 'RCR_PV_',I-1
        CALL USER_PEEKR(TRIM(KEY),1,PV,'STOP',CRESLT,RZ)
        BETA=DT/(RD*CAP)
        TRIAL=(PC+DT*Q/CAP+BETA*PV)/(1.0+BETA)
        CALL POKER('/USER_DATA/HTRIAL',I,TRIAL,
     &             'STOP',CRESLT,RZ)
      END DO
      FIRST=.FALSE.
      CALL POKER('/USER_DATA/HREADY',1,1.0,'STOP',CRESLT,RZ)
      END


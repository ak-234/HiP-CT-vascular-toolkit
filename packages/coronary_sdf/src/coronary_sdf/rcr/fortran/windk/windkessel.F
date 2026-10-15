C Adapted from SCAPIS 24f36a5, Akash Sivananthan / Kaihong Wang.
C See ../THIRD_PARTY.md. Fixed-form CFX user routine.
#include "cfx5ext.h"
dllexport(windkessel)
      SUBROUTINE WINDKESSEL
     & (NLOC,NRET,NARG,RET,ARGS,CRESLT,CZ,DZ,IZ,LZ,RZ)
      IMPLICIT NONE
#include "stack_point.h"
#include "cfd_constants.h"
      INTEGER NLOC,NRET,NARG,IZ(*),I,J,K
      REAL RET(NLOC,NRET),ARGS(NLOC,NARG),RZ(*),PC,Q,RP
      CHARACTER CRESLT*(*),CZ(*)*1
      DOUBLE PRECISION DZ(*)
      LOGICAL LZ(*),LFIL
      EXTERNAL LFIL
C WK(index,dt,Rp,Rd,C,initial_pc,initial_q,distal_p).
C Start box already evaluates the trial. CEL calls never advance history.
      IF (NARG.NE.8) THEN
        CRESLT='FAIL'
        RETURN
      END IF
      K=INT(ARGS(1,1))+1
      RP=ARGS(1,3)
      PC=ARGS(1,6)
      Q=ARGS(1,7)
      IF (LFIL('/USER_DATA/HTRIAL')) THEN
        CALL PEEKR('/USER_DATA/HTRIAL',K,PC,'STOP',CRESLT,RZ)
        CALL PEEKR('/USER_DATA/HQ',K,Q,'STOP',CRESLT,RZ)
      END IF
      DO I=1,NLOC
        DO J=1,NRET
          RET(I,J)=PC+RP*Q
        END DO
      END DO
      CRESLT='GOOD'
      END


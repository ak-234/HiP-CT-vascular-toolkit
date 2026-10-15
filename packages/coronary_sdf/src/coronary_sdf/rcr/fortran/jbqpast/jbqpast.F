C Adapted from SCAPIS 24f36a5, Akash Sivananthan / Kaihong Wang.
C See ../THIRD_PARTY.md. Fixed-form CFX user routine.
#include "cfx5ext.h"
dllexport(jbqpast)
      SUBROUTINE JBQPAST(CZ,DZ,IZ,LZ,RZ)
      IMPLICIT NONE
#include "stack_point.h"
#include "parallel_partitioning.h"
#include "cfd_constants.h"
      CHARACTER CZ(*)*1, CRESLT*4, OUTNAME*80
      DOUBLE PRECISION DZ(*)
      INTEGER IZ(*), NOUT, I, STEP, OFFSET, UNIT
      LOGICAL LZ(*), LFIL
      EXTERNAL LFIL
      REAL RZ(*), PC, Q, MASS, DENS, READY, DT
      SAVE STEP
      DATA STEP /0/
      IF (.NOT.LFIL('/USER_DATA/HREADY')) RETURN
      CALL PEEKR('/USER_DATA/HREADY',1,READY,'STOP',CRESLT,RZ)
      IF (READY.LT.0.5) RETURN
      CALL USER_PEEKI('NOUTLET',1,NOUT,'STOP',CRESLT,CZ)
      CALL USER_PEEKI('RCR_STEP',1,OFFSET,'STOP',CRESLT,CZ)
      CALL USER_PEEKR('DENSITY',1,DENS,'STOP',CRESLT,RZ)
      CALL USER_PEEKR('RCR_DT',1,DT,'STOP',CRESLT,RZ)
      STEP=STEP+1
C Append complete checkpoints; a partial final record is never accepted.
      IF (LProcessIO) THEN
        OPEN(NEWUNIT=UNIT,FILE='rcr_state.dat',STATUS='UNKNOWN',
     &       POSITION='APPEND',ACTION='WRITE')
        WRITE(UNIT,*) NOUT,OFFSET+STEP,DT
      END IF
      DO I=1,NOUT
        CALL PEEKR('/USER_DATA/HTRIAL',I,PC,'STOP',CRESLT,RZ)
        CALL POKER('/USER_DATA/HPC',I,PC,'STOP',CRESLT,RZ)
        WRITE(OUTNAME,'(A,I0)') 'Outlet',I-1
C Collective read stays outside the IO-rank guard.
        CALL USER_GET_GVAR('',TRIM(OUTNAME),'massFlow',
     &                    CRESLT,MASS,CZ,DZ,IZ,LZ,RZ)
        Q=-MASS/DENS
        CALL POKER('/USER_DATA/HQ',I,Q,'STOP',CRESLT,RZ)
        IF (LProcessIO) WRITE(UNIT,*) I-1,PC,Q
      END DO
      IF (LProcessIO) THEN
        WRITE(UNIT,*) 'COMMIT',OFFSET+STEP
        CLOSE(UNIT)
      END IF
      CALL POKER('/USER_DATA/HREADY',1,0.0,'STOP',CRESLT,RZ)
      END


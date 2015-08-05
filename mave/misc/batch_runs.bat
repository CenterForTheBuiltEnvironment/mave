echo off
setlocal EnableDelayedExpansion

SET LOGFILE=batch.log

SET F= Main_cbe_01.csv
SET N= 3
SET K= 5 10
SET R= 0.2

FOR %%f in (%F%) DO ( 
    FOR %%n in (%N%) DO ( 
        FOR %%k in (%K%) DO (
            FOR %%r in (%R%) DO (
                Echo params: %%f -n %%n -k %%k -r %%r > %LOGFILE%
                python bpe.py %%f -c 0.1 -v -n 15 -nv %%n -pf 0.33 -k %%k -rs %%r -s >> %LOGFILE%  
            )
        )
    )
)
pause

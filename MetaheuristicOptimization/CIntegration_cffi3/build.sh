#!/bin/bash
#gcc -o generaloptimizer -fPIC -mavx2 -O3 -march=native generaloptimizer.c DISOwithRCF.c -lm -lOpenCL
gcc -shared -o generaloptimizer.so -fPIC -mavx2 -O3 -march=native generaloptimizer.c DISOwithRCF.c GPC.c GWCA.c SPO.c ANS.c BSA.c CS.c DE.c EA.c GA.c HS.c MA.c SFS.c GLS.c ILS.c SA.c TS.c VNS.c AFSA.c ES.c HBO.c KA.c PO.c SO.c ADS.c AAA.c BCO.c BBM.c CFO.c CBO.c CRO.c CA.c CulA.c FA.c FFO.c SDS.c KCA.c LS.c WCA.c ACO.c ALO.c ABC.c BA.c CroSA.c CSO.c CO.c CucS.c DEA.c EHO.c EPO.c FHO.c FlyFO.c FirefA.c HHO.c GWO.c JA.c KHO.c MfA.c MSO.c LOA.c MFO.c OSA.c PuO.c RDA.c SFL.c SMO.c SSA.c SWO.c CMOA.c WO.c WOA.c FSA.c BBO.c ICA.c SEO.c SLCO.c TLBO.c SGO.c DRA.c CSS.c BHA.c CheRO.c EFO.c EVO.c SCA.c LSA.c TEO.c TFWO.c ARFO.c FPA.c POA.c WPA.c AMO.c GlowSO.c DHLO.c HPO.c IWD.c WGMO.c PRO.c SaSA.c SOA.c -lm -lOpenCL -fopenmp -lgomp

#gcc -shared -o generaloptimizer.so -fPIC -mavx2 -O3 -march=native -ftree-vectorize -funroll-loops generaloptimizer.c DISOwithRCF.c GPC.c GWCA.c SPO.c -lm
#GPC.c GWCA.c SPO.c

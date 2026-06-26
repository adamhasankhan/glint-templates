#!/bin/bash
{ time python3 glint_pe.py --npool 10 --event GW231226_101520 --label GW231226_101520-perturber_param_test --perturber_params --rand 1082632 --maxmcmc 6144 --nact 15 --nlive 1024 &> glint_pe.log; } 2> time.log

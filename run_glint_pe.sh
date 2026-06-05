#!/bin/bash
{ time python3 glint_pe.py --npool 20 --event GW231226_101520 --label GW231226_101520-perturber_param_test --perturber_params --rand 1082632 --maxmcmc 8192 --nact 20 --nlive 2048 &> glint_pe.log; } 2> time.log

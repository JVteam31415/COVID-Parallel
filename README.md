# COVID-Parallel
Final project for CSCI 4230

To change params, edit `slurmSpectrum.sh`
arguments are in the following order:
pop_size = atoi(argv[1]);
world_width = atoi(argv[2]);
world_height = atoi(argv[3]);
days = atoi (argv[4]);
infection_radius = atoi(argv[5]);
infect_chance = atof(argv[6]);
symptom_chance = atof(argv[7]);
recovery_time = atoi(argv[8]);
threshold = atoi(argv[9]);
behavior1 = atoi(argv[10]);
behavior2 = atoi(argv[11]);

to compile run `make`

to run on AiMos `sbatch -N 1 --ntasks-per-node=6 --gres=gpu:6 -t 30 ./slurmSpectrum.sh`
with the desired numer of rangs and gpus

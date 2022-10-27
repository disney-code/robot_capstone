# robot_CIFAR10_dataset

conda create -n tf2-gpu tensorflow-gpu==2.2.0
conda activate tf2-gpu


step1 (get baseline model by performing step1)
step1: python train_model.py


step2 (generate fol guided fuzzing adversarial examples)
step2.1: python fol_guided_fuzzing_5mins.py
step2.2: python fol_guided_fuzzing_10mins.py
step2.3: python fol_guided_fuzzing_20mins.py


step4 (generate PGD and FGSM adv examples for testing of robust model later)
step4: python gen_adv.py


step5: (retrain baseline model that you got from step 1 with fol guided fuzzing adv examples)
step 5.1: select_retrain_5min_fol.py
step 5.2: select_retrain_10min_fol.py
step 5.3: select_retrain_20min_fol.py


step6: (evaluate the 3 robust models you get)
step6.1: evaluate.py -> for robust model trained on 5mins worth of fol fuzz adv examples
step6.2: evaluate.py -> for robust model trained on 10mins worth of fol fuzz adv examples
step6.3: evaluate.py -> for robust model trained on 20mins worth of fol fuzz adv examples


